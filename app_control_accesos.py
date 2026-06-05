from __future__ import annotations

import os
import re
import time
import threading
import tkinter as tk
from collections import Counter
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import easyocr
import numpy as np
import serial
import tensorflow as tf
from PIL import Image, ImageTk

from access_control import (
    generar_reporte_pdf,
    init_db,
    listar_placas_autorizadas,
    normalizar_placa,
    obtener_registros,
    registrar_lectura,
)

# global

MODELO_TENSORFLOW = "modelo_heatmap.keras"

IMG_SIZE = 320
GRID_SIZE = 20
CAMARA_ID = "/dev/video2"

PUERTO_SERIAL = "/dev/ttyACM0"
BAUDRATE = 9600
ENVIAR_SERIAL = True

OCR_CADA_N_FRAMES = 30
SCORE_MINIMO_POPUP = 7
COOLDOWN_POPUP_SEGUNDOS = 8
COOLDOWN_SEGUNDOS = 8


class ControlAccesosApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Talanquera inteligente - TensorFlow")
        self.geometry("1280x760")
        self.minsize(1100, 680)

        init_db()

        self.placa_var = tk.StringVar()
        self.vehiculo_var = tk.BooleanVar(value=True)
        self.inicio_var = tk.StringVar()
        self.fin_var = tk.StringVar()
        self.estado_var = tk.StringVar(
            value="Sistema listo. Esperando detección de placa."
        )

        self.modelo = tf.keras.models.load_model(MODELO_TENSORFLOW, compile=False)
        self.reader = easyocr.Reader(["en"], gpu=False)

        self.serial_conn = None
        self.iniciar_serial()

        self.cap = cv2.VideoCapture(CAMARA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        self.frame_contador = 0
        self.procesando_ia = False
        self.popup_abierto = False
        self.ultimo_popup = 0.0

        self.ultimo_envio: dict[str, float] = {}
        self.texto_actual = ""
        self.caja_actual = None

        self._crear_widgets()
        self.refrescar()
        self.actualizar_video()

        self.protocol("WM_DELETE_WINDOW", self.cerrar)

    # serial

    def iniciar_serial(self) -> None:
        if not ENVIAR_SERIAL:
            return

        try:
            self.serial_conn = serial.Serial(PUERTO_SERIAL, BAUDRATE, timeout=1)
            time.sleep(2)
            print(f"Serial conectado en {PUERTO_SERIAL}")
        except Exception as e:
            self.serial_conn = None
            print(f"No se pudo abrir serial: {e}")

    def enviar_s_serial(self) -> None:
        if self.serial_conn:
            try:
                self.serial_conn.write(b"s")
                
                print("Serial enviado: s")
            except Exception as e:
                print(f"Error enviando serial: {e}")

    # OCS

    def limpiar_texto(self, texto: str) -> str:
        texto = texto.upper()
        return re.sub(r"[^A-Z0-9]", "", texto)

    def puntuar_texto(self, texto: str) -> int:
        if len(texto) < 5:
            return 0

        score = 0

        if texto.startswith("P"):
            score += 3

        if 5 <= len(texto) <= 8:
            score += 3

        if sum(c.isdigit() for c in texto) >= 2:
            score += 2

        if sum(c.isalpha() for c in texto) >= 2:
            score += 2

        return score

    def leer_easyocr(self, crop):
        if crop is None or crop.size == 0:
            return "", 0

        try:
            gris = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gris = cv2.resize(
                gris,
                None,
                fx=4,
                fy=4,
                interpolation=cv2.INTER_CUBIC,
            )

            versiones = [gris]

            _, otsu = cv2.threshold(
                gris,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

            versiones.append(otsu)
            versiones.append(cv2.bitwise_not(otsu))

            mejor_texto = ""
            mejor_score = 0

            for img_proc in versiones:
                resultados = self.reader.readtext(
                    img_proc,
                    detail=0,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                )

                texto = self.limpiar_texto("".join(resultados))
                score = self.puntuar_texto(texto)

                if score > mejor_score:
                    mejor_texto = texto
                    mejor_score = score

            return mejor_texto, mejor_score

        except Exception:
            return "", 0

    # Tensorflow

    def detectar_centro_tensorflow(self, frame):
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        entrada = np.expand_dims(img, axis=0)
        pred = self.modelo(entrada, training=False).numpy()[0]

        fila, columna = np.unravel_index(
            np.argmax(pred[:, :, 0]),
            (GRID_SIZE, GRID_SIZE),
        )

        x = (columna + 0.5) / GRID_SIZE
        y = (fila + 0.5) / GRID_SIZE

        return x, y

    def recortes_candidatos(self, frame, x, y):
        h, w = frame.shape[:2]

        cx = int(x * w)
        cy = int(y * h)

        candidatos = [
            (0.16, 0.07, 0, 0),
            (0.18, 0.08, 0, 0),
            (0.22, 0.09, 0, 0),
            (0.20, 0.12, 0, 0),
            (0.18, 0.08, 0, -0.03),
            (0.18, 0.08, 0, 0.03),
            (0.18, 0.08, -0.03, 0),
            (0.18, 0.08, 0.03, 0),
        ]

        recortes = []

        for escala_w, escala_h, dx, dy in candidatos:
            centro_x = int(cx + dx * w)
            centro_y = int(cy + dy * h)

            caja_w = int(w * escala_w)
            caja_h = int(h * escala_h)

            x1 = max(0, centro_x - caja_w // 2)
            y1 = max(0, centro_y - caja_h // 2)
            x2 = min(w, centro_x + caja_w // 2)
            y2 = min(h, centro_y + caja_h // 2)

            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:
                recortes.append((crop, (x1, y1, x2, y2)))

        return recortes

    def leer_mejor_recorte(self, frame, x, y):
        mejor_texto = ""
        mejor_score = 0
        mejor_caja = None

        for crop, caja in self.recortes_candidatos(frame, x, y):
            texto, score = self.leer_easyocr(crop)

            if score > mejor_score:
                mejor_texto = texto
                mejor_score = score
                mejor_caja = caja

        return mejor_texto, mejor_score, mejor_caja

    # Control de acceso

    def placa_autorizada(self, placa: str) -> bool:
        placas = {
            normalizar_placa(p["placa"])
            for p in listar_placas_autorizadas()
            if p["activo"]
        }

        return normalizar_placa(placa) in placas

    def procesar_placa_detectada(self, placa: str) -> None:
        placa = normalizar_placa(placa)

        if not placa:
            messagebox.showwarning("Placa requerida", "No se ingresó una placa válida.")
            return

        if not self.placa_autorizada(placa):
            self.estado_var.set(f"Placa detectada no autorizada: {placa}")
            messagebox.showwarning("No autorizada", f"La placa {placa} no está autorizada.")
            return

        ahora = time.time()

        if placa in self.ultimo_envio:
            if ahora - self.ultimo_envio[placa] < COOLDOWN_SEGUNDOS:
                return

        evento = registrar_lectura(
            placa=placa,
            vehiculo_detectado=True,
            origen="camara_tensorflow_popup",
            detalle="Lectura confirmada desde popup con TensorFlow + EasyOCR",
        )

        if evento["tipo_evento"] in ["ENTRADA", "SALIDA"]:
            self.enviar_s_serial()
            self.ultimo_envio[placa] = ahora

        if evento["tipo_evento"] == "ENTRADA":
            self.estado_var.set(f"Entrada registrada: {placa}. Serial enviado.")
        elif evento["tipo_evento"] == "SALIDA":
            self.estado_var.set(
                f"Salida registrada: {placa}. Tiempo: {evento['duracion_texto']}. Serial enviado."
            )
        else:
            self.estado_var.set(f"Lectura rechazada: {placa}")

        self.refrescar()

    def mostrar_popup_registro(self, placa_detectada: str) -> None:
        if self.popup_abierto:
            return

        self.popup_abierto = True

        ventana = tk.Toplevel(self)
        ventana.title("Placa detectada")
        ventana.geometry("360x210")
        ventana.resizable(False, False)
        ventana.grab_set()

        placa_var = tk.StringVar(value=normalizar_placa(placa_detectada))

        ttk.Label(
            ventana,
            text="Se detectó una posible placa",
            font=("Segoe UI", 10, "bold"),
        ).pack(pady=(14, 8))

        ttk.Label(ventana, text="Confirma o corrige la placa:").pack()

        entry = ttk.Entry(
            ventana,
            textvariable=placa_var,
            font=("Segoe UI", 14),
            justify="center",
        )
        entry.pack(pady=10)
        entry.focus()
        entry.select_range(0, tk.END)

        botones = ttk.Frame(ventana)
        botones.pack(pady=10)

        def aceptar():
            placa = placa_var.get()
            ventana.destroy()
            self.popup_abierto = False
            self.procesar_placa_detectada(placa)

        def cancelar():
            ventana.destroy()
            self.popup_abierto = False
            self.estado_var.set("Lectura cancelada por el usuario.")

        ttk.Button(botones, text="Registrar", command=aceptar).pack(side="left", padx=8)
        ttk.Button(botones, text="Cancelar", command=cancelar).pack(side="left", padx=8)

        ventana.protocol("WM_DELETE_WINDOW", cancelar)

    # video

    def procesar_ia_en_hilo(self, frame) -> None:
        try:
            x, y = self.detectar_centro_tensorflow(frame)
            texto, score, caja = self.leer_mejor_recorte(frame, x, y)

            self.caja_actual = caja

            if score >= SCORE_MINIMO_POPUP:
                texto = normalizar_placa(texto)
                self.texto_actual = texto

                ahora = time.time()

                if not self.popup_abierto and ahora - self.ultimo_popup >= COOLDOWN_POPUP_SEGUNDOS:
                    self.ultimo_popup = ahora
                    self.after(0, self.mostrar_popup_registro, texto)

        except Exception as e:
            print("Error IA:", e)

        finally:
            self.procesando_ia = False

    def actualizar_video(self) -> None:
        ok, frame = self.cap.read()

        if ok:
            self.frame_contador += 1

            if self.frame_contador % OCR_CADA_N_FRAMES == 0 and not self.procesando_ia:
                self.procesando_ia = True
                frame_copia = frame.copy()

                hilo = threading.Thread(
                    target=self.procesar_ia_en_hilo,
                    args=(frame_copia,),
                    daemon=True,
                )
                hilo.start()

            if self.caja_actual:
                x1, y1, x2, y2 = self.caja_actual
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    self.texto_actual,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                "TensorFlow + EasyOCR",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (520, 320))

            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.after(30, self.actualizar_video)

    # interfaz

    def _crear_widgets(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        titulo = ttk.Label(
            root,
            text="Control de parqueo - TensorFlow",
            font=("Segoe UI", 16, "bold"),
        )
        titulo.pack(anchor="w")

        cuerpo = ttk.Frame(root)
        cuerpo.pack(fill="both", expand=True, pady=(12, 0))

        panel = ttk.LabelFrame(cuerpo, text="Lectura de placa", padding=12)
        panel.pack(side="left", fill="y", padx=(0, 12))

        ttk.Label(panel, text="Video de cámara").pack(anchor="w")
        self.video_label = ttk.Label(panel)
        self.video_label.pack(fill="x", pady=(0, 12))

        ttk.Label(panel, text="Placa detectada/manual").pack(anchor="w")
        placa_entry = ttk.Entry(panel, textvariable=self.placa_var, width=24)
        placa_entry.pack(fill="x", pady=(0, 8))
        placa_entry.focus()

        ttk.Checkbutton(
            panel,
            text="Vehiculo detectado por camara",
            variable=self.vehiculo_var,
        ).pack(anchor="w", pady=(0, 12))

        ttk.Button(
            panel,
            text="Registrar lectura manual",
            command=self.registrar_manual,
        ).pack(fill="x", pady=(0, 8))

        ttk.Button(
            panel,
            text="Refrescar",
            command=self.refrescar,
        ).pack(fill="x", pady=(0, 16))

        ttk.Separator(panel).pack(fill="x", pady=8)

        ttk.Label(panel, text="Reporte de parqueo").pack(anchor="w", pady=(4, 6))

        ttk.Label(panel, text="Inicio (YYYY-MM-DD o YYYY-MM-DD HH:MM:SS)").pack(anchor="w")
        ttk.Entry(panel, textvariable=self.inicio_var, width=28).pack(fill="x", pady=(0, 8))

        ttk.Label(panel, text="Fin (YYYY-MM-DD o YYYY-MM-DD HH:MM:SS)").pack(anchor="w")
        ttk.Entry(panel, textvariable=self.fin_var, width=28).pack(fill="x", pady=(0, 8))

        ttk.Button(panel, text="Generar PDF", command=self.generar_pdf).pack(fill="x")

        ttk.Label(panel, textvariable=self.estado_var, wraplength=480).pack(
            fill="x",
            pady=(18, 0),
        )

        tablas = ttk.Frame(cuerpo)
        tablas.pack(side="left", fill="both", expand=True)

        placas_box = ttk.LabelFrame(tablas, text="Placas autorizadas", padding=8)
        placas_box.pack(fill="x", pady=(0, 12))

        self.placas_tree = ttk.Treeview(
            placas_box,
            columns=("placa", "propietario", "activo"),
            show="headings",
            height=6,
        )

        self.placas_tree.heading("placa", text="Placa")
        self.placas_tree.heading("propietario", text="Propietario")
        self.placas_tree.heading("activo", text="Activo")

        self.placas_tree.column("placa", width=120, anchor="center")
        self.placas_tree.column("propietario", width=260)
        self.placas_tree.column("activo", width=80, anchor="center")
        self.placas_tree.pack(fill="x")

        movimientos_box = ttk.LabelFrame(tablas, text="Movimientos de parqueo", padding=8)
        movimientos_box.pack(fill="both", expand=True)

        columnas = (
            "id",
            "placa",
            "entrada",
            "salida",
            "duracion",
            "estado",
            "autorizado",
            "vehiculo",
            "origen",
        )

        self.movimientos_tree = ttk.Treeview(
            movimientos_box,
            columns=columnas,
            show="headings",
        )

        encabezados = {
            "id": "ID",
            "placa": "Placa",
            "entrada": "Entrada",
            "salida": "Salida",
            "duracion": "Tiempo",
            "estado": "Estado",
            "autorizado": "Aut.",
            "vehiculo": "Veh.",
            "origen": "Origen",
        }

        anchos = {
            "id": 50,
            "placa": 90,
            "entrada": 150,
            "salida": 150,
            "duracion": 130,
            "estado": 105,
            "autorizado": 55,
            "vehiculo": 55,
            "origen": 150,
        }

        for col in columnas:
            self.movimientos_tree.heading(col, text=encabezados[col])
            self.movimientos_tree.column(col, width=anchos[col], anchor="center")

        yscroll = ttk.Scrollbar(
            movimientos_box,
            orient="vertical",
            command=self.movimientos_tree.yview,
        )

        self.movimientos_tree.configure(yscrollcommand=yscroll.set)
        self.movimientos_tree.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

    # botones

    def registrar_manual(self) -> None:
        placa = normalizar_placa(self.placa_var.get())

        if not placa:
            messagebox.showwarning("Placa requerida", "Ingresa una placa antes de registrar.")
            return

        evento = registrar_lectura(
            placa=placa,
            vehiculo_detectado=self.vehiculo_var.get(),
            origen="interfaz",
            detalle="Lectura manual desde Tkinter",
        )

        if evento["tipo_evento"] == "ENTRADA":
            mensaje = f"Entrada registrada para {evento['placa']}."
        elif evento["tipo_evento"] == "SALIDA":
            mensaje = (
                f"Salida registrada para {evento['placa']}. "
                f"Tiempo: {evento['duracion_texto']}."
            )
        else:
            mensaje = f"Lectura rechazada para {evento['placa']}."

        self.estado_var.set(mensaje)
        self.placa_var.set("")
        self.refrescar()

    def refrescar(self) -> None:
        for item in self.placas_tree.get_children():
            self.placas_tree.delete(item)

        for placa in listar_placas_autorizadas():
            self.placas_tree.insert(
                "",
                "end",
                values=(
                    placa["placa"],
                    placa["propietario"],
                    "SI" if placa["activo"] else "NO",
                ),
            )

        for item in self.movimientos_tree.get_children():
            self.movimientos_tree.delete(item)

        for registro in obtener_registros(limite=100):
            origen = registro["origen_entrada"]

            if registro["origen_salida"]:
                origen = f"{origen}/{registro['origen_salida']}"

            self.movimientos_tree.insert(
                "",
                "end",
                values=(
                    registro["id"],
                    registro["placa"],
                    registro["hora_entrada"],
                    registro["hora_salida"] or "--",
                    registro["duracion_texto"],
                    registro["estado"],
                    "SI" if registro["autorizado"] else "NO",
                    "SI" if registro["vehiculo_detectado_entrada"] else "NO",
                    origen,
                ),
            )

    def generar_pdf(self) -> None:
        ruta = filedialog.asksaveasfilename(
            title="Guardar reporte PDF",
            defaultextension=".pdf",
            filetypes=(("PDF", "*.pdf"),),
            initialfile="reporte_parqueo.pdf",
        )

        if not ruta:
            return

        salida = generar_reporte_pdf(
            fecha_inicio=self.inicio_var.get().strip() or None,
            fecha_fin=self.fin_var.get().strip() or None,
            salida=Path(ruta),
        )

        self.estado_var.set(f"Reporte generado: {salida}")
        messagebox.showinfo("Reporte generado", f"Se genero el reporte:\n{salida}")

    def cerrar(self) -> None:
        if self.cap:
            self.cap.release()

        if self.serial_conn:
            self.serial_conn.close()

        self.destroy()


def main() -> None:
    app = ControlAccesosApp()
    app.mainloop()


if __name__ == "__main__":
    main()