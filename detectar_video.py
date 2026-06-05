import cv2
import numpy as np
import tensorflow as tf
import pytesseract
import re
from collections import Counter
from datetime import datetime

from access_control import registrar_lectura

IMG_SIZE = 320
GRID_SIZE = 20

modelo = tf.keras.models.load_model("modelo_heatmap.keras")

lecturas = []
ultimo_registro_placa = ""
ultimo_registro_tiempo = datetime.min
ultimo_estado = "SIN EVENTO"
COOLDOWN_SEGUNDOS = 12


def limpiar_texto(texto):
    texto = texto.upper()
    return re.sub(r"[^A-Z0-9]", "", texto)


def puntuar_texto(texto):
    if len(texto) < 5:
        return 0

    score = 0

    if texto.startswith("P"):
        score += 3

    if 5 <= len(texto) <= 8:
        score += 3

    score += sum(c.isdigit() for c in texto) // 2
    score += sum(c.isalpha() for c in texto) // 2

    return score


def detectar_centro(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    pred = modelo.predict(
        np.expand_dims(img, axis=0),
        verbose=0
    )[0]

    fila, columna = np.unravel_index(
        np.argmax(pred[:, :, 0]),
        (GRID_SIZE, GRID_SIZE)
    )

    x = (columna + 0.5) / GRID_SIZE
    y = (fila + 0.5) / GRID_SIZE

    return x, y


def preprocesar(crop):
    gris = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    gris = cv2.resize(
        gris,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    gris = cv2.GaussianBlur(gris, (3, 3), 0)

    _, th = cv2.threshold(
        gris,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return th


def leer_crop(crop):
    if crop.size == 0:
        return ""

    try:
        proc = preprocesar(crop)

        texto = pytesseract.image_to_string(
            proc,
            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        return limpiar_texto(texto)

    except:
        return ""


def validar_vehiculo_por_contexto(frame, caja):
    x1, y1, x2, y2 = caja
    h, w = frame.shape[:2]
    placa_w = max(1, x2 - x1)
    placa_h = max(1, y2 - y1)

    rx1 = max(0, x1 - int(placa_w * 2.0))
    ry1 = max(0, y1 - int(placa_h * 2.5))
    rx2 = min(w, x2 + int(placa_w * 2.0))
    ry2 = min(h, y2 + int(placa_h * 3.0))

    roi = frame[ry1:ry2, rx1:rx2]

    if roi.size == 0:
        return False

    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(gris, 60, 160)

    densidad_bordes = cv2.countNonZero(bordes) / float(bordes.size)
    proporcion_area = (roi.shape[0] * roi.shape[1]) / float(h * w)

    return densidad_bordes > 0.015 and proporcion_area > 0.04


camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("No se pudo abrir la cámara")
    exit()

contador_frames = 0

while True:

    ok, frame = camara.read()

    if not ok:
        break

    contador_frames += 1

    texto_final = ""
    vehiculo_actual = False
    deteccion_actual = False

    if contador_frames % 5 == 0:

        x, y = detectar_centro(frame)

        h, w = frame.shape[:2]

        cx = int(x * w)
        cy = int(y * h)

        candidatos = [
            (0.18, 0.08),
            (0.20, 0.10),
            (0.22, 0.10),
            (0.25, 0.12)
        ]

        mejor_score = -1
        mejor_texto = ""

        mejor_caja = None

        for escala_w, escala_h in candidatos:

            caja_w = int(w * escala_w)
            caja_h = int(h * escala_h)

            x1 = max(0, cx - caja_w // 2)
            y1 = max(0, cy - caja_h // 2)
            x2 = min(w, cx + caja_w // 2)
            y2 = min(h, cy + caja_h // 2)

            crop = frame[y1:y2, x1:x2]

            texto = leer_crop(crop)

            score = puntuar_texto(texto)

            if score > mejor_score:
                mejor_score = score
                mejor_texto = texto
                mejor_caja = (x1, y1, x2, y2)

        if mejor_texto:

            lecturas.append(mejor_texto)

            if len(lecturas) > 20:
                lecturas.pop(0)

        if mejor_caja:

            x1, y1, x2, y2 = mejor_caja
            vehiculo_actual = validar_vehiculo_por_contexto(frame, mejor_caja)
            deteccion_actual = bool(mejor_texto)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    placa_estable = ""

    if lecturas:

        placa_estable = Counter(lecturas).most_common(1)[0][0]

    if placa_estable and deteccion_actual:
        ahora = datetime.now()
        debe_registrar = (
            placa_estable != ultimo_registro_placa
            or (ahora - ultimo_registro_tiempo).total_seconds() >= COOLDOWN_SEGUNDOS
        )

        if debe_registrar:
            evento = registrar_lectura(
                placa=placa_estable,
                vehiculo_detectado=vehiculo_actual,
                origen="detectar_video_heatmap",
                detalle="Lectura estable con modelo heatmap + Tesseract",
            )
            ultimo_registro_placa = placa_estable
            ultimo_registro_tiempo = ahora
            ultimo_estado = f"{evento['tipo_evento']} {evento['accion']}"

    cv2.putText(
        frame,
        f"OCR: {placa_estable}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Acceso: {ultimo_estado}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if "ABRIR" in ultimo_estado else (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Vehiculo: {'SI' if vehiculo_actual else 'NO'}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("Detector de Placas", frame)

    tecla = cv2.waitKey(1)

    if tecla == 27:
        break

camara.release()
cv2.destroyAllWindows()
