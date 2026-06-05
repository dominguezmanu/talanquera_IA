import cv2
import numpy as np
import pytesseract
import tensorflow as tf
import re
from pathlib import Path

IMG_SIZE = 320
GRID_SIZE = 20

modelo = tf.keras.models.load_model("modelo_heatmap.keras")

CARPETA_IMAGENES = Path("dataset_procesado/test/images")
SALIDA = Path("ocr_multi")
SALIDA.mkdir(exist_ok=True)


def limpiar_texto(texto):
    texto = texto.upper()
    texto = re.sub(r"[^A-Z0-9]", "", texto)
    return texto


def puntuar_texto(texto):
    if len(texto) < 5:
        return 0

    puntos = 0

    if texto.startswith("P"):
        puntos += 3

    if 5 <= len(texto) <= 8:
        puntos += 3

    letras = sum(c.isalpha() for c in texto)
    numeros = sum(c.isdigit() for c in texto)

    if letras >= 1:
        puntos += 1

    if numeros >= 2:
        puntos += 2

    return puntos


def preprocesar(placa):
    gris = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
    gris = cv2.resize(gris, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gris = cv2.GaussianBlur(gris, (3, 3), 0)

    _, th = cv2.threshold(
        gris,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return th


def detectar_centro(img):
    img_red = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_red, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0

    pred = modelo.predict(np.expand_dims(img_norm, axis=0), verbose=0)[0]

    fila, columna = np.unravel_index(
        np.argmax(pred[:, :, 0]),
        (GRID_SIZE, GRID_SIZE)
    )

    x = (columna + 0.5) / GRID_SIZE
    y = (fila + 0.5) / GRID_SIZE

    return x, y


def recortar(img, x, y, escala_w, escala_h, despl_x=0, despl_y=0):
    h, w = img.shape[:2]

    cx = int(x * w + despl_x * w)
    cy = int(y * h + despl_y * h)

    caja_w = int(w * escala_w)
    caja_h = int(h * escala_h)

    x1 = max(0, cx - caja_w // 2)
    y1 = max(0, cy - caja_h // 2)
    x2 = min(w, cx + caja_w // 2)
    y2 = min(h, cy + caja_h // 2)

    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def leer_ocr(img_crop):
    if img_crop is None or img_crop.size == 0:
        return "", None

    h, w = img_crop.shape[:2]

    if h < 10 or w < 10:
        return "", None

    procesada = preprocesar(img_crop)

    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    try:
        texto = pytesseract.image_to_string(procesada, config=config)
        return limpiar_texto(texto), procesada
    except pytesseract.TesseractError:
        return "", procesada
    
for img_path in list(CARPETA_IMAGENES.glob("*.jpg"))[:30]:
    img = cv2.imread(str(img_path))
    x, y = detectar_centro(img)

    candidatos = [
        (0.16, 0.07, 0, 0),
        (0.18, 0.08, 0, 0),
        (0.22, 0.09, 0, 0),
        (0.20, 0.12, 0, 0),

        # pequeños desplazamientos porque el heatmap a veces cae arriba/abajo
        (0.18, 0.08, 0, -0.03),
        (0.18, 0.08, 0, 0.03),
        (0.18, 0.08, -0.03, 0),
        (0.18, 0.08, 0.03, 0),
    ]

    mejor_texto = ""
    mejor_score = -1
    mejor_caja = None
    mejor_crop = None
    mejor_proc = None

    for escala_w, escala_h, dx, dy in candidatos:
        crop, caja = recortar(img, x, y, escala_w, escala_h, dx, dy)

        if crop.size == 0:
            continue

        texto, procesada = leer_ocr(crop)
        score = puntuar_texto(texto)

        if score > mejor_score:
            mejor_score = score
            mejor_texto = texto
            mejor_caja = caja
            mejor_crop = crop
            mejor_proc = procesada

    resultado = img.copy()

    if mejor_caja:
        x1, y1, x2, y2 = mejor_caja
        cv2.rectangle(resultado, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            resultado,
            mejor_texto,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imwrite(str(SALIDA / f"res_{img_path.name}"), resultado)

    if mejor_crop is not None:
        cv2.imwrite(str(SALIDA / f"crop_{img_path.name}"), mejor_crop)

    if mejor_proc is not None:
        cv2.imwrite(str(SALIDA / f"proc_{img_path.name}"), mejor_proc)

    print(img_path.name, "=>", mejor_texto, "score:", mejor_score)

print("Listo. Revisa la carpeta ocr_multi")