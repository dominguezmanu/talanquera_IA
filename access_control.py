from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from textwrap import wrap


DB_PATH = Path("talanquera_accesos.sqlite3")
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

PLACAS_DEMO = [
    ("P657LCQ", "Manuel Dominguez"),
    ("P191LBM", "Eduardo Rubio"),
    ("P362JFT", "Julio Tzicap"),
    ("P257LKQ", "Ana Castillo"),
    
    
]


def normalizar_placa(placa: str) -> str:
    return "".join(c for c in placa.upper() if c.isalnum())


def formatear_duracion(segundos: int | None) -> str:
    if segundos is None:
        return "--"

    segundos = max(0, int(segundos))
    horas, resto = divmod(segundos, 3600)
    minutos, segundos = divmod(resto, 60)

    if horas:
        return f"{horas}h {minutos:02d}m {segundos:02d}s"
    if minutos:
        return f"{minutos}m {segundos:02d}s"
    return f"{segundos}s"


def _conectar(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    with _conectar(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS placas_autorizadas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                placa TEXT NOT NULL UNIQUE,
                propietario TEXT NOT NULL,
                activo INTEGER NOT NULL DEFAULT 1,
                creado_en TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS movimientos_estacionamiento (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                placa TEXT NOT NULL,
                autorizado INTEGER NOT NULL,
                vehiculo_detectado_entrada INTEGER NOT NULL,
                vehiculo_detectado_salida INTEGER,
                hora_entrada TEXT NOT NULL,
                hora_salida TEXT,
                duracion_segundos INTEGER,
                estado TEXT NOT NULL,
                accion_entrada TEXT NOT NULL,
                accion_salida TEXT,
                origen_entrada TEXT NOT NULL,
                origen_salida TEXT,
                confianza_entrada REAL,
                confianza_salida REAL,
                detalle_entrada TEXT,
                detalle_salida TEXT,
                imagen_entrada_path TEXT,
                imagen_salida_path TEXT
            )
            """
        )
        conn.commit()

    sembrar_placas_demo(db_path)
    sembrar_movimientos_demo(db_path)


def sembrar_placas_demo(db_path: Path = DB_PATH) -> None:
    ahora = datetime.now().strftime(DATETIME_FORMAT)
    with _conectar(db_path) as conn:
        for placa, propietario in PLACAS_DEMO:
            conn.execute(
                """
                INSERT OR IGNORE INTO placas_autorizadas
                    (placa, propietario, activo, creado_en)
                VALUES (?, ?, 1, ?)
                """,
                (normalizar_placa(placa), propietario, ahora),
            )
        conn.commit()


def sembrar_movimientos_demo(db_path: Path = DB_PATH) -> None:
    with _conectar(db_path) as conn:
        existe_demo = conn.execute(
            """
            SELECT 1
            FROM movimientos_estacionamiento
            WHERE origen_entrada = 'demo'
            LIMIT 1
            """
        ).fetchone()

        if existe_demo:
            return

        ahora = datetime.now().replace(microsecond=0)
        registros = [
            ("P123ABC", ahora - timedelta(hours=8, minutes=10), 82),
            ("P456DEF", ahora - timedelta(hours=6, minutes=35), 48),
            ("P789GHI", ahora - timedelta(hours=5, minutes=20), 121),
            ("P321JKL", ahora - timedelta(hours=3, minutes=40), 36),
            ("P654MNO", ahora - timedelta(hours=2, minutes=15), 74),
        ]

        for placa, entrada, minutos in registros:
            salida = entrada + timedelta(minutes=minutos)
            conn.execute(
                """
                INSERT INTO movimientos_estacionamiento (
                    placa, autorizado, vehiculo_detectado_entrada,
                    vehiculo_detectado_salida, hora_entrada, hora_salida,
                    duracion_segundos, estado, accion_entrada, accion_salida,
                    origen_entrada, origen_salida, detalle_entrada,
                    detalle_salida
                )
                VALUES (?, 1, 1, 1, ?, ?, ?, 'SALIO', 'ABRIR', 'ABRIR',
                        'demo', 'demo', 'Registro demo de entrada',
                        'Registro demo de salida')
                """,
                (
                    placa,
                    entrada.strftime(DATETIME_FORMAT),
                    salida.strftime(DATETIME_FORMAT),
                    int((salida - entrada).total_seconds()),
                ),
            )
        conn.commit()


def listar_placas_autorizadas(db_path: Path = DB_PATH) -> list[dict]:
    init_db(db_path)
    with _conectar(db_path) as conn:
        rows = conn.execute(
            """
            SELECT placa, propietario, activo, creado_en
            FROM placas_autorizadas
            ORDER BY placa
            """
        ).fetchall()
    return [dict(row) for row in rows]


def _placa_esta_autorizada_conn(conn: sqlite3.Connection, placa: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM placas_autorizadas
        WHERE placa = ? AND activo = 1
        """,
        (placa,),
    ).fetchone()
    return row is not None


def placa_esta_autorizada(placa: str, db_path: Path = DB_PATH) -> bool:
    init_db(db_path)
    placa = normalizar_placa(placa)
    if not placa:
        return False

    with _conectar(db_path) as conn:
        return _placa_esta_autorizada_conn(conn, placa)


def registrar_lectura(
    placa: str,
    vehiculo_detectado: bool = True,
    origen: str = "manual",
    confianza: float | None = None,
    detalle: str = "",
    imagen_path: str | None = None,
    db_path: Path = DB_PATH,
) -> dict:
    init_db(db_path)
    placa = normalizar_placa(placa)
    fecha_hora = datetime.now().replace(microsecond=0)
    fecha_hora_txt = fecha_hora.strftime(DATETIME_FORMAT)
    vehiculo_ok = bool(vehiculo_detectado)

    with _conectar(db_path) as conn:
        autorizada = bool(
            placa and vehiculo_ok and _placa_esta_autorizada_conn(conn, placa)
        )

        if not autorizada:
            cursor = conn.execute(
                """
                INSERT INTO movimientos_estacionamiento (
                    placa, autorizado, vehiculo_detectado_entrada,
                    hora_entrada, estado, accion_entrada, origen_entrada,
                    confianza_entrada, detalle_entrada, imagen_entrada_path
                )
                VALUES (?, 0, ?, ?, 'RECHAZADO', 'BLOQUEAR', ?, ?, ?, ?)
                """,
                (
                    placa or "SIN_LECTURA",
                    int(vehiculo_ok),
                    fecha_hora_txt,
                    origen,
                    confianza,
                    detalle,
                    imagen_path,
                ),
            )
            conn.commit()
            return {
                "id": cursor.lastrowid,
                "placa": placa or "SIN_LECTURA",
                "tipo_evento": "RECHAZADO",
                "autorizado": False,
                "vehiculo_detectado": vehiculo_ok,
                "accion": "BLOQUEAR",
                "fecha_hora": fecha_hora_txt,
                "duracion_segundos": None,
                "duracion_texto": "--",
            }

        abierta = conn.execute(
            """
            SELECT id, hora_entrada
            FROM movimientos_estacionamiento
            WHERE placa = ? AND autorizado = 1 AND estado = 'ESTACIONADO'
            ORDER BY hora_entrada DESC, id DESC
            LIMIT 1
            """,
            (placa,),
        ).fetchone()

        if abierta:
            entrada = datetime.strptime(abierta["hora_entrada"], DATETIME_FORMAT)
            duracion_segundos = int((fecha_hora - entrada).total_seconds())
            conn.execute(
                """
                UPDATE movimientos_estacionamiento
                SET vehiculo_detectado_salida = ?,
                    hora_salida = ?,
                    duracion_segundos = ?,
                    estado = 'SALIO',
                    accion_salida = 'ABRIR',
                    origen_salida = ?,
                    confianza_salida = ?,
                    detalle_salida = ?,
                    imagen_salida_path = ?
                WHERE id = ?
                """,
                (
                    int(vehiculo_ok),
                    fecha_hora_txt,
                    duracion_segundos,
                    origen,
                    confianza,
                    detalle,
                    imagen_path,
                    abierta["id"],
                ),
            )
            conn.commit()
            return {
                "id": abierta["id"],
                "placa": placa,
                "tipo_evento": "SALIDA",
                "autorizado": True,
                "vehiculo_detectado": vehiculo_ok,
                "accion": "ABRIR",
                "fecha_hora": fecha_hora_txt,
                "duracion_segundos": duracion_segundos,
                "duracion_texto": formatear_duracion(duracion_segundos),
            }

        cursor = conn.execute(
            """
            INSERT INTO movimientos_estacionamiento (
                placa, autorizado, vehiculo_detectado_entrada,
                hora_entrada, estado, accion_entrada, origen_entrada,
                confianza_entrada, detalle_entrada, imagen_entrada_path
            )
            VALUES (?, 1, ?, ?, 'ESTACIONADO', 'ABRIR', ?, ?, ?, ?)
            """,
            (
                placa,
                int(vehiculo_ok),
                fecha_hora_txt,
                origen,
                confianza,
                detalle,
                imagen_path,
            ),
        )
        conn.commit()
        return {
            "id": cursor.lastrowid,
            "placa": placa,
            "tipo_evento": "ENTRADA",
            "autorizado": True,
            "vehiculo_detectado": vehiculo_ok,
            "accion": "ABRIR",
            "fecha_hora": fecha_hora_txt,
            "duracion_segundos": None,
            "duracion_texto": "En parqueo",
        }


def registrar_evento(
    placa: str,
    tipo_evento: str = "LECTURA",
    vehiculo_detectado: bool = True,
    origen: str = "manual",
    confianza: float | None = None,
    detalle: str = "",
    imagen_path: str | None = None,
    db_path: Path = DB_PATH,
) -> dict:
    return registrar_lectura(
        placa=placa,
        vehiculo_detectado=vehiculo_detectado,
        origen=origen,
        confianza=confianza,
        detalle=detalle or f"Lectura recibida como {tipo_evento}",
        imagen_path=imagen_path,
        db_path=db_path,
    )


def _normalizar_inicio(valor: str | None) -> str | None:
    if not valor:
        return None
    valor = valor.strip()
    if len(valor) <= 10:
        return f"{valor} 00:00:00"
    return valor


def _normalizar_fin(valor: str | None) -> str | None:
    if not valor:
        return None
    valor = valor.strip()
    if len(valor) <= 10:
        return f"{valor} 23:59:59"
    return valor


def obtener_registros(
    fecha_inicio: str | None = None,
    fecha_fin: str | None = None,
    limite: int | None = None,
    db_path: Path = DB_PATH,
) -> list[dict]:
    init_db(db_path)
    inicio = _normalizar_inicio(fecha_inicio)
    fin = _normalizar_fin(fecha_fin)
    filtros = []
    params: list[object] = []

    if inicio and fin:
        filtros.append(
            """
            (
                hora_entrada BETWEEN ? AND ?
                OR (hora_salida IS NOT NULL AND hora_salida BETWEEN ? AND ?)
            )
            """
        )
        params.extend([inicio, fin, inicio, fin])
    elif inicio:
        filtros.append(
            """
            (
                hora_entrada >= ?
                OR (hora_salida IS NOT NULL AND hora_salida >= ?)
            )
            """
        )
        params.extend([inicio, inicio])
    elif fin:
        filtros.append(
            """
            (
                hora_entrada <= ?
                OR (hora_salida IS NOT NULL AND hora_salida <= ?)
            )
            """
        )
        params.extend([fin, fin])

    where = f"WHERE {' AND '.join(filtros)}" if filtros else ""
    limit_sql = "LIMIT ?" if limite else ""
    if limite:
        params.append(limite)

    with _conectar(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT id, placa, autorizado, vehiculo_detectado_entrada,
                   vehiculo_detectado_salida, hora_entrada, hora_salida,
                   duracion_segundos, estado, accion_entrada, accion_salida,
                   origen_entrada, origen_salida
            FROM movimientos_estacionamiento
            {where}
            ORDER BY hora_entrada DESC, id DESC
            {limit_sql}
            """,
            params,
        ).fetchall()

    registros = [dict(row) for row in rows]
    ahora = datetime.now().replace(microsecond=0)
    for registro in registros:
        registro["autorizado"] = bool(registro["autorizado"])
        registro["vehiculo_detectado_entrada"] = bool(
            registro["vehiculo_detectado_entrada"]
        )
        if registro["vehiculo_detectado_salida"] is not None:
            registro["vehiculo_detectado_salida"] = bool(
                registro["vehiculo_detectado_salida"]
            )

        duracion = registro["duracion_segundos"]
        if duracion is None and registro["estado"] == "ESTACIONADO":
            entrada = datetime.strptime(registro["hora_entrada"], DATETIME_FORMAT)
            duracion = int((ahora - entrada).total_seconds())
            registro["duracion_actual_segundos"] = duracion
            registro["duracion_texto"] = f"{formatear_duracion(duracion)} (activo)"
        else:
            registro["duracion_actual_segundos"] = duracion
            registro["duracion_texto"] = formatear_duracion(duracion)
    return registros


def obtener_eventos(
    fecha_inicio: str | None = None,
    fecha_fin: str | None = None,
    limite: int | None = None,
    db_path: Path = DB_PATH,
) -> list[dict]:
    return obtener_registros(fecha_inicio, fecha_fin, limite, db_path)


def resumen_registros(registros: list[dict]) -> dict:
    entradas = sum(
        1 for r in registros if r["autorizado"] and r["estado"] in {"ESTACIONADO", "SALIO"}
    )
    salidas = sum(1 for r in registros if r["hora_salida"])
    estacionados = sum(1 for r in registros if r["estado"] == "ESTACIONADO")
    rechazados = sum(1 for r in registros if r["estado"] == "RECHAZADO")
    tiempo_total = sum(
        int(r["duracion_segundos"] or 0)
        for r in registros
        if r["duracion_segundos"] is not None
    )
    return {
        "total": len(registros),
        "entradas": entradas,
        "salidas": salidas,
        "estacionados": estacionados,
        "rechazados": rechazados,
        "tiempo_total": tiempo_total,
    }


def resumen_eventos(eventos: list[dict]) -> dict:
    return resumen_registros(eventos)


def _pdf_escape(texto: object) -> str:
    valor = str(texto)
    return valor.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _partir_lineas(lineas: list[str], ancho: int = 104) -> list[str]:
    resultado: list[str] = []
    for linea in lineas:
        if len(linea) <= ancho:
            resultado.append(linea)
            continue
        resultado.extend(wrap(linea, width=ancho, replace_whitespace=False))
    return resultado


def _escribir_pdf_simple(path: Path, titulo: str, lineas: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lineas = _partir_lineas(lineas)
    lineas_por_pagina = 44
    paginas = [
        lineas[i : i + lineas_por_pagina]
        for i in range(0, max(len(lineas), 1), lineas_por_pagina)
    ]

    objetos: dict[int, bytes] = {}
    objetos[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    kids = []
    siguiente_id = 4
    for indice, pagina_lineas in enumerate(paginas, start=1):
        contenido_id = siguiente_id
        pagina_id = siguiente_id + 1
        siguiente_id += 2
        kids.append(f"{pagina_id} 0 R")

        comandos = [
            "BT",
            "/F1 15 Tf",
            "50 760 Td",
            f"({_pdf_escape(titulo)}) Tj",
            "/F1 9 Tf",
            "0 -20 Td",
            f"(Pagina {indice} de {len(paginas)}) Tj",
            "0 -18 Td",
        ]
        for linea in pagina_lineas:
            comandos.append(f"({_pdf_escape(linea)}) Tj")
            comandos.append("0 -14 Td")
        comandos.append("ET")

        stream = "\n".join(comandos).encode("latin-1", errors="replace")
        objetos[contenido_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"\nendstream"
        )
        objetos[pagina_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {contenido_id} 0 R >>"
        ).encode("latin-1")

    objetos[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objetos[2] = (
        f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    ).encode("latin-1")

    max_id = max(objetos)
    salida = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj_id in range(1, max_id + 1):
        offsets.append(len(salida))
        salida.extend(f"{obj_id} 0 obj\n".encode("latin-1"))
        salida.extend(objetos[obj_id])
        salida.extend(b"\nendobj\n")

    xref_inicio = len(salida)
    salida.extend(f"xref\n0 {max_id + 1}\n".encode("latin-1"))
    salida.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        salida.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    salida.extend(
        f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_inicio}\n%%EOF\n".encode("latin-1")
    )
    path.write_bytes(salida)


def generar_reporte_pdf(
    fecha_inicio: str | None = None,
    fecha_fin: str | None = None,
    salida: Path | str = "reporte_parqueo.pdf",
    db_path: Path = DB_PATH,
) -> Path:
    registros = obtener_registros(fecha_inicio, fecha_fin, db_path=db_path)
    resumen = resumen_registros(registros)
    salida = Path(salida)

    lineas = [
        f"Generado: {datetime.now().strftime(DATETIME_FORMAT)}",
        f"Rango: {fecha_inicio or 'inicio'} a {fecha_fin or 'fin'}",
        "",
        "Resumen operativo",
        f"Registros en el rango: {resumen['total']}",
        f"Entradas autorizadas: {resumen['entradas']}",
        f"Salidas registradas: {resumen['salidas']}",
        f"Vehiculos aun dentro: {resumen['estacionados']}",
        f"Lecturas rechazadas: {resumen['rechazados']}",
        f"Tiempo total completado: {formatear_duracion(resumen['tiempo_total'])}",
        "",
        "Detalle de carros",
        "ID | Placa | Entrada | Salida | Tiempo en parqueo | Estado | Origen entrada/salida",
        "-" * 104,
    ]

    if not registros:
        lineas.append("No hay movimientos para el rango seleccionado.")
    else:
        for registro in registros:
            origen = registro["origen_entrada"]
            if registro["origen_salida"]:
                origen = f"{origen}/{registro['origen_salida']}"
            lineas.append(
                f"{registro['id']} | {registro['placa']} | "
                f"{registro['hora_entrada']} | {registro['hora_salida'] or '--'} | "
                f"{registro['duracion_texto']} | {registro['estado']} | {origen}"
            )

    _escribir_pdf_simple(salida, "Reporte de parqueo - Talanquera inteligente", lineas)
    return salida

