"""Exportacion de resultados a Excel."""

import json
from datetime import datetime

import pandas as pd

from .display import _momentos, _preguntas, _score_y_razon

__all__ = ["exportar_resultados_excel"]


def exportar_resultados_excel(resultados_totales, character, nombre_archivo=None):
    """
    Guarda TODO lo devuelto por el modelo en un Excel de 4 hojas:
      - Detalle    : una fila por (dimension, momento, pregunta) con score y reasoning
      - Promedios  : promedio por dimension y momento (ignora los N/A)
      - Analisis   : analysis + conclusion + book de cada dimension
      - JSON crudo : el JSON completo tal cual lo devolvio el modelo (respaldo)
    Devuelve el nombre del archivo.
    """
    if nombre_archivo is None:
        sello = datetime.now().strftime("%Y%m%d_%H%M")
        nombre_archivo = f"Analisis_PTMV_{character}_{sello}.xlsx"

    filas_detalle, filas_prom, filas_texto, filas_json = [], [], [], []

    for dim, res in resultados_totales.items():
        if not isinstance(res, dict) or "error" in res:
            filas_json.append({"Dimension": dim, "JSON": str(res)})
            continue

        scores = res.get("scores", {})
        claves, nombres = _momentos(scores)
        preguntas = _preguntas(scores, claves)

        filas_texto.append({
            "Dimension":  dim,
            "Personaje":  res.get("character", character),
            "Obra":       res.get("book", ""),
            "Analysis":   res.get("analysis", ""),
            "Conclusion": res.get("conclusion", ""),
        })
        filas_json.append({"Dimension": dim,
                           "JSON": json.dumps(res, ensure_ascii=False, indent=2)})

        fila_prom = {"Dimension": dim}
        for clave, nombre in zip(claves, nombres):
            numericos = []
            for q in preguntas:
                score, razon = _score_y_razon(scores.get(clave, {}).get(q, {}))
                filas_detalle.append({
                    "Dimension": dim, "Momento": nombre, "Pregunta": q,
                    "Score": score, "Reasoning": razon,
                })
                try:
                    numericos.append(float(score))
                except (TypeError, ValueError):
                    pass  # los N/A no entran al promedio

            fila_prom[nombre] = round(sum(numericos) / len(numericos), 2) if numericos else "N/A"
        filas_prom.append(fila_prom)

    with pd.ExcelWriter(nombre_archivo, engine="openpyxl") as writer:
        pd.DataFrame(filas_detalle).to_excel(writer, sheet_name="Detalle", index=False)
        pd.DataFrame(filas_prom).to_excel(writer, sheet_name="Promedios", index=False)
        pd.DataFrame(filas_texto).to_excel(writer, sheet_name="Analisis", index=False)
        pd.DataFrame(filas_json).to_excel(writer, sheet_name="JSON crudo", index=False)

        # ancho de columnas + wrap para que el reasoning se lea
        from openpyxl.styles import Alignment
        for hoja in writer.book.worksheets:
            for col in hoja.columns:
                letra = col[0].column_letter
                largo = max((len(str(c.value)) for c in col if c.value), default=10)
                hoja.column_dimensions[letra].width = min(max(12, largo + 2), 60)
            for fila in hoja.iter_rows(min_row=2):
                for celda in fila:
                    celda.alignment = Alignment(wrap_text=True, vertical="top")

    print(f"Excel guardado: {nombre_archivo}")
    return nombre_archivo
