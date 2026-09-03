"""Presentacion de resultados en el notebook (tablas y promedios)."""

import pandas as pd

__all__ = ["mostrar_resultados_bonitos", "mostrar_resumen_global", "promedios_por_momento"]


def _display(obj):
    try:
        from IPython.display import display
        display(obj)
    except ImportError:
        print(obj)


def _md(texto):
    try:
        from IPython.display import display, Markdown
        display(Markdown(texto))
    except ImportError:
        print(texto)


def _momentos(scores):
    """Claves de momento presentes en el resultado, con su nombre legible."""
    claves = list(scores.keys())
    nombres = [c.replace("Moment_", "Momento ").replace("_", " ") for c in claves]
    return claves, nombres


def _preguntas(scores, claves):
    for c in claves:
        bloque = scores.get(c)
        if isinstance(bloque, dict) and bloque:
            return list(bloque.keys())
    return []


def _score_y_razon(q_data):
    if isinstance(q_data, dict):
        return q_data.get("score", "N/A"), q_data.get("reasoning", "")
    return q_data, ""


def promedios_por_momento(scores):
    """dict {nombre_momento: promedio o 'N/A'}, ignorando los N/A."""
    claves, nombres = _momentos(scores)
    preguntas = _preguntas(scores, claves)
    salida = {}
    for clave, nombre in zip(claves, nombres):
        numericos = []
        for q in preguntas:
            score, _ = _score_y_razon(scores.get(clave, {}).get(q, {}))
            try:
                numericos.append(float(score))
            except (TypeError, ValueError):
                pass
        salida[nombre] = round(sum(numericos) / len(numericos), 2) if numericos else "N/A"
    return salida


def mostrar_resultados_bonitos(resultado_json, completo=False):
    """
    Muestra los resultados de UNA dimension.

    completo=False (vista compacta, la que se usa al analizar las 6 dimensiones):
        tabla de puntajes + explicaciones de los N/A + promedios.
    completo=True (al analizar una sola dimension):
        ademas personaje y obra, el reasoning de TODAS las preguntas agrupado por
        momento, y los campos `analysis` y `conclusion` que devolvio el modelo.
    """
    if not isinstance(resultado_json, dict) or "error" in resultado_json:
        _md(f"**Error en esta dimension:** `{resultado_json}`")
        return

    dim = resultado_json.get("dimension_analyzed", "Desconocida")
    scores = resultado_json.get("scores", {})
    claves, nombres = _momentos(scores)
    preguntas = _preguntas(scores, claves)

    _md(f"## DIMENSION: **{dim}**")

    if completo:
        personaje = resultado_json.get("character", "")
        obra = resultado_json.get("book", "")
        cabecera = [x for x in (f"**Personaje:** {personaje}" if personaje else "",
                                f"**Obra/Fragmento:** {obra}" if obra else "") if x]
        if cabecera:
            _md("  \n".join(cabecera))

    data_df = {n: [] for n in nombres}
    na_explanations = []

    for q in preguntas:
        for clave, nombre in zip(claves, nombres):
            score, reasoning = _score_y_razon(scores.get(clave, {}).get(q, {}))
            data_df[nombre].append(score)
            try:
                float(score)
            except (TypeError, ValueError):
                na_explanations.append(
                    f"- **{nombre} - {q}**: {reasoning or 'Sin justificacion'}"
                )

    _display(pd.DataFrame(data_df, index=preguntas))

    if completo:
        # Todos los razonamientos, agrupados por momento
        _md("### RAZONAMIENTO POR PREGUNTA")
        for clave, nombre in zip(claves, nombres):
            _md(f"#### {nombre}")
            for q in preguntas:
                score, reasoning = _score_y_razon(scores.get(clave, {}).get(q, {}))
                _md(f"- **{q}** — puntaje `{score}`  \n  {reasoning or 'Sin justificacion'}")
    elif na_explanations:
        _md("### Explicaciones de puntajes N/A:")
        for exp in na_explanations:
            _md(exp)

    _md("### PROMEDIOS POR MOMENTO")
    for nombre, prom in promedios_por_momento(scores).items():
        if isinstance(prom, str):
            _md(f"- **{nombre}**: N/A (Sin valores numericos)")
        else:
            _md(f"- **{nombre}**: {prom:.1f}")

    if completo:
        analisis = resultado_json.get("analysis", "")
        conclusion = resultado_json.get("conclusion", "")
        if analisis:
            _md(f"### ANALISIS\n\n{analisis}")
        if conclusion:
            _md(f"### CONCLUSION\n\n{conclusion}")

    _md("---")


def mostrar_resumen_global(resultados_totales):
    """Tabla final: promedio de cada dimension en cada momento."""
    _md("## Promedios por dimension por momento")

    filas, indices = [], []
    for dim, res in resultados_totales.items():
        if not isinstance(res, dict) or "error" in res:
            continue
        indices.append(dim)
        filas.append(promedios_por_momento(res.get("scores", {})))

    df = pd.DataFrame(filas, index=indices)
    df.index.name = "Dimension"
    _display(df)
    return df
