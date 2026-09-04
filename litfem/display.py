"""Vistas crudas sobre el formato largo (lo numerico "oficial" esta en `agregacion`)."""

import numpy as np
import pandas as pd

from .comun import _md, _display

__all__ = ["tabla_cruda", "mostrar_razonamientos", "mostrar_na",
           "residuos_posicion", "efecto_posicion"]


def tabla_cruda(largo, dimension, iteracion=None):
    """Tabla preguntas x unidades con los puntajes tal como los devolvio el modelo."""
    sub = largo[largo["dimension"] == dimension]
    if iteracion is not None:
        sub = sub[sub["iteracion"] == iteracion]
    df = sub.pivot_table(index="pregunta", columns="fuente", values="score_txt",
                         aggfunc="first")
    _md(f"## Dimension **{dimension}** — puntajes por unidad")
    _display(df)
    return df


def mostrar_razonamientos(largo, dimension=None, unidad=None, pregunta=None, max_filas=40):
    """Razonamiento de cada respuesta, filtrable."""
    sub = largo
    for col, val in (("dimension", dimension), ("fuente", unidad), ("pregunta", pregunta)):
        if val is not None:
            sub = sub[sub[col] == val]
    _md(f"### Razonamientos ({len(sub)} respuestas)")
    for r in sub.head(max_filas).itertuples():
        _md(f"- **{r.dimension}{r.pregunta[1:]}** · unidad `{r.fuente}` (etapa {r.momento}, "
            f"preguntada en posicion {r.posicion}) — puntaje `{r.score_txt}`  \n  "
            f"{r.reasoning or 'Sin justificacion'}")
    if len(sub) > max_filas:
        _md(f"... {len(sub) - max_filas} mas.")


def mostrar_na(largo):
    """Solo las respuestas N/A con su justificacion."""
    mostrar_razonamientos(largo[largo["score"].isna()])


def residuos_posicion(largo, control="auto"):
    """
    Anade la columna `residuo`: el puntaje MENOS lo que se esperaria para esa
    misma pregunta, para poder atribuir lo que sobra a la posicion en el prompt.

    control:
      "celda"    - resta la media de la MISMA pregunta en la MISMA unidad
                   (comparacion pareada perfecta; necesita >1 observacion por
                   celda, es decir varias iteraciones, cada una con otro barajado).
      "aditivo"  - resta el efecto de la pregunta y el de la unidad:
                   residuo = score - media(pregunta) - media(unidad) + media(global).
                   Sirve con UNA sola corrida.
      "auto"     - "celda" si hay repeticiones; si no, "aditivo".

    Sin control, comparar posiciones mezcla el efecto de la posicion con el de
    QUE preguntas y QUE unidades cayeron ahi.
    """
    df = largo.copy()
    celda = ["dimension", "pregunta", "fuente", "momento"]
    repeticiones = df.groupby(celda)["score"].transform("size")

    if control == "auto":
        control = "celda" if repeticiones.median() > 1 else "aditivo"

    if control == "celda":
        df = df[repeticiones > 1]
        if df.empty:
            raise ValueError("No hay repeticiones por celda: usa control='aditivo' o itera "
                             "(cada iteracion baraja distinto).")
        df["residuo"] = df["score"] - df.groupby(celda)["score"].transform("mean")
    elif control == "aditivo":
        mu = df["score"].mean()
        df["residuo"] = (df["score"]
                         - df.groupby(["dimension", "pregunta"])["score"].transform("mean")
                         - df.groupby(["fuente", "momento"])["score"].transform("mean") + mu)
    elif control == "ninguno":
        df["residuo"] = df["score"] - df["score"].mean()
    else:
        raise ValueError("control debe ser 'auto', 'celda', 'aditivo' o 'ninguno'")

    df.attrs["control"] = control
    return df


def _pendiente(x, y):
    """Regresion simple y = a + b*x; devuelve (b, error_estandar_de_b, n)."""
    m = x.notna() & y.notna()
    x, y = x[m].astype(float), y[m].astype(float)
    n = len(x)
    sxx = ((x - x.mean()) ** 2).sum()
    if n < 3 or sxx == 0:
        return np.nan, np.nan, n
    b = ((x - x.mean()) * (y - y.mean())).sum() / sxx
    resid = y - (y.mean() + b * (x - x.mean()))
    se = np.sqrt((resid ** 2).sum() / (n - 2) / sxx)
    return b, se, n


def efecto_posicion(largo, control="auto", bins=6, mostrar=True):
    """
    ¿Responde peor el modelo al final del prompt?

    Compara cada respuesta contra lo esperado para ESA MISMA pregunta
    (`residuos_posicion`) y agrupa por el lugar que ocupo en su llamada. Un
    residuo medio cercano a 0 en todos los tramos = no hay efecto de posicion.

    Devuelve la tabla por tramo; ademas ajusta una recta residuo ~ posicion e
    informa la pendiente por cada 10 posiciones con su error.
    """
    df = residuos_posicion(largo, control)
    ctrl = df.attrs["control"]
    df["tramo"] = pd.qcut(df["posicion"], q=min(bins, df["posicion"].nunique()), duplicates="drop")

    t = df.groupby("tramo", observed=True).agg(
        n=("score_txt", "size"),
        residuo_medio=("residuo", "mean"),
        error=("residuo", lambda s: s.std(ddof=1) / np.sqrt(s.count()) if s.count() > 1 else np.nan),
        puntaje_medio=("score", "mean"),
        tasa_na=("score", lambda s: s.isna().mean()),
        palabras_reasoning=("reasoning", lambda s: s.str.split().str.len().mean()),
    ).round(3)

    b, se, n = _pendiente(df["posicion"], df["residuo"])
    b_na, se_na, _ = _pendiente(df["posicion"], df["score"].isna().astype(float))
    b_w, se_w, _ = _pendiente(df["posicion"], df["reasoning"].str.split().str.len())

    # Magnitud: con n grande, |t| > 2 no implica que el efecto importe.
    sd = df["residuo"].std(ddof=1)
    tramo_pos = df["posicion"].max() - df["posicion"].min()
    efecto_total = b * tramo_pos
    en_sd = efecto_total / sd if sd else np.nan
    var_explicada = (b * df["posicion"].std(ddof=1) / sd) ** 2 if sd else np.nan

    if mostrar:
        etiqueta = {"celda": "misma pregunta en la misma unidad (pareado)",
                    "aditivo": "efecto de pregunta y de unidad descontados",
                    "ninguno": "sin control"}[ctrl]
        _md(f"### Efecto de la posicion en el prompt\n\nControl: **{etiqueta}** · {n:,} respuestas puntuadas")
        _display(t)
        if not (np.isnan(b) or np.isnan(se) or se == 0):
            _md(f"- Puntaje: **{10*b:+.3f} ± {10*se:.3f}** por cada 10 posiciones "
                f"(t = {b/se:+.1f}). De la primera a la ultima pregunta: **{efecto_total:+.2f} puntos** "
                f"= {en_sd:.2f} desviaciones del residuo (sd = {sd:.2f}); "
                f"la posicion explica el **{var_explicada:.1%}** de la variabilidad.")
            if not (np.isnan(b_na) or np.isnan(se_na) or se_na == 0):
                _md(f"- Tasa de N/A: **{10*b_na:+.1%} ± {10*se_na:.1%}** por cada 10 posiciones "
                    f"(t = {b_na/se_na:+.1f}).")
            if not (np.isnan(b_w) or np.isnan(se_w) or se_w == 0):
                _md(f"- Largo del razonamiento: **{10*b_w:+.1f} ± {10*se_w:.1f}** palabras por cada "
                    f"10 posiciones (t = {b_w/se_w:+.1f}) — el esfuerzo puede caer aunque el puntaje no.")
            sospechosos = [nom for nom, (bb, ss) in (("el puntaje", (b, se)),
                                                     ("la tasa de N/A", (b_na, se_na)),
                                                     ("el largo del razonamiento", (b_w, se_w)))
                           if ss and not np.isnan(ss) and not np.isnan(bb) and abs(bb / ss) >= 2]
            puntaje_afectado = "el puntaje" in sospechosos

            if not sospechosos:
                _md(f"**Sin efecto de posicion** (|t| < 2 en todo): preguntar las "
                    f"{int(df['posicion'].max())} juntas es seguro.")
            elif puntaje_afectado and abs(en_sd) >= 0.5:
                _md(f"**Efecto grande en el puntaje**: {efecto_total:+.2f} puntos de punta a punta "
                    f"({en_sd:.2f} sd, {var_explicada:.1%} de la variabilidad). Conviene partir "
                    "`PREGUNTAS` en lotes y llamar varias veces por unidad; el resto no cambia.")
            elif puntaje_afectado:
                _md(f"**Efecto en el puntaje detectable pero pequeno**: {efecto_total:+.2f} puntos "
                    f"de punta a punta ({en_sd:.2f} sd, {var_explicada:.1%} de la variabilidad). "
                    "Como el orden se baraja en cada llamada, cada pregunta cae temprano y tarde por "
                    "igual: **no sesga** las tablas consolidadas, solo les agrega algo de ruido. "
                    "Con n grande, |t| >= 2 dice que el efecto existe, no que importe.")
            else:
                _md("**El puntaje no se ve afectado** por la posicion, pero si "
                    f"{' y '.join(sospechosos)}. Las tablas consolidadas no cambian; revisa los "
                    "razonamientos de las ultimas posiciones si los usas como material cualitativo.")
    return t
