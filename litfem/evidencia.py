"""Modo C: extraer evidencia por fragmento y juzgar despues, por etapa.

Paso 1 (`extraer_evidencia`): una llamada por fragmento con las 30 preguntas
barajadas y con etiquetas neutras (el extractor tampoco sabe a que dimension
pertenece cada una). Devuelve citas verbatim, verificadas contra el fragmento.

Paso 2: las citas de cada etapa se convierten en UNA unidad
(`unidades_evidencia`) y se puntuan con el mismo `puntuar_unidades` de siempre,
con el prompt del juez. 3 llamadas.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, create_model
from functools import lru_cache

from .comun import MOMENTOS, _md, _display
from .llm import llm_function, MODELO_POR_DEFECTO
from .preguntas import barajar, formatear_preguntas, parsear_preguntas

__all__ = ["Evidence", "esquema_evidencia", "extraer_evidencia", "resumen_evidencia",
           "mostrar_evidencia", "formatear_evidencia", "unidades_evidencia", "agregar_jackknife"]

COLUMNAS_EVIDENCIA = ["fragmento", "etapa", "dimension", "pregunta", "id_pregunta",
                      "polaridad", "cita", "nota", "verificada"]


class Evidence(BaseModel):
    question: str = Field(description="Question label exactly as listed, e.g. 'S7'")
    quote: str = Field(description="VERBATIM quote from the excerpt (exact substring, 1-3 sentences)")
    polarity: Literal["supports", "contradicts", "context"]
    note: str = Field(description="One sentence: why this quote is relevant to that question")


@lru_cache(maxsize=None)
def esquema_evidencia():
    return create_model("FragmentEvidence",
                        evidence=(List[Evidence],
                                  Field(description="All relevant evidence found; may be empty")))


def _norm(s):
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def extraer_evidencia(unidades, character, instructions, preguntas, max_citas=3,
                      model=MODELO_POR_DEFECTO, temperature=0.0, max_workers=4,
                      semilla=0, verbose=True):
    """
    Una llamada por unidad (fragmento) para TODAS las preguntas, barajadas y sin
    revelar la dimension. `instructions` usa {character} {unit_id} {stage}
    {excerpt} {questions} {max_citas}.

    Devuelve un DataFrame con una fila por cita (COLUMNAS_EVIDENCIA); las
    etiquetas neutras se traducen de vuelta a (dimension, pregunta) aqui mismo,
    asi que el juez recibe siempre las preguntas canonicas.
    """
    if isinstance(preguntas, dict):
        preguntas = parsear_preguntas(preguntas)

    def _una(u):
        orden = barajar(preguntas, (semilla, "ev", u["id"]))
        mapa = dict(orden)
        prompt = instructions.format(character=character, unit_id=u["id"], stage=u["etapa"],
                                     excerpt=u["texto"], questions=formatear_preguntas(orden),
                                     max_citas=max_citas)
        try:
            evs = llm_function(prompt, esquema_evidencia(), model=model,
                               temperature=temperature).get("evidence", [])
        except Exception as e:
            return u, [], f"{type(e).__name__}: {e}"
        return u, [(mapa.get(ev.get("question", "").strip()), ev) for ev in evs], None

    if verbose:
        print(f"Extrayendo evidencia de {len(unidades)} fragmentos x {len(preguntas)} preguntas...")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        salidas = list(pool.map(_una, unidades))

    filas, errores = [], {}
    for u, pares, err in salidas:
        if err:
            errores[u["id"]] = err
            continue
        texto_norm = _norm(u["texto"])
        for q, ev in pares:
            if q is None:          # etiqueta inventada por el modelo
                continue
            cita = ev.get("quote", "")
            filas.append({"fragmento": u["id"], "etapa": u["etapa"], "dimension": q["dimension"],
                          "pregunta": q["pregunta"], "id_pregunta": q["id"],
                          "polaridad": ev.get("polarity", ""), "cita": cita, "nota": ev.get("note", ""),
                          "verificada": bool(cita) and _norm(cita) in texto_norm})

    df = pd.DataFrame(filas, columns=COLUMNAS_EVIDENCIA)
    df.attrs["errores"] = errores
    if verbose:
        print(f"{len(df)} citas extraidas; {int((~df['verificada']).sum())} no verificadas literalmente"
              + (f"; {len(errores)} fragmentos con error" if errores else ""))
    return df


def resumen_evidencia(df):
    """Tabla dimension x (etapa, polaridad) con el numero de citas."""
    if df.empty:
        return pd.DataFrame()
    return pd.pivot_table(df, index="dimension", columns=["etapa", "polaridad"],
                          values="cita", aggfunc="count", fill_value=0)


def mostrar_evidencia(df, dimension=None, etapa=None, solo_verificadas=False, max_filas=60):
    """Listado legible de citas, filtrable por dimension / etapa."""
    for col, val in (("dimension", dimension), ("etapa", etapa)):
        if val:
            df = df[df[col] == val]
    if solo_verificadas:
        df = df[df["verificada"]]
    titulo = " · ".join(filter(None, (dimension and f"Dimension {dimension}",
                                      etapa and f"Etapa {etapa}"))) or "Evidencia"
    _md(f"### {titulo} ({len(df)} citas)")
    for r in df.sort_values(["dimension", "pregunta", "fragmento"]).head(max_filas).itertuples():
        marca = "" if r.verificada else " ⚠︎ no verificada"
        _md(f"- **{r.id_pregunta}** · Frag. {r.fragmento} ({r.etapa}) · *{r.polaridad}*{marca}  \n"
            f"  > {r.cita}  \n  {r.nota}")
    if len(df) > max_filas:
        _md(f"... {len(df) - max_filas} citas mas (usa filtros o `max_filas`).")


def formatear_evidencia(sub):
    """Bloque de texto con las citas agrupadas por pregunta (id canonico)."""
    if sub.empty:
        return "(No evidence was extracted for this stage.)"
    bloques = []
    for _, g in sub.sort_values(["id_pregunta", "fragmento"]).groupby("id_pregunta", sort=False):
        lineas = [f"[{g['id_pregunta'].iloc[0]}]"] + [
            f'  - Fragment {r.fragmento} ({r.polaridad}): "{r.cita}" -- {r.nota}' for r in g.itertuples()]
        bloques.append("\n".join(lineas))
    return "\n\n".join(bloques)


def unidades_evidencia(df, etapas=MOMENTOS, solo_verificadas=False, excluir_fragmentos=()):
    """
    Una unidad por etapa, cuyo `texto` es la evidencia de esa etapa ya formateada.
    Lista para `puntuar_unidades(..., instructions=PROMPT_JUEZ)`.

    excluir_fragmentos: para el jackknife (dejar un fragmento fuera).
    """
    base = df[df["verificada"]] if solo_verificadas else df
    base = base[~base["fragmento"].isin(excluir_fragmentos)]
    unidades = []
    for et in etapas:
        sub = base[base["etapa"] == et]
        frags = sorted(set(df.loc[df["etapa"] == et, "fragmento"]) - set(excluir_fragmentos))
        unidades.append({"id": et, "etapa": et, "texto": formatear_evidencia(sub),
                         "meta": {"fragment_ids": ", ".join(map(str, frags)) or "-"}})
    return unidades


def agregar_jackknife(obs, error="sem"):
    """
    Agregador para observaciones leave-one-out: valor = media de las estimaciones,
    error = sqrt((n-1)/n * sum((theta_i - theta_bar)^2)) (error estandar jackknife);
    con error="std", la desviacion estandar simple.
    """
    v = obs["score"].dropna()
    n = int(v.count())
    if n == 0:
        return {"valor": np.nan, "error": np.nan, "n": 0}
    media = v.mean()
    if n == 1:
        return {"valor": media, "error": np.nan, "n": 1}
    err = np.sqrt((n - 1) / n * ((v - media) ** 2).sum()) if error == "sem" else v.std(ddof=1)
    return {"valor": media, "error": err, "n": n}
