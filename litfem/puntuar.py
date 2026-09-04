"""El nucleo unico: UNA llamada por unidad, con TODAS las preguntas barajadas.

`puntuar_unidades` devuelve directamente el FORMATO LARGO que consume
`agregacion.consolidar`, asi que no hay estructuras intermedias por modo:

    iteracion | dimension | momento | pregunta | score | score_txt |
    reasoning | fuente (id de unidad) | etiqueta (S7) | posicion (7)

`posicion` es el lugar que ocupo la pregunta en ese prompt: permite medir a
posteriori si el modelo responde peor al final (ver `display.efecto_posicion`).
"""

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .comun import a_numero, es_error
from .llm import llm_function, MODELO_POR_DEFECTO
from .preguntas import barajar, formatear_preguntas, esquema_puntajes, parsear_preguntas

__all__ = ["COLUMNAS_LARGO", "puntuar_unidades", "iterar"]

COLUMNAS_LARGO = ["iteracion", "dimension", "momento", "pregunta", "score", "score_txt",
                  "reasoning", "fuente", "etiqueta", "posicion"]


def _puntuar_una(unidad, character, instructions, preguntas, model, temperature,
                 semilla, iteracion):
    orden = barajar(preguntas, (semilla, unidad["id"], iteracion))
    contexto = {"character": character, "stage": unidad["etapa"], "unit_id": unidad["id"],
                "excerpt": unidad["texto"], "questions": formatear_preguntas(orden),
                **unidad.get("meta", {})}
    try:
        res = llm_function(instructions.format(**contexto), esquema_puntajes(len(orden)),
                           model=model, temperature=temperature)
    except Exception as e:
        res = {"error": f"{type(e).__name__}: {e}"}

    filas = []
    for pos, (etiqueta, q) in enumerate(orden, start=1):
        if es_error(res):
            score, razon = "Error", str(res.get("error", res))
        else:
            resp = res.get(etiqueta) or {}
            score, razon = resp.get("score", "N/A"), resp.get("reasoning", "")
        filas.append({"iteracion": iteracion, "dimension": q["dimension"], "momento": unidad["etapa"],
                      "pregunta": q["pregunta"], "score": a_numero(score), "score_txt": score,
                      "reasoning": razon, "fuente": unidad["id"], "etiqueta": etiqueta, "posicion": pos})
    return filas


def puntuar_unidades(unidades, character, instructions, preguntas, model=MODELO_POR_DEFECTO,
                     temperature=0.0, max_workers=6, semilla=0, iteracion=1, verbose=True):
    """
    Una llamada por unidad con las `preguntas` barajadas (etiquetas neutras S1..Sn).

    unidades    : lista de `unidades.unidades_*`
    instructions: template con {character} {stage} {unit_id} {excerpt} {questions}
                  (mas las claves de `meta` de la unidad; las no usadas se ignoran)
    preguntas   : lista de `preguntas.parsear_preguntas`, o el dict de dimensiones
    semilla     : cambia el barajado; el orden real es reproducible por
                  (semilla, id de unidad, iteracion)

    Devuelve el DataFrame en formato largo.
    """
    if isinstance(preguntas, dict):
        preguntas = parsear_preguntas(preguntas)
    if verbose:
        print(f"Puntuando {len(unidades)} unidades x {len(preguntas)} preguntas "
              f"({len(unidades)} llamadas, {max_workers} hilos)...")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        lotes = pool.map(lambda u: _puntuar_una(u, character, instructions, preguntas,
                                                model, temperature, semilla, iteracion), unidades)
        filas = [f for lote in lotes for f in lote]

    largo = pd.DataFrame(filas, columns=COLUMNAS_LARGO)
    # el barajado altera el orden de las filas: se reordena al orden canonico
    # (dimensiones como en `dimensiones`, preguntas Q1..Q5) para que las tablas
    # salgan siempre igual. `posicion` conserva el orden real del prompt.
    orden_dim = list(dict.fromkeys(q["dimension"] for q in preguntas))
    largo["dimension"] = pd.Categorical(largo["dimension"], categories=orden_dim, ordered=True)
    largo = largo.sort_values(["iteracion", "fuente", "dimension", "pregunta"],
                              kind="stable").reset_index(drop=True)
    largo["dimension"] = largo["dimension"].astype(str)

    fallidas = largo.loc[largo["score_txt"] == "Error", "fuente"].nunique()
    if verbose and fallidas:
        print(f"AVISO: {fallidas} unidades fallaron (ver score_txt == 'Error').")
    return largo


def iterar(fn, num_iteraciones=3, verbose=True):
    """
    Repite `fn(iteracion, semilla)` -> DataFrame largo, y concatena las corridas.
    Cada iteracion usa su propia semilla, asi que ademas cambia el barajado.

        iterar(lambda it, s: puntuar_unidades(us, PERSONAJE, PROMPT, preg,
                                              iteracion=it, semilla=s), 3)
    """
    partes = []
    for k in range(1, num_iteraciones + 1):
        if verbose:
            print(f"=== Iteracion {k} de {num_iteraciones} ===")
        partes.append(fn(k, k))
    return pd.concat(partes, ignore_index=True)
