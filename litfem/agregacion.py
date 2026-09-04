"""Consolidacion de resultados con estimacion de error (todos los modos).

Todos los modos producen ya el MISMO formato largo (`puntuar.puntuar_unidades`),
una fila por observacion, y desde ahi se agrega SIEMPRE en el mismo orden:
  1) por (dimension, momento, pregunta) con `agregador(obs, error)` -> valor, error, n
  2) por (dimension, momento) con `combinador(valores, errores, error)` -> valor, error

Por defecto (`agregar_celda`, `combinar_preguntas`): media de los numericos con
error "sem" (std/sqrt(n)) o "std"; a nivel dimension, media de las medias de las
preguntas con error propagado (sem: sqrt(sum sem_q^2)/k; std: sqrt(mean std_q^2)).
Ambas funciones se pueden reemplazar desde el notebook.
"""

import numpy as np
import pandas as pd

from .comun import MOMENTOS, _md, _display

__all__ = [
    "MOMENTOS", "unir_largos", "agregar_celda", "combinar_preguntas", "consolidar",
    "formatear_con_error", "mostrar_consolidado", "comparar_consolidados", "comparar_modos",
]


def unir_largos(*largos):
    """Concatena largos de varias corridas renumerando la iteracion 1..k."""
    return pd.concat([df.assign(iteracion=k) for k, df in enumerate(largos, start=1)],
                     ignore_index=True)


# ---------------------------------------------------------------------------
#  Pasos 1 y 2: agregar
# ---------------------------------------------------------------------------

def agregar_celda(obs, error="sem"):
    """
    AGREGADOR POR DEFECTO de una celda (dimension x momento x pregunta).
    obs: sub-DataFrame largo de la celda (score float con NaN = N/A, score_txt,
    reasoning, fuente, iteracion). Devuelve {"valor", "error", "n"}.
    """
    v = obs["score"].dropna()
    n = int(v.count())
    sd = v.std(ddof=1) if n > 1 else np.nan
    return {"valor": v.mean() if n else np.nan,
            "error": sd / np.sqrt(n) if error == "sem" else sd,
            "n": n}


def combinar_preguntas(valores, errores, error="sem"):
    """
    COMBINADOR POR DEFECTO del nivel dimension (para un momento): media simple de
    las preguntas con valor; error propagado
      sem -> sqrt(sum err_q^2) / k       std -> sqrt(mean err_q^2)
    """
    k = valores.count()
    if k == 0:
        return np.nan, np.nan
    err = (np.sqrt((errores ** 2).sum(min_count=1)) / k if error == "sem"
           else np.sqrt((errores ** 2).mean()))
    return valores.mean(), err


def consolidar(df_largo, error="sem", momentos=MOMENTOS, agregador=None, combinador=None):
    """
    Devuelve un dict:
      preguntas / preguntas_err / preguntas_n : {dim: DataFrame pregunta x momento}
      dimensiones / dimensiones_err           : DataFrame dimension x momento
      n_total / n_na                          : DataFrame dimension x momento
      error, largo
    """
    if error not in ("sem", "std"):
        raise ValueError("error debe ser 'sem' o 'std'")
    agregador = agregador or agregar_celda
    combinador = combinador or combinar_preguntas

    dims = list(dict.fromkeys(df_largo["dimension"]))
    out = {"preguntas": {}, "preguntas_err": {}, "preguntas_n": {}}
    dim_valor, dim_err, n_total, n_na = {}, {}, {}, {}

    for dim in dims:
        sub = df_largo[df_largo["dimension"] == dim]
        qs = list(dict.fromkeys(sub["pregunta"]))

        celdas = pd.DataFrame({k: agregador(obs, error) for k, obs in sub.groupby(["pregunta", "momento"])}).T
        celdas.index.names = ["pregunta", "momento"]

        def tabla(campo):
            return celdas[campo].unstack("momento").reindex(index=qs, columns=momentos)

        valor, err = tabla("valor").astype(float), tabla("error").astype(float)
        n = tabla("n").fillna(0).astype(int)
        out["preguntas"][dim], out["preguntas_err"][dim], out["preguntas_n"][dim] = valor, err, n

        combinado = {m: combinador(valor[m], err[m], error) for m in momentos}
        dim_valor[dim] = {m: v for m, (v, _) in combinado.items()}
        dim_err[dim] = {m: e for m, (_, e) in combinado.items()}
        n_total[dim] = n.sum().to_dict()
        n_na[dim] = sub["score"].isna().groupby(sub["momento"]).sum().reindex(momentos).fillna(0).astype(int).to_dict()

    def tabla_dim(d):
        t = pd.DataFrame.from_dict(d, orient="index").reindex(index=dims, columns=momentos)
        t.index.name, t.columns.name = "Dimension", "momento"
        return t

    out.update(dimensiones=tabla_dim(dim_valor), dimensiones_err=tabla_dim(dim_err),
               n_total=tabla_dim(n_total), n_na=tabla_dim(n_na), error=error, largo=df_largo)
    return out


# ---------------------------------------------------------------------------
#  Presentacion
# ---------------------------------------------------------------------------

def formatear_con_error(media, err, dec=2):
    """DataFrame de strings 'media ± error' ('media' si el error es NaN, '—' sin media)."""
    def celda(m, e):
        if pd.isna(m):
            return "—"
        return f"{m:.{dec}f}" if pd.isna(e) else f"{m:.{dec}f} ± {e:.{dec}f}"
    return pd.DataFrame({c: [celda(m, e) for m, e in zip(media[c], err[c])] for c in media.columns},
                        index=media.index)


_ETIQUETA_ERROR = {"sem": "error estandar de la media", "std": "desviacion estandar"}


def mostrar_consolidado(cons, titulo="", dec=2, por_pregunta=True):
    """Tablas consolidadas (media ± error) de un dict de `consolidar`."""
    et = _ETIQUETA_ERROR[cons["error"]]
    if titulo:
        _md(f"## {titulo}")
    if por_pregunta:
        _md(f"### Por pregunta y momento (media ± {et})")
        for dim, media in cons["preguntas"].items():
            _md(f"**Dimension {dim}**")
            _display(formatear_con_error(media, cons["preguntas_err"][dim], dec))
    _md(f"### Por dimension y momento (media de las preguntas ± {et} propagado)")
    _display(formatear_con_error(cons["dimensiones"], cons["dimensiones_err"], dec))
    _md("### Observaciones numericas usadas (n) y N/A descartados")
    _display(cons["n_total"].astype(str) + " (N/A: " + cons["n_na"].astype(str) + ")")
    return cons["dimensiones"]


def comparar_consolidados(cons_a, cons_b, nombre_a="A", nombre_b="B", dec=2):
    """Dos consolidados lado a lado (dimension x momento) y su diferencia a - b."""
    a, b = cons_a["dimensiones"], cons_b["dimensiones"]
    comb = pd.concat({nombre_a: a.round(dec), nombre_b: b.round(dec),
                      f"{nombre_a} - {nombre_b}": (a - b).round(dec)}, axis=1)
    _md(f"## Comparacion {nombre_a} vs {nombre_b} (dimension x momento)")
    _display(comb)
    return comb


def comparar_modos(consolidados, dec=2, con_error=True):
    """Varios consolidados lado a lado: {nombre: cons}. Celdas 'media ± error' o solo media."""
    bloques = {nombre: (formatear_con_error(c["dimensiones"], c["dimensiones_err"], dec)
                        if con_error else c["dimensiones"].round(dec))
               for nombre, c in consolidados.items()}
    comb = pd.concat(bloques, axis=1)
    _md(f"## Comparacion de modos: {' · '.join(consolidados)} (dimension x momento)")
    _display(comb)
    return comb
