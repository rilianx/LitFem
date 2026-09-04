"""Calibrar el surrogado con un corpus de obras, y validar que la calibracion sirva.

Con N obras analizadas de las dos maneras (resumen y libro) se puede corregir el
surrogado: si el resumen subestima Vr en 0.4 puntos en todas las obras, ese
desfase se resta y listo. Lo que NO se puede es ajustar la correccion con las
mismas obras con las que se la evalua: eso mide cuanto memorizo el corpus, no
cuanto va a acertar en la obra 21.

Por eso todo aqui se evalua con **validacion cruzada dejando una obra fuera**:
se ajusta con N-1 obras y se predice la que quedo afuera, N veces. La diferencia
entre el error de ajuste y el de validacion es exactamente el sobreajuste.

Regla practica: si un modelo mas complejo (una recta por dimension) no baja el
error DE VALIDACION respecto de uno simple (un desfase por dimension), el simple
es mejor aunque ajuste peor: generaliza.
"""

import numpy as np
import pandas as pd

from .comun import _md, _display

__all__ = ["pares_de_consolidados", "calibrar", "MODELOS"]


def pares_de_consolidados(por_obra, nombre_sur="surrogado", nombre_ref="referencia"):
    """
    {obra: (cons_surrogado, cons_referencia)} -> DataFrame de pares, una fila por
    (obra, dimension, momento) con las dos estimaciones.
    """
    filas = []
    for obra, (cons_sur, cons_ref) in por_obra.items():
        sur, ref = cons_sur["dimensiones"], cons_ref["dimensiones"]
        for dim in sur.index:
            if dim not in ref.index:
                continue
            for momento in sur.columns:
                s, r = sur.loc[dim, momento], ref.loc[dim, momento]
                if pd.notna(s) and pd.notna(r):
                    filas.append({"obra": obra, "dimension": dim, "momento": momento,
                                  nombre_sur: float(s), nombre_ref: float(r)})
    return pd.DataFrame(filas)


# --- modelos de calibracion: cada uno es (ajustar, aplicar) -------------------

def _sin_calibrar():
    return (lambda tr, s, r: None), (lambda p, df, s: df[s].values)


def _desfase(por_dimension):
    def ajustar(tr, s, r):
        if por_dimension:
            return (tr[r] - tr[s]).groupby(tr["dimension"]).mean().to_dict()
        return (tr[r] - tr[s]).mean()

    def aplicar(p, df, s):
        if por_dimension:
            base = np.mean(list(p.values())) if p else 0.0
            return df[s].values + df["dimension"].map(lambda d: p.get(d, base)).values
        return df[s].values + p
    return ajustar, aplicar


def _lineal(por_dimension):
    def _fit(x, y):
        if len(x) < 3 or np.ptp(x) == 0:
            return (1.0, float(np.mean(y - x)) if len(x) else 0.0)
        b, a = np.polyfit(x, y, 1)
        return (b, a)

    def ajustar(tr, s, r):
        if por_dimension:
            return {d: _fit(g[s].values, g[r].values) for d, g in tr.groupby("dimension")}
        return _fit(tr[s].values, tr[r].values)

    def aplicar(p, df, s):
        if por_dimension:
            glob = (np.mean([b for b, _ in p.values()]), np.mean([a for _, a in p.values()]))
            coef = df["dimension"].map(lambda d: p.get(d, glob))
            b = np.array([c[0] for c in coef]); a = np.array([c[1] for c in coef])
            return a + b * df[s].values
        b, a = p
        return a + b * df[s].values
    return ajustar, aplicar


MODELOS = {
    "sin calibrar": _sin_calibrar(),
    "desfase global": _desfase(False),
    "desfase por dimension": _desfase(True),
    "lineal global": _lineal(False),
    "lineal por dimension": _lineal(True),
}


def calibrar(pares, surrogado="surrogado", referencia="referencia",
             modelos=None, mostrar=True):
    """
    Ajusta y valida modelos de calibracion con leave-one-obra-out.

    pares: DataFrame de `pares_de_consolidados` (obra, dimension, momento, sur, ref)

    Devuelve {"tabla", "predicciones", "parametros"}:
      tabla         : por modelo, MAE y sesgo de AJUSTE y de VALIDACION
      predicciones  : los pares con la prediccion validada de cada modelo
      parametros    : los coeficientes ajustados con TODAS las obras (para usar
                      de aqui en adelante en obras nuevas)
    """
    modelos = modelos or MODELOS
    obras = sorted(pares["obra"].unique())
    if len(obras) < 3:
        _md(f"> Solo hay {len(obras)} obra(s): la validacion cruzada necesita al menos 3, "
            "e idealmente 15-20 para que la calibracion sea creible.")

    filas, pred_cols, params_full = [], {}, {}
    for nombre, (ajustar, aplicar) in modelos.items():
        p_full = ajustar(pares, surrogado, referencia)
        params_full[nombre] = p_full
        pred_ajuste = aplicar(p_full, pares, surrogado)

        pred_cv = np.full(len(pares), np.nan)
        if len(obras) >= 3:
            for obra in obras:
                fuera = (pares["obra"] == obra).values
                p = ajustar(pares[~fuera], surrogado, referencia)
                pred_cv[fuera] = aplicar(p, pares[fuera], surrogado)

        real = pares[referencia].values
        fila = {"modelo": nombre,
                "MAE ajuste": np.abs(pred_ajuste - real).mean(),
                "sesgo ajuste": (pred_ajuste - real).mean()}
        if not np.isnan(pred_cv).all():
            fila["MAE validacion"] = np.abs(pred_cv - real).mean()
            fila["sesgo validacion"] = (pred_cv - real).mean()
            fila["sobreajuste"] = fila["MAE validacion"] - fila["MAE ajuste"]
            pred_cols[nombre] = pred_cv
        filas.append(fila)

    tabla = pd.DataFrame(filas).set_index("modelo").round(3)
    predicciones = pares.assign(**pred_cols)

    if mostrar:
        _md(f"## Calibracion del surrogado ({len(obras)} obras, {len(pares)} celdas)")
        _display(tabla)
        col = "MAE validacion" if "MAE validacion" in tabla else "MAE ajuste"
        mejor = tabla[col].idxmin()
        base = tabla.loc["sin calibrar", col]
        _md(f"**Mejor por error de validacion: {mejor}** ({tabla.loc[mejor, col]:.3f} vs "
            f"{base:.3f} sin calibrar, una mejora de {100*(1-tabla.loc[mejor, col]/base):.0f} %). "
            "Ese numero es lo que cabe esperar en una obra NUEVA; el de ajuste siempre es "
            "optimista. Si dos modelos empatan en validacion, quedate con el simple.")
        if "sobreajuste" in tabla:
            peor = tabla["sobreajuste"].idxmax()
            _md(f"El que mas sobreajusta es **{peor}** ({tabla.loc[peor, 'sobreajuste']:+.3f} "
                "entre ajuste y validacion): ahi es donde el corpus se esta memorizando.")
        _md("### Error residual por dimension (modelo elegido)")
        if mejor in predicciones:
            res = (predicciones[mejor] - predicciones[referencia])
            por_dim = pd.DataFrame({
                "MAE": res.abs().groupby(predicciones["dimension"]).mean(),
                "sesgo": res.groupby(predicciones["dimension"]).mean(),
            }).round(3).sort_values("MAE", ascending=False)
            _display(por_dim)
            _md("Las dimensiones de arriba son donde el resumen sigue sin servir aunque se "
                "corrija: alli conviene marcar el resultado como poco fiable, o no reportarlo.")

    return {"tabla": tabla, "predicciones": predicciones, "parametros": params_full}
