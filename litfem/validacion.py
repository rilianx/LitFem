"""Validacion del instrumento: fiabilidad, saturacion y eleccion de agregador.

Todo esto se calcula SOBRE DATOS YA OBTENIDOS (no gasta llamadas), pero necesita
que el largo traiga VARIAS iteraciones: sin repeticiones no hay con que estimar
cuanto se repite el modo a si mismo.

El orden correcto para elegir un agregador es:
  1) `repetibilidad`  - cuanto se repite un modo a si mismo. Es el TECHO: dos
     modos no pueden coincidir mejor de lo que cada uno coincide consigo mismo.
  2) `comparar_agregadores` con criterio = fiabilidad.
  3) Recien entonces `acuerdo` contra el modo de referencia.
Elegir el agregador por el acuerdo es ajustar el analisis al resultado que se
quiere: el numero sale bonito y no significa nada.
"""

import numpy as np
import pandas as pd

from .comun import _md, _display
from .agregacion import consolidar, agregar_celda

__all__ = ["repetibilidad", "fiabilidad_preguntas", "curva_unidades", "comparar_agregadores"]


def _tablas_por_iteracion(largo, **kw):
    return {it: consolidar(sub, **kw)["dimensiones"]
            for it, sub in largo.groupby("iteracion")}


def repetibilidad(largo, mostrar=True, **kw):
    """
    Test-retest: consolida cada iteracion por separado y compara las tablas entre si.

    Devuelve {"mae_medio", "por_par", "sd_por_celda", "tabla_media"}:
      mae_medio    : diferencia absoluta media entre dos corridas del MISMO modo.
                     Es el techo de acuerdo alcanzable contra cualquier otro modo.
      sd_por_celda : desviacion entre iteraciones de cada celda (donde es inestable)
    """
    tablas = _tablas_por_iteracion(largo, **kw)
    its = sorted(tablas)
    if len(its) < 2:
        raise ValueError("Se necesitan >= 2 iteraciones. Corre el modo varias veces con `iterar`.")

    pares = []
    for i, a in enumerate(its):
        for b in its[i + 1:]:
            d = (tablas[a] - tablas[b]).values.ravel()
            d = d[~np.isnan(d)]
            pares.append({"iter A": a, "iter B": b, "MAE": np.abs(d).mean(),
                          "sesgo": d.mean(), "max |dif|": np.abs(d).max()})
    por_par = pd.DataFrame(pares).round(3)
    apilado = np.dstack([t.values for t in tablas.values()])
    sd = pd.DataFrame(apilado.std(axis=2, ddof=1), index=tablas[its[0]].index,
                      columns=tablas[its[0]].columns).round(3)
    media = pd.DataFrame(apilado.mean(axis=2), index=tablas[its[0]].index,
                         columns=tablas[its[0]].columns).round(2)
    mae = por_par["MAE"].mean()

    if mostrar:
        _md(f"### Repetibilidad ({len(its)} corridas del mismo modo)")
        _display(por_par)
        _md(f"**MAE medio entre dos corridas: {mae:.3f} puntos.** Es el techo: ningun otro modo "
            f"puede coincidir con este mejor que {mae:.2f}, porque ni el mismo se repite mejor.")
        _md("### Desviacion entre corridas por celda (donde el modo es inestable)")
        _display(sd)
    return {"mae_medio": mae, "por_par": por_par, "sd_por_celda": sd, "tabla_media": media}


def fiabilidad_preguntas(largo, mostrar=True, top=10):
    """
    Por pregunta: cuanto N/A produce, cuanto varia entre corridas de la misma
    unidad (ruido) y cuanto separa unas unidades de otras (senal).

    `discriminacion` = sd entre unidades / (sd entre unidades + sd dentro de celda).
    Cerca de 1: la pregunta distingue unidades. Cerca de 0: contesta casi lo mismo
    en todas partes, o el ruido se come la senal -> candidata a reescribir.
    """
    df = largo.copy()
    celda = ["dimension", "pregunta", "fuente", "momento"]
    dentro = df.groupby(celda)["score"].transform("mean")

    filas = []
    for (dim, q), g in df.groupby(["dimension", "pregunta"], sort=False):
        medias_unidad = g.groupby(["fuente", "momento"])["score"].mean()
        sd_entre = medias_unidad.std(ddof=1)
        sd_dentro = (g["score"] - dentro[g.index]).std(ddof=1)
        total = (sd_entre or 0) + (sd_dentro or 0)
        filas.append({
            "id": f"{dim}{q[1:]}", "dimension": dim, "pregunta": q,
            "n": int(g["score"].notna().sum()),
            "tasa N/A": g["score"].isna().mean(),
            "media": g["score"].mean(),
            "sd entre unidades": sd_entre,
            "sd dentro (ruido)": sd_dentro,
            "discriminacion": (sd_entre / total) if total else np.nan,
        })
    t = pd.DataFrame(filas).set_index("id").round(3)

    if mostrar:
        _md("### Fiabilidad por pregunta")
        _md("Las mas problematicas (mucho N/A o poca discriminacion):")
        _display(t.sort_values(["discriminacion", "tasa N/A"], ascending=[True, False]).head(top))
        muchas_na = t.index[t["tasa N/A"] > 0.5].tolist()
        planas = t.index[t["discriminacion"] < 0.4].tolist()
        if muchas_na:
            _md(f"- **> 50 % de N/A**: {', '.join(muchas_na)} — el texto casi nunca las contesta; "
                "revisa el enunciado o acepta que miden cobertura mas que grado.")
        if planas:
            _md(f"- **Discriminacion baja**: {', '.join(planas)} — responden casi igual en todas "
                "las unidades, asi que aportan poco a las diferencias entre momentos.")
    return t


def curva_unidades(largo, tamanos=None, repeticiones=20, semilla=0, mostrar=True, **kw):
    """
    ¿Cuantas unidades hacen falta? Submuestrea k unidades al azar, consolida y
    compara con la tabla completa. No gasta llamadas: reusa lo ya obtenido.

    Devuelve un DataFrame k -> MAE respecto de la tabla completa (media y sd).
    """
    completo = consolidar(largo, **kw)["dimensiones"]
    unidades = sorted(largo["fuente"].unique())
    tamanos = tamanos or [k for k in (3, 5, 8, 10, 15, 20, 30) if k < len(unidades)]
    rng = np.random.default_rng(semilla)

    filas = []
    for k in tamanos:
        maes = []
        for _ in range(repeticiones):
            muestra = rng.choice(unidades, size=k, replace=False)
            sub = largo[largo["fuente"].isin(muestra)]
            try:
                d = (consolidar(sub, **kw)["dimensiones"] - completo).values.ravel()
            except Exception:
                continue
            d = d[~np.isnan(d)]
            if len(d):
                maes.append(np.abs(d).mean())
        if maes:
            filas.append({"unidades": k, "MAE vs completo": np.mean(maes), "sd": np.std(maes)})
    t = pd.DataFrame(filas).set_index("unidades").round(3)

    if mostrar:
        _md(f"### Curva de saturacion ({len(unidades)} unidades disponibles)")
        _display(t)
        _md("Donde la curva se aplana, agregar mas fragmentos ya no cambia la tabla: "
            "ese es el tamano de muestra suficiente.")
    return t


def comparar_agregadores(agregadores, largo, largo_referencia=None, error="sem", mostrar=True):
    """
    Evalua varios agregadores sobre LOS MISMOS datos.

    Criterio primario: `repetibilidad` (MAE entre corridas del propio modo). El
    mejor agregador es el que menos ruido deja, no el que mas se parece a otro modo.
    Si se pasa `largo_referencia`, se anade el acuerdo contra el, pero como dato
    secundario: elegir por ahi es ajustar el analisis al resultado buscado.

    agregadores: {"nombre": funcion(obs, error) -> {"valor","error","n"}}
    """
    from .acuerdo import acuerdo as _acuerdo

    filas = []
    for nombre, fn in agregadores.items():
        fila = {"agregador": nombre}
        try:
            fila["repetibilidad (MAE)"] = repetibilidad(largo, mostrar=False,
                                                        error=error, agregador=fn)["mae_medio"]
        except ValueError:
            fila["repetibilidad (MAE)"] = np.nan
        cons = consolidar(largo, error=error, agregador=fn)
        fila["celdas sin valor"] = int(cons["dimensiones"].isna().sum().sum())
        # Cuidado: un agregador que devuelve casi lo mismo en todas las celdas es
        # perfectamente repetible y no mide nada. Lo que hay que maximizar es la
        # senal (cuanto separa las celdas) dividida por el ruido (lo que cambia
        # entre corridas).
        senal = float(np.nanstd(cons["dimensiones"].values.astype(float), ddof=1))
        fila["senal (sd entre celdas)"] = senal
        rep = fila["repetibilidad (MAE)"]
        fila["senal / ruido"] = senal / rep if rep and not np.isnan(rep) else np.nan
        if largo_referencia is not None:
            ref = consolidar(largo_referencia, error=error, agregador=fn)
            r = _acuerdo(ref, cons, mostrar=False)["resumen"]
            fila["acuerdo: sesgo"] = r["sesgo (sur - ref)"]
            fila["acuerdo: MAE"] = r["MAE"]
        filas.append(fila)

    t = pd.DataFrame(filas).set_index("agregador").round(3)
    if mostrar:
        _md("### Comparacion de agregadores")
        _display(t)
        if t["senal / ruido"].notna().any():
            mejor = t["senal / ruido"].idxmax()
            _md(f"Mejor relacion senal/ruido: **{mejor}** ({t.loc[mejor, 'senal / ruido']:.1f}). "
                "Mira esta columna y no solo la repetibilidad: un agregador que aplasta todas las "
                "celdas hacia el centro se repite perfecto y no mide nada. El acuerdo con el otro "
                "modo es informativo, pero no es el criterio de eleccion.")
    return t
