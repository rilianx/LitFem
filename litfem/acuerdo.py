"""¿Sirve un modo como SURROGADO de otro?

No basta con poner dos tablas lado a lado: la pregunta "puedo usar el resumen en
vez del libro" es una pregunta de ACUERDO entre dos medidas, y se responde con
sesgo, error tipico y limites de acuerdo, no con una correlacion.

Aviso sobre la correlacion: con 18 celdas casi todas entre 3 y 5, r mide sobre
todo si las dos medidas separan las dimensiones altas de las bajas. Si el acuerdo
depende de dos dimensiones extremas, r sube aunque el resto no coincida; por eso
aqui r se reporta junto al sesgo y al error, nunca solo.
"""

import numpy as np
import pandas as pd

from .comun import _md, _display

__all__ = ["acuerdo", "grafico_acuerdo"]

# Paleta categorica validada (6 slots, orden fijo) y marcadores por momento:
# la identidad nunca queda solo en el color.
_COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
_MARCADORES = {"A": "o", "B": "s", "C": "^"}


def acuerdo(referencia, surrogado, nombre_ref="Referencia", nombre_sur="Surrogado",
            umbral=0.5, mostrar=True, grafico=False):
    """
    Compara celda a celda (dimension x momento) dos dicts de `consolidar`.

    referencia : el modo que se toma como patron (p. ej. el libro completo)
    surrogado  : el modo barato que se quiere usar en su lugar (p. ej. el resumen)
    umbral     : cuanta diferencia se considera tolerable, en puntos de la escala

    Devuelve {"resumen", "por_celda", "por_dimension"}:
      resumen       : sesgo, MAE, RMSE, limites de acuerdo, r, rho, calibracion
      por_celda     : diferencia de cada celda y, si ambos traen error, el z
                      (diferencia / error combinado): |z| >= 2 = mayor que el ruido
      por_dimension : |diferencia| media por dimension, para ver si el desacuerdo
                      se concentra en unas pocas
    """
    ref, sur = referencia["dimensiones"], surrogado["dimensiones"]
    dims = [d for d in sur.index if d in ref.index]
    ref, sur = ref.loc[dims], sur.loc[dims]
    dif = sur - ref

    x, y = ref.values.ravel(), sur.values.ravel()
    d = dif.values.ravel()
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y, d = x[ok], y[ok], d[ok]
    sd = d.std(ddof=1) if len(d) > 1 else np.nan

    # calibracion: surrogado ~ a + b * referencia
    b, a = np.polyfit(x, y, 1) if len(x) > 2 else (np.nan, np.nan)

    resumen = pd.Series({
        "n celdas": len(d),
        "sesgo (sur - ref)": d.mean(),
        "MAE": np.abs(d).mean(),
        "RMSE": np.sqrt((d ** 2).mean()),
        "limite inf. de acuerdo": d.mean() - 1.96 * sd,
        "limite sup. de acuerdo": d.mean() + 1.96 * sd,
        f"celdas con |dif| <= {umbral}": float((np.abs(d) <= umbral).mean()),
        "r (Pearson)": np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan,
        "rho (Spearman)": pd.Series(x).corr(pd.Series(y), method="spearman"),
        "calibracion: pendiente": b,
        "calibracion: intercepto": a,
    }).round(3)

    por_celda = dif.round(2)
    # z = diferencia / error combinado. Basta con que UNO de los dos traiga error
    # (entonces mide la diferencia en unidades del ruido de ese modo).
    cero = pd.DataFrame(0.0, index=dims, columns=ref.columns)
    e_ref = referencia.get("dimensiones_err")
    e_sur = surrogado.get("dimensiones_err")
    e_ref = cero if e_ref is None else e_ref.loc[dims].fillna(0)
    e_sur = cero if e_sur is None else e_sur.loc[dims].fillna(0)
    comb = np.sqrt(e_ref ** 2 + e_sur ** 2).replace(0, np.nan)
    err = (dif / comb).round(1) if comb.notna().any().any() else None
    solo_uno = (referencia.get("dimensiones_err") is None) != (surrogado.get("dimensiones_err") is None)

    por_dimension = pd.DataFrame({
        "|dif| media": dif.abs().mean(axis=1).round(2),
        "dif media": dif.mean(axis=1).round(2),
        f"{nombre_ref} (media)": ref.mean(axis=1).round(2),
        f"{nombre_sur} (media)": sur.mean(axis=1).round(2),
    }).sort_values("|dif| media", ascending=False)

    if mostrar:
        _md(f"## ¿Sirve **{nombre_sur}** como surrogado de **{nombre_ref}**?")
        _display(resumen.to_frame("valor"))
        _md(f"### Diferencia por celda ({nombre_sur} − {nombre_ref})")
        _display(por_celda)
        if err is not None:
            _md("### La diferencia, en unidades de error combinado (|z| ≥ 2 = mayor que el ruido)")
            _display(err)
            _md(f"> {int((err.abs() >= 2).sum().sum())} de {len(d)} celdas difieren mas que el ruido."
                + (" Solo uno de los dos modos trae error, asi que el z usa ese: itera tambien el"
                   " otro para que la comparacion sea simetrica." if solo_uno else ""))
        else:
            _md("> Ninguno de los dos trae error por celda: **itera ambos modos** antes de concluir "
                "nada, o no sabras si las diferencias son reales o ruido de una sola corrida.")
        _md("### Dónde se concentra el desacuerdo")
        _display(por_dimension)

        sesgo, mae = resumen["sesgo (sur - ref)"], resumen["MAE"]
        peor = por_dimension.index[0]
        sin_peor = dif.drop(index=por_dimension.index[:2]).abs().mean().mean()
        _md(f"**Lectura.** Sesgo {sesgo:+.2f} puntos y error tipico {mae:.2f} en una escala de 1 a 5; "
            f"el 95 % de las diferencias cae entre {resumen['limite inf. de acuerdo']:+.2f} y "
            f"{resumen['limite sup. de acuerdo']:+.2f}. El desacuerdo se concentra en **{peor}** "
            f"(|dif| media {por_dimension['|dif| media'].iloc[0]:.2f}); quitando las dos peores "
            f"dimensiones el error baja a {sin_peor:.2f}. "
            + ("Un sesgo casi constante se puede corregir: mira la calibracion."
               if abs(sesgo) > umbral and resumen["calibracion: pendiente"] > 0.7 else ""))

    if grafico:
        grafico_acuerdo(ref, sur, nombre_ref, nombre_sur)

    return {"resumen": resumen, "por_celda": por_celda, "z": err, "por_dimension": por_dimension}


def grafico_acuerdo(ref, sur, nombre_ref="Referencia", nombre_sur="Surrogado", umbral=0.5):
    """
    Dos paneles: identidad (surrogado vs referencia, con la recta y = x) y
    Bland-Altman (diferencia vs media, con sesgo y limites de acuerdo).
    Color = dimension (orden fijo), forma = momento: la identidad nunca depende
    solo del color.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no disponible; omito el grafico.")
        return None

    dims = list(ref.index)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.patch.set_facecolor("#fcfcfb")

    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color="#e6e5e2", linewidth=0.8)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color("#c9c8c4")
        ax.tick_params(colors="#52514e", labelsize=9)

    lim = (0.8, 5.2)
    ax1.plot(lim, lim, color="#c9c8c4", linewidth=2, zorder=1)
    for k, dim in enumerate(dims):
        color = _COLORES[k % len(_COLORES)]
        for momento in ref.columns:
            x, y = ref.loc[dim, momento], sur.loc[dim, momento]
            if pd.isna(x) or pd.isna(y):
                continue
            ax1.scatter(x, y, s=90, color=color, marker=_MARCADORES.get(momento, "o"),
                        edgecolor="#fcfcfb", linewidth=2, zorder=3,
                        label=dim if momento == ref.columns[0] else None)
            if abs(y - x) > 1.0:                      # etiqueta solo los desacuerdos grandes
                ax1.annotate(f"{dim}{momento}", (x, y), textcoords="offset points",
                             xytext=(7, 4), fontsize=8, color="#52514e")
    ax1.set(xlim=lim, ylim=lim, xlabel=f"{nombre_ref} (puntaje)", ylabel=f"{nombre_sur} (puntaje)")
    ax1.set_title(f"{nombre_sur} vs {nombre_ref}   ·   la linea es el acuerdo perfecto",
                  fontsize=10, color="#0b0b0b", loc="left")
    ax1.legend(frameon=False, fontsize=8, labelcolor="#52514e", ncol=3, loc="upper left")

    medias = ((ref + sur) / 2).values.ravel()
    difs = (sur - ref).values.ravel()
    ok = ~(np.isnan(medias) | np.isnan(difs))
    medias, difs = medias[ok], difs[ok]
    sesgo, sd = difs.mean(), difs.std(ddof=1)

    ax2.axhline(0, color="#c9c8c4", linewidth=2)
    ax2.axhline(sesgo, color="#52514e", linewidth=2, linestyle="--")
    for lim_ac in (sesgo - 1.96 * sd, sesgo + 1.96 * sd):
        ax2.axhline(lim_ac, color="#c9c8c4", linewidth=1.5, linestyle=":")
    ax2.axhspan(sesgo - 1.96 * sd, sesgo + 1.96 * sd, color="#52514e", alpha=0.05)
    for k, dim in enumerate(dims):
        color = _COLORES[k % len(_COLORES)]
        for momento in ref.columns:
            m = (ref.loc[dim, momento] + sur.loc[dim, momento]) / 2
            df_ = sur.loc[dim, momento] - ref.loc[dim, momento]
            if pd.isna(m):
                continue
            ax2.scatter(m, df_, s=90, color=color, marker=_MARCADORES.get(momento, "o"),
                        edgecolor="#fcfcfb", linewidth=2, zorder=3)
    ax2.annotate(f"sesgo {sesgo:+.2f}", (lim[1], sesgo), textcoords="offset points",
                 xytext=(-4, 5), ha="right", fontsize=8, color="#52514e")
    ax2.annotate(f"limites de acuerdo ±{1.96*sd:.2f}", (lim[0], sesgo + 1.96 * sd),
                 textcoords="offset points", xytext=(4, 4), fontsize=8, color="#52514e")
    ax2.set(xlim=lim, xlabel="Media de los dos modos", ylabel=f"{nombre_sur} − {nombre_ref}")
    ax2.set_title("Bland-Altman   ·   cuanto se desvia y en que rango de puntaje",
                  fontsize=10, color="#0b0b0b", loc="left")

    fig.text(0.5, -0.02, "Forma: ● momento A   ■ momento B   ▲ momento C",
             ha="center", fontsize=8, color="#52514e")
    fig.tight_layout()
    return fig
