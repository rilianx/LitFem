#!/usr/bin/env python3
"""
PTM-V · ¿Sirve el resumen como surrogado del libro?  (version script del notebook PTMV_Corpus20)

Corre cada obra del corpus de dos maneras y las compara celda a celda:
  A · resumen  cada momento marcado del resumen     (3 unidades  x N_A iteraciones)
  B · libro    fragmentos de ~2000 palabras 5-10-5  (20 unidades x N_B iteraciones)

Subcomandos (en este orden):
  verificar  revisa marcadores de momento y costo antes de gastar llamadas
  correr     corre A y B por caso, en paralelo; reanudable (saltea los CSV que ya existen)
  revisar    lista CSV con llamadas fallidas o faltantes (--borrar para rehacerlos)
  analizar   repetibilidad, acuerdo A-B, calibracion, diagnosticos; no gasta llamadas

Ejemplos:
  python scripts/ptmv_corpus.py verificar --manifiesto corpus.csv --raiz "/content/drive/MyDrive/TESTS GT/20 LIBROS"
  python scripts/ptmv_corpus.py correr    --manifiesto corpus.csv --raiz ... --salida resultados/corpus20_v4 --quien anto
  python scripts/ptmv_corpus.py revisar   --salida resultados/corpus20_v4
  python scripts/ptmv_corpus.py analizar  --salida resultados/corpus20_v4

Manifiesto (CSV, una fila por obra; rutas relativas a --raiz):
  carpeta,libro,resumen,obra,anio,personajes,responsable
  adeline_mowbray_1804,adeline_mowbray_libro.txt,adeline_mowbray_summary.txt,Adeline Mowbray,1804,Adeline,anto
  orgullo,orgullo.txt,orgullo_resumen.txt,Orgullo y prejuicio,1813,"Elizabeth; Darcy",vicente
`personajes` separa con ";". Cualquier columna extra viaja hasta los resultados.

La API key se lee de OPENAI_API_KEY (variable de entorno, Colab Secrets o .env).
"""
import argparse
import glob
import hashlib
import json
import os
import random
import runpy
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import litfem
import litfem.llm as _llm
import litfem.puntuar as _puntuar
from litfem import (agregador_por_dimension, agregar_celda, calibrar, cargar_corpus,
                    configurar_api_key, consolidar, correr_corpus, curva_unidades,
                    efecto_posicion, fiabilidad_preguntas, pares_de_consolidados,
                    parsear_preguntas, repetibilidad, acuerdo, verificar_corpus)

RAIZ_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── NO CAMBIAR: define la medicion; tiene que ser igual para todo el equipo ───
MEDICION = {
    "modelo": "gpt-5.4-mini",
    "semilla": 0,
    "n_a": 5,               # iteraciones del resumen (el modo ruidoso, el que va a produccion)
    "n_b": 3,               # iteraciones del libro
    "max_words": 2000,      # palabras por fragmento
    "seleccion": [5, 10, 5],  # fragmentos de inicio / desarrollo / final
}
TIPO_ERROR = "sem"
BANCO_POR_DEFECTO = os.path.join(RAIZ_REPO, "bancos", "ptmv_v4.py")

# Marcadores de momento: acepta "Momento A" y tambien "A. Introduction" / "B. The Body..." /
# "C. Finale..." (con o sin ### o **). El texto antes de "A." (metadatos del narrador) se descarta.
PATRON_MOMENTOS = (r"(?im)^[ \t#*>_-]*((?:moment[oa]?|stage|etapa)[ \t]*[:\-]?[ \t]*[ABC][ \t]*[:.\)\-]?"
                   r"|[ABC]\.[ \t]*(?:introduction|the body|finale)\b[^\n]*?)[ \t*_]*$")

# Agregacion de celda. T (impacto transformador) es una dimension de EVENTO: en el libro se
# agrega con un percentil alto entre fragmentos ("¿hizo algo decisivo en esta etapa?"). En el
# resumen la celda son repeticiones del MISMO texto, asi que A va con la media.
EVENTO = {"T"}
Q_EVENTO = 0.8
AGREGADOR_A = agregar_celda
AGREGADOR_B = agregador_por_dimension(evento=EVENTO, q=Q_EVENTO)


# ═══════════════════════════════ configuracion y firma ═══════════════════════════════

def cargar_banco(ruta):
    """El banco es un .py con PREGUNTAS_PTMV y PROMPT_PUNTUAR (ver bancos/)."""
    m = runpy.run_path(ruta)
    return m["PREGUNTAS_PTMV"], m["PROMPT_PUNTUAR"]


def firma_de(preguntas_ptmv, prompt):
    config = {"prompt": prompt, "preguntas": preguntas_ptmv, **MEDICION,
              "patron": PATRON_MOMENTOS, "litfem": litfem.__version__}
    config = json.loads(json.dumps(config))          # tuplas -> listas, igual que al leer el archivo
    firma = hashlib.sha1(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]
    return firma, config


def asegurar_firma(salida, firma, config):
    """La primera corrida deja la huella en `salida`; las siguientes deben coincidir."""
    ruta = os.path.join(salida, "_config.json")
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            previa = json.load(f)
        if previa["firma"] != firma:
            distintos = [k for k in config if config[k] != previa["config"].get(k)]
            sys.exit(f"Tu configuracion NO coincide con la de {salida} (difiere: {distintos}). "
                     "No cambies preguntas, prompt ni parametros: usa otra carpeta de salida.")
        print(f"✓ configuracion igual a la de {salida} (firma {firma})")
    else:
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump({"firma": firma, "config": config}, f, ensure_ascii=False, indent=1)
        print(f"Firma de configuracion creada: {firma}")


def sellar_csv(salida, casos, firma):
    """Anota la firma en cada CSV nuevo: asi `analizar` detecta corridas medidas distinto."""
    for caso in casos:
        for modo in "AB":
            ruta = os.path.join(salida, f"{caso}_{modo}.csv")
            if not os.path.exists(ruta):
                continue
            d = pd.read_csv(ruta, dtype=str, keep_default_na=False)   # deja "N/A" tal cual
            if "firma" not in d.columns:
                d.assign(firma=firma).to_csv(ruta, index=False)


# ═══════════════════════════════════ manifiesto ═══════════════════════════════════

def leer_manifiesto(ruta, raiz):
    m = pd.read_csv(ruta)
    m.columns = [c.strip().lower() for c in m.columns]
    faltan = {"carpeta", "libro", "resumen", "obra", "personajes"} - set(m.columns)
    if faltan:
        sys.exit(f"Al manifiesto le faltan columnas: {sorted(faltan)}")
    extra = [c for c in m.columns if c not in ("carpeta", "libro", "resumen", "obra", "personajes")]
    filas = []
    for _, r in m.iterrows():
        rl = os.path.join(raiz, str(r["carpeta"]), str(r["libro"]))
        rr = os.path.join(raiz, str(r["carpeta"]), str(r["resumen"]))
        for p in str(r["personajes"]).split(";"):
            p = p.strip()
            filas.append({"caso": f"{r['obra']}__{p}".replace(" ", "_"), "obra": r["obra"],
                          "archivo": r["libro"], "personaje": p, "ruta_libro": rl, "ruta_resumen": rr,
                          **{c: r[c] for c in extra},
                          "hay_libro": os.path.exists(rl), "hay_resumen": os.path.exists(rr)})
    man = pd.DataFrame(filas)
    malos = man[~(man["hay_libro"] & man["hay_resumen"])]
    for carpeta in sorted({os.path.dirname(x) for x in malos["ruta_libro"]}):
        print(f"✗ {carpeta}\n   contiene:",
              os.listdir(carpeta) if os.path.isdir(carpeta) else "LA CARPETA NO EXISTE")
    if len(malos):
        sys.exit("Corrige los nombres en el manifiesto (copia exacto lo que sale en 'contiene').")
    return man


def filtrar(man, quien, obras=None):
    if obras:
        pedidas = [o.strip().lower() for o in obras.split(",") if o.strip()]
        man = man[man["obra"].str.lower().isin(pedidas)]
        if man.empty:
            sys.exit(f"Ninguna obra del manifiesto coincide con {obras!r}.")
    if quien:
        if "responsable" not in man.columns:
            sys.exit("--quien necesita una columna `responsable` en el manifiesto.")
        man = man[man["responsable"].astype(str).str.strip().str.lower() == quien.lower()]
        if man.empty:
            sys.exit(f"Ningun caso con responsable = {quien!r}.")
    return man.reset_index(drop=True)


# ═══════════════════════════════════ subcomandos ═══════════════════════════════════

def cmd_verificar(a):
    man = filtrar(leer_manifiesto(a.manifiesto, a.raiz), a.quien, a.obras)
    ver = verificar_corpus(man, max_words=MEDICION["max_words"], seleccion=tuple(MEDICION["seleccion"]),
                           n_a=MEDICION["n_a"], n_b=MEDICION["n_b"], patron=PATRON_MOMENTOS, mostrar=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(ver.to_string(index=False))
    # Dos resumenes con los mismos conteos por momento casi seguro son el mismo archivo copiado.
    dup = ver[ver.duplicated("momentos", keep=False) & ver["momentos ok"].astype(bool)]
    if len(dup):
        print(f"\n⚠ Resumenes con conteos identicos (¿archivo copiado?): {', '.join(dup['obra'])}")
    if (ver["momentos ok"] != True).any():
        sys.exit("\n⛔ Hay resumenes sin sus 3 marcadores de momento: no corras todavia.")
    return ver


def cmd_correr(a):
    if not configurar_api_key(interactivo=False):
        sys.exit("Falta OPENAI_API_KEY (variable de entorno, Colab Secrets o .env).")
    preguntas_ptmv, prompt = cargar_banco(a.banco)
    preguntas = parsear_preguntas(preguntas_ptmv)
    man = filtrar(leer_manifiesto(a.manifiesto, a.raiz), a.quien, a.obras)
    cmd_verificar(a)

    os.makedirs(a.salida, exist_ok=True)
    firma, config = firma_de(preguntas_ptmv, prompt)
    asegurar_firma(a.salida, firma, config)

    # Reintentos con espera creciente (429, timeouts, cortes de red)
    original = _llm.llm_function

    def con_reintentos(*args, **kw):
        for intento in range(6):
            try:
                return original(*args, **kw)
            except Exception:
                if intento == 5:
                    raise
                time.sleep(min(60, 2 ** intento * 2 + random.random() * 3))
    _puntuar.llm_function = con_reintentos

    lock = threading.Lock()

    def un_caso(i):
        t0 = time.time()
        e = correr_corpus(man.loc[[i]], prompt=prompt, preguntas=preguntas, carpeta_salida=a.salida,
                          n_a=MEDICION["n_a"], n_b=MEDICION["n_b"], max_words=MEDICION["max_words"],
                          seleccion=tuple(MEDICION["seleccion"]), model=MEDICION["modelo"],
                          semilla_base=MEDICION["semilla"], patron=PATRON_MOMENTOS,
                          max_workers=a.hilos, verbose=False)
        r = e.iloc[0].to_dict()
        with lock:
            print(f"  ✓ {r['caso']}: A {r.get('A')} · B {r.get('B')}  ({(time.time() - t0) / 60:.1f} min)")
        return r

    print(f"\nCorriendo {len(man)} casos · {a.paralelo} a la vez × {a.hilos} llamadas...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.paralelo) as pool:
        estado = pd.DataFrame([f.result() for f in as_completed(
            [pool.submit(un_caso, i) for i in man.index])])
    print(f"\nListo en {(time.time() - t0) / 60:.1f} min")
    sellar_csv(a.salida, man["caso"], firma)
    print(estado.to_string(index=False))
    a.borrar = False
    cmd_revisar(a, casos=set(man["caso"]))


def cmd_revisar(a, casos=None):
    """Llamadas fallidas aun con reintentos quedan como "Error"; correr_corpus las SALTA."""
    malos, vistos = [], set()
    for f in sorted(glob.glob(os.path.join(a.salida, "*_[AB].csv"))):
        caso = os.path.basename(f).rsplit("_", 1)[0]
        if casos is not None and caso not in casos:
            continue
        vistos.add(caso)
        s = pd.read_csv(f, usecols=["score_txt"])["score_txt"].astype(str)
        n = int((s == "Error").sum())
        print(f"{'✗' if n else '✓'} {os.path.basename(f)}: {len(s)} filas · {n} con Error")
        if n:
            malos.append(f)
    esperados = casos if casos is not None else vistos
    faltan = sorted(f"{c}_{m}" for c in esperados for m in "AB"
                    if not os.path.exists(os.path.join(a.salida, f"{c}_{m}.csv")))
    if faltan:
        print("Faltan:", faltan)
    if malos and a.borrar:
        for f in malos:
            os.remove(f)
            print("borrado:", os.path.basename(f))
        print("Vuelve a ejecutar `correr` para rehacerlos.")
    print("Todo OK." if not (malos or faltan) else f"{len(malos)} CSV con errores · {len(faltan)} faltan.")


# ═══════════════════════════════════ analisis ═══════════════════════════════════

def chequear_firmas(casos):
    """Separa los casos medidos con otra configuracion (otra firma, o sin firma)."""
    firmas = {}
    for caso, d in casos.items():
        fs = set()
        for m in "AB":
            fs |= set(d[m]["firma"].dropna().astype(str)) if "firma" in d[m].columns else {"(sin firma)"}
        firmas[caso] = "/".join(sorted(fs))
    # CSV anteriores a la firma: el prompt v3+ obliga a empezar con "Material: yes/no".
    for caso, d in casos.items():
        if firmas[caso] == "(sin firma)":
            con_paso = pd.concat([d["A"], d["B"]])["reasoning"].astype(str).str.startswith("Material").mean()
            firmas[caso] = "(sin firma, prompt v3+)" if con_paso > 0.9 else "(sin firma, prompt < v3)"
    cuenta = pd.Series(firmas).value_counts()
    if len(cuenta) > 1:
        print("⚠ La carpeta mezcla configuraciones:")
        for f, n in cuenta.items():
            print(f"   {f}: {n} casos — {', '.join(c for c, x in firmas.items() if x == f)}")
    return firmas


def agregadores_por_caso(casos, agregadores):
    """
    Compara agregadores de B CASO POR CASO. Consolidar el corpus entero como una sola obra (lo
    que hace `comparar_agregadores(largo_B)`) junta ~150 fragmentos por celda: el maximo sale
    casi siempre 5 y parece perfectamente repetible por artefacto.
    """
    filas = []
    for nombre, fn in agregadores.items():
        T, ruido_T, dif_TA, rep = [], [], [], []
        pares = {}
        for caso, d in casos.items():
            cA = consolidar(d["A"], error=TIPO_ERROR, agregador=AGREGADOR_A)
            cB = consolidar(d["B"], error=TIPO_ERROR, agregador=fn)
            pares[caso] = (cA, cB)
            r = repetibilidad(d["B"], mostrar=False, error=TIPO_ERROR, agregador=fn)
            rep.append(r["mae_medio"])
            ruido_T.append(float(np.nanmean(r["sd_por_celda"].loc["T"].values)))
            T.append(cB["dimensiones"].loc["T"].values)
            dif_TA.append((cB["dimensiones"].loc["T"] - cA["dimensiones"].loc["T"]).values)
        T, dif_TA = np.vstack(T), np.vstack(dif_TA)
        sd_entre = float(np.nanmean(np.nanstd(T, axis=0, ddof=1)))
        meta = {c: {"obra": d["obra"], "personaje": d["personaje"]} for c, d in casos.items()}
        cal = calibrar(pares_de_consolidados(pares, "A", "B", metadatos=meta),
                       surrogado="A", referencia="B", grupo="obra", mostrar=False)["tabla"]
        filas.append({"agregador": nombre,
                      "T media (B)": np.nanmean(T),
                      "% celdas T >= 4.9 (techo)": 100 * np.mean(T >= 4.9),
                      "T: sd entre personajes": sd_entre,
                      "T: ruido entre corridas": np.nanmean(ruido_T),
                      "T: discriminacion (sd/ruido)": sd_entre / np.nanmean(ruido_T),
                      "T: sesgo B-A": np.nanmean(dif_TA),
                      "T: |dif| B-A": np.nanmean(np.abs(dif_TA)),
                      "repetibilidad B (todas)": np.mean(rep),
                      "MAE val. sin calibrar": cal.loc["sin calibrar", "MAE validacion"],
                      "MAE val. lineal x dim": cal.loc["lineal por dimension", "MAE validacion"]})
    return pd.DataFrame(filas).set_index("agregador").round(3)


def tabla_na(largo):
    """Por pregunta: tasa de N/A y N/A inconsistente partido en bajo / medio / alto."""
    filas = []
    for (dim, q), g in largo.groupby(["dimension", "pregunta"], sort=False):
        por_u = g.groupby(["caso", "fuente", "momento"])["score"]
        inc = por_u.apply(lambda s: s.isna().any() and s.notna().any())
        mx, mn = por_u.max()[inc], por_u.min()[inc]
        n_u = len(inc)
        filas.append({"id": f"{dim}{q[1:]}", "dimension": dim,
                      "tasa N/A": g["score"].isna().mean(),
                      "N/A inconsistente": inc.mean(),
                      "↔ bajo (1-2)": (mx <= 2).sum() / n_u,
                      "↔ alto (4-5)": (mn >= 4).sum() / n_u,
                      "↔ 3 o mixto": ((mx > 2) & (mn < 4)).sum() / n_u})
    orden = [f"{d}{k}" for d in ("P", "T", "M", "Vn", "Vr", "Vp") for k in range(1, 6)]
    return pd.DataFrame(filas).set_index("id").reindex(orden)


def cmd_analizar(a):
    casos = cargar_corpus(a.salida)
    if not casos:
        sys.exit(f"No hay casos completos en {a.salida}.")
    firmas = chequear_firmas(casos)
    if a.solo_firma:
        casos = {c: d for c, d in casos.items() if a.solo_firma in firmas[c]}
        print(f"Analizo solo la firma {a.solo_firma}: {len(casos)} casos")
    out = a.analisis or os.path.join(a.salida, "analisis")
    os.makedirs(out, exist_ok=True)
    guardar = lambda df, nombre, **kw: (df.to_csv(os.path.join(out, nombre), **kw), print(f"  → {nombre}"))
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    consolidados, metadatos, filas = {}, {}, []
    for caso, d in casos.items():
        consolidados[caso] = (consolidar(d["A"], error=TIPO_ERROR, agregador=AGREGADOR_A),
                              consolidar(d["B"], error=TIPO_ERROR, agregador=AGREGADOR_B))
        metadatos[caso] = {"obra": d["obra"], "personaje": d["personaje"]}
        fila = {"caso": caso, **metadatos[caso], "firma": firmas[caso]}
        for modo, agg in (("A", AGREGADOR_A), ("B", AGREGADOR_B)):
            try:
                fila[f"repet. {modo}"] = repetibilidad(d[modo], mostrar=False, error=TIPO_ERROR,
                                                       agregador=agg)["mae_medio"]
            except ValueError:
                fila[f"repet. {modo}"] = np.nan
        r = acuerdo(consolidados[caso][1], consolidados[caso][0], "B", "A", mostrar=False)["resumen"]
        fila.update({"sesgo": r["sesgo (sur - ref)"], "MAE": r["MAE"], "r": r["r (Pearson)"]})
        filas.append(fila)
    largo_A = pd.concat([d["A"] for d in casos.values()], ignore_index=True)
    largo_B = pd.concat([d["B"] for d in casos.values()], ignore_index=True)

    # 4.1-4.2 · repetibilidad (techo) y acuerdo A-B por caso
    por_caso = pd.DataFrame(filas).set_index("caso")
    por_caso["techo A-B"] = np.hypot(por_caso["repet. A"], por_caso["repet. B"])
    por_caso = por_caso.round(3)
    print("\n## Acuerdo resumen (A) vs libro (B), por caso\n", por_caso.drop(columns="firma").to_string())
    print(f"\nsesgo medio {por_caso['sesgo'].mean():+.3f} (sd {por_caso['sesgo'].std():.3f}) · "
          f"MAE medio {por_caso['MAE'].mean():.3f} · techo medio {por_caso['techo A-B'].mean():.3f}")
    guardar(por_caso, "acuerdo_por_caso.csv")

    # 4.3 · calibracion con validacion cruzada dejando fuera la obra completa
    pares = pares_de_consolidados(consolidados, "A_resumen", "B_libro", metadatos=metadatos)
    cal = calibrar(pares, surrogado="A_resumen", referencia="B_libro", grupo="obra", mostrar=False)
    print(f"\n## Calibracion ({pares['obra'].nunique()} obras, {len(pares)} celdas)\n", cal["tabla"].to_string())
    guardar(pares, "pares.csv", index=False)
    guardar(cal["tabla"], "calibracion.csv")

    # 4.4 · diagnosticos
    pos = efecto_posicion(largo_B, control="celda", mostrar=False)
    print("\n## Efecto de la posicion en el prompt (libro)\n", pos.round(3).to_string())
    guardar(pos, "efecto_posicion.csv")
    ruido = fiabilidad_preguntas(largo_B, mostrar=False, top=30)
    guardar(ruido, "fiabilidad_preguntas.csv")
    curva = curva_unidades(largo_B, error=TIPO_ERROR, agregador=AGREGADOR_B, mostrar=False)
    print("\n## Curva de saturacion (fragmentos de B)\n", curva.to_string())
    guardar(curva, "curva_unidades.csv")

    agg = agregadores_por_caso(casos, {
        "media": agregar_celda,
        "T p70": agregador_por_dimension({"T"}, q=0.7),
        "T p80 (actual)": AGREGADOR_B,
        "T p90": agregador_por_dimension({"T"}, q=0.9),
        "T max": agregador_por_dimension({"T"}, q=1.0)})
    print("\n## Agregadores de T (caso por caso)\n", agg.T.to_string())
    guardar(agg, "agregadores_T.csv")

    # 4.6 · discrepancias de N/A
    na_B, na_A = tabla_na(largo_B), tabla_na(largo_A)
    print("\n## N/A inconsistente (libro), peores 8 · meta <= 0.15\n",
          na_B.sort_values("N/A inconsistente", ascending=False).head(8).round(3).to_string())
    guardar(na_B.join(na_A[["tasa N/A"]].rename(columns={"tasa N/A": "tasa N/A resumen"})), "na_por_pregunta.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        from litfem import grafico_acuerdo_corpus
        fig, _ = grafico_acuerdo_corpus(pares, surrogado="A_resumen", referencia="B_libro", color="dimension",
                                        titulo="Acuerdo resumen vs libro · todas las celdas del corpus")
        if fig is not None:
            fig.savefig(os.path.join(out, "acuerdo_corpus.png"), dpi=160, bbox_inches="tight")
            print("  → acuerdo_corpus.png")
    except ImportError:
        print("(sin matplotlib: omito los graficos)")
    print(f"\nResultados en {out}")


# ═══════════════════════════════════════ CLI ═══════════════════════════════════════

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def con_manifiesto(sp):
        sp.add_argument("--manifiesto", required=True, help="CSV con una fila por obra")
        sp.add_argument("--raiz", default=".", help="carpeta base de las rutas del manifiesto")
        sp.add_argument("--quien", help="corre solo las filas con este `responsable`")
        sp.add_argument("--obras", help="corre solo estas obras (nombres separados por coma)")

    sp = sub.add_parser("verificar", help="marcadores de momento y costo, sin llamadas")
    con_manifiesto(sp)
    sp.set_defaults(fn=cmd_verificar)

    sp = sub.add_parser("correr", help="corre A y B (reanudable)")
    con_manifiesto(sp)
    sp.add_argument("--salida", required=True, help="carpeta de resultados, compartida por el equipo")
    sp.add_argument("--banco", default=BANCO_POR_DEFECTO, help="banco de preguntas + prompt (.py)")
    sp.add_argument("--paralelo", type=int, default=3, help="casos simultaneos (bajar si hay 429)")
    sp.add_argument("--hilos", type=int, default=8, help="llamadas simultaneas dentro de cada caso")
    sp.set_defaults(fn=cmd_correr)

    sp = sub.add_parser("revisar", help="CSV con errores o faltantes")
    sp.add_argument("--salida", required=True)
    sp.add_argument("--borrar", action="store_true", help="borra los CSV con errores para rehacerlos")
    sp.set_defaults(fn=cmd_revisar)

    sp = sub.add_parser("analizar", help="analisis completo sobre los CSV guardados")
    sp.add_argument("--salida", required=True)
    sp.add_argument("--analisis", help="donde escribir tablas y graficos (por defecto <salida>/analisis)")
    sp.add_argument("--solo-firma", help="analiza solo los casos con esta firma de configuracion")
    sp.set_defaults(fn=cmd_analizar)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
