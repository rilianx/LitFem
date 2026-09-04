"""Correr el corpus completo a partir de una planilla, de forma reanudable.

La planilla lista, por fila, el ARCHIVO (el mismo nombre en la carpeta de libros
y en la de resumenes) y el o los PERSONAJES a analizar en esa obra:

    archivo,personajes,rol
    alicia.txt,Alicia,protagonista
    orgullo.txt,"Elizabeth, Darcy",protagonista

Cualquier columna extra de la planilla (rol, genero, año...) se conserva y viaja
hasta los resultados: sirve para ver despues si el surrogado funciona distinto
segun el tipo de personaje.

Cada par (archivo, personaje) es un CASO. Dos personajes de la misma obra son dos
casos que COMPARTEN el texto: por eso el caso lleva siempre su `obra`, y la
validacion cruzada de `calibrar` debe agrupar por obra y no por caso, o entrena y
evalua con el mismo libro.

Flujo:
    man = leer_manifiesto("corpus.csv", CARPETA_LIBROS, CARPETA_RESUMENES)
    verificar_corpus(man)      # antes de gastar llamadas: momentos, tamanos, costo
    correr_corpus(man, ...)    # reanudable: lo ya hecho se saltea
    casos = cargar_corpus()
"""

import os
import re
import glob

import pandas as pd

from .comun import _md, _display
from .unidades import unidades_resumen, unidades_libro, dividir_en_fragmentos
from .puntuar import puntuar_unidades, iterar

__all__ = ["plantilla_manifiesto", "leer_manifiesto", "verificar_corpus",
           "correr_corpus", "cargar_corpus"]

_SEPARADOR = re.compile(r"\s*[;,/|]\s*")


def plantilla_manifiesto(ruta="corpus.csv"):
    """Escribe una planilla de ejemplo con las columnas esperadas."""
    pd.DataFrame([
        {"archivo": "alicia.txt", "personajes": "Alicia", "obra": "Alicia en el Pais de las Maravillas",
         "rol": "protagonista"},
        {"archivo": "orgullo.txt", "personajes": "Elizabeth, Darcy", "obra": "Orgullo y prejuicio",
         "rol": "protagonista"},
    ]).to_csv(ruta, index=False)
    print(f"Plantilla escrita en {ruta}. Columnas: archivo y personajes (obligatorias), "
          "obra (opcional), y las que quieras agregar (rol, genero...): se conservan.")
    return ruta


def leer_manifiesto(ruta, carpeta_libros, carpeta_resumenes):
    """
    Lee la planilla (.csv o .xlsx) y la normaliza a UNA FILA POR CASO
    (archivo, personaje), con las rutas resueltas y una marca de si existen.

    Acepta tanto una fila por obra con varios personajes en una celda, como una
    fila por personaje repitiendo el archivo.
    """
    leer = pd.read_excel if str(ruta).lower().endswith((".xlsx", ".xls")) else pd.read_csv
    df = leer(ruta)
    df.columns = [c.strip().lower() for c in df.columns]

    col_pers = next((c for c in ("personajes", "personaje", "character", "characters")
                     if c in df.columns), None)
    if "archivo" not in df.columns or col_pers is None:
        raise ValueError(f"La planilla necesita las columnas 'archivo' y 'personajes'. "
                         f"Tiene: {list(df.columns)}")

    filas = []
    for _, fila in df.iterrows():
        archivo = str(fila["archivo"]).strip()
        obra = str(fila.get("obra") or os.path.splitext(archivo)[0]).strip()
        for personaje in _SEPARADOR.split(str(fila[col_pers]).strip()):
            if not personaje or personaje.lower() == "nan":
                continue
            ruta_libro = os.path.join(carpeta_libros, archivo)
            ruta_resumen = os.path.join(carpeta_resumenes, archivo)
            extra = {c: fila[c] for c in df.columns
                     if c not in ("archivo", "obra", col_pers)}
            filas.append({
                "caso": f"{obra}__{personaje}".replace(" ", "_"),
                "obra": obra, "archivo": archivo, "personaje": personaje,
                **extra,          # columnas adicionales de la planilla (rol, genero, año...)
                "ruta_libro": ruta_libro, "ruta_resumen": ruta_resumen,
                "hay_libro": os.path.exists(ruta_libro),
                "hay_resumen": os.path.exists(ruta_resumen),
            })
    man = pd.DataFrame(filas)
    faltan = man[~(man["hay_libro"] & man["hay_resumen"])]
    if len(faltan):
        _md(f"**Faltan archivos en {len(faltan)} caso(s)** (se excluyen):")
        _display(faltan[["obra", "personaje", "archivo", "hay_libro", "hay_resumen"]])
    return man[man["hay_libro"] & man["hay_resumen"]].reset_index(drop=True)


def _leer(ruta):
    with open(ruta, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def verificar_corpus(manifiesto, max_words=2000, seleccion=(5, 10, 5),
                     n_a=5, n_b=3, patron=None, mostrar=True):
    """
    ANTES de gastar llamadas: revisa cada obra una sola vez (no por caso) y
    reporta si los momentos del resumen se detectan, cuantos fragmentos tiene el
    libro y cuantas llamadas costara todo.

    Un resumen sin sus 3 marcadores es el fallo mas caro de descubrir tarde.
    """
    kw = {"patron": patron} if patron else {}
    filas = []
    for obra, g in manifiesto.groupby("obra", sort=False):
        fila = {"obra": obra, "personajes": ", ".join(g["personaje"])}
        try:
            resumen = _leer(g["ruta_resumen"].iloc[0])
            fila["palabras resumen"] = len(resumen.split())
            us = unidades_resumen(resumen, verbose=False, **kw)
            fila["momentos"] = " ".join(f"{u['etapa']}:{len(u['texto'].split())}" for u in us)
            fila["momentos ok"] = all(u["meta"]["marcador"] != "(por posicion)" for u in us)
        except Exception as e:
            fila["momentos"] = f"ERROR: {e}"
            fila["momentos ok"] = False
        try:
            libro = _leer(g["ruta_libro"].iloc[0])
            fila["palabras libro"] = len(libro.split())
            todos = len(dividir_en_fragmentos(libro, max_words))
            fila["fragmentos"] = todos
            fila["unidades B"] = min(todos, sum(seleccion)) if seleccion else todos
        except Exception as e:
            fila["fragmentos"] = f"ERROR: {e}"
            fila["unidades B"] = 0
        filas.append(fila)

    t = pd.DataFrame(filas)
    casos = len(manifiesto)
    llamadas = int(casos * (3 * n_a + pd.to_numeric(
        t.set_index("obra").loc[manifiesto["obra"], "unidades B"], errors="coerce").fillna(0).mean() * n_b))

    if mostrar:
        _md(f"### Verificacion del corpus: {len(t)} obras, {casos} casos")
        _display(t)
        malas = t.loc[t["momentos ok"] == False, "obra"].tolist()
        if malas:
            _md(f"**Revisa los marcadores de momento en: {', '.join(malas)}.** Sin los 3 marcadores "
                "el resumen se corta por posicion, que no es lo mismo y ensucia la comparacion. "
                "Pasa `patron=[...]` con tus etiquetas o corrige el archivo.")
        else:
            _md("Todos los resumenes traen sus 3 momentos marcados.")
        _md(f"**Costo estimado: ~{llamadas:,} llamadas** ({n_a} iteraciones de A y {n_b} de B por caso).")
    return t


def correr_corpus(manifiesto, personaje_col="personaje", prompt=None, preguntas=None,
                  carpeta_salida="corpus_resultados", n_a=5, n_b=3, max_words=2000,
                  seleccion=(5, 10, 5), model=None, semilla_base=0, patron=None,
                  verbose=True, **kw):
    """
    Corre los modos A y B de cada caso y guarda un CSV por (caso, modo).

    **Reanudable**: si el CSV de un caso ya existe, se saltea. Colab se desconecta;
    volver a ejecutar la celda continua donde iba en vez de repetir 500 llamadas.

    Devuelve un DataFrame con el estado de cada caso.
    """
    if prompt is None or preguntas is None:
        raise ValueError("Pasa `prompt` (el de puntuacion) y `preguntas`.")
    os.makedirs(carpeta_salida, exist_ok=True)
    kw_res = {"patron": patron} if patron else {}
    model_kw = {"model": model} if model else {}

    estado, cache_texto = [], {}
    for i, fila in manifiesto.iterrows():
        caso, personaje = fila["caso"], fila[personaje_col]
        registro = {"caso": caso, "obra": fila["obra"], "personaje": personaje}
        for modo in ("A", "B"):
            ruta = os.path.join(carpeta_salida, f"{caso}_{modo}.csv")
            if os.path.exists(ruta):
                registro[modo] = "ya estaba"
                continue
            try:
                clave = (fila["obra"], modo)
                if clave not in cache_texto:
                    texto = _leer(fila["ruta_resumen"] if modo == "A" else fila["ruta_libro"])
                    cache_texto[clave] = (unidades_resumen(texto, verbose=False, **kw_res)
                                          if modo == "A" else
                                          unidades_libro(texto, max_words, seleccion, verbose=False))
                unidades = cache_texto[clave]
                n = n_a if modo == "A" else n_b
                if verbose:
                    print(f"[{i+1}/{len(manifiesto)}] {caso} · modo {modo}: "
                          f"{len(unidades)} unidades x {n} iteraciones")
                largo = iterar(
                    lambda it, s, u=unidades: puntuar_unidades(
                        u, personaje, prompt, preguntas, semilla=semilla_base + s,
                        iteracion=it, verbose=False, **model_kw, **kw),
                    num_iteraciones=n, verbose=False)
                largo.assign(caso=caso, obra=fila["obra"], personaje=personaje, modo=modo,
                             archivo=fila["archivo"]).to_csv(ruta, index=False)
                registro[modo] = f"{len(largo)} filas"
            except Exception as e:
                registro[modo] = f"ERROR: {type(e).__name__}: {e}"
                if verbose:
                    print(f"    fallo: {registro[modo]}")
        estado.append(registro)

    t = pd.DataFrame(estado)
    if verbose:
        _md("### Estado del corpus")
        _display(t)
        errores = t[t[["A", "B"]].apply(lambda c: c.astype(str).str.startswith("ERROR")).any(axis=1)]
        if len(errores):
            _md(f"**{len(errores)} caso(s) con error.** Corrige y vuelve a ejecutar: "
                "lo que ya se guardo no se repite.")
    return t


def cargar_corpus(carpeta_salida="corpus_resultados"):
    """
    Lee los CSV guardados y devuelve {caso: {"A": largo, "B": largo, "obra":, "personaje":}}.
    """
    casos = {}
    for ruta in sorted(glob.glob(os.path.join(carpeta_salida, "*_[AB].csv"))):
        largo = pd.read_csv(ruta)
        caso = largo["caso"].iloc[0]
        modo = largo["modo"].iloc[0]
        casos.setdefault(caso, {"obra": largo["obra"].iloc[0],
                                "personaje": largo["personaje"].iloc[0]})[modo] = largo
    completos = {k: v for k, v in casos.items() if "A" in v and "B" in v}
    print(f"{len(completos)} casos completos de {len(casos)} "
          f"({len(set(v['obra'] for v in completos.values()))} obras)")
    return completos
