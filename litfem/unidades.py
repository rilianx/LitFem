"""Construccion de UNIDADES de analisis.

Una unidad es lo que ve el modelo en UNA llamada:
    {"id": ..., "etapa": "A"|"B"|"C", "texto": ..., "meta": {...}}

Los tres modos solo se diferencian en como se construyen las unidades:
  A · resumen  : 3 unidades, los momentos que el propio resumen trae marcados
  B · libro    : N unidades, fragmentos de ~max_words palabras (muestra 5-10-5)
  C · juez     : 3 unidades, la evidencia extraida de los fragmentos de cada etapa

De ahi en adelante el camino es identico (`puntuar.puntuar_unidades`).
"""

import math
import re

from .comun import MOMENTOS, etapas_de_lista, etapa_por_indice

__all__ = ["unidades_resumen", "unidades_libro", "dividir_en_fragmentos",
           "seleccionar_fragmentos", "PATRON_MOMENTOS"]

# Reconoce cabeceras tipo "MOMENT A", "Momento B:", "## Stage C", "PARTE 1", "### A)"
PATRON_MOMENTOS = r"(?im)^[\s#*>_-]*(?:moment[oa]?|stage|etapa|parte|part|section)?\s*[:\-]?\s*\(?([ABC]|[123])\)?\s*[:.\)\-]?\s*$"


def unidades_resumen(texto, patron=PATRON_MOMENTOS, etapas=MOMENTOS, verbose=True):
    """
    Parte el resumen en 3 unidades usando los marcadores de momento que el texto
    ya trae. Si no encuentra exactamente 3 marcadores, cae a un corte por
    posicion (25/50/25) y lo avisa.

    `patron`: regex con UN grupo de captura (la etiqueta del momento). Tambien
    acepta una lista de literales, p. ej. ["MOMENT A", "MOMENT B", "MOMENT C"].
    """
    if isinstance(patron, (list, tuple)):
        patron = "(?im)^.*(" + "|".join(re.escape(p) for p in patron) + ").*$"

    cortes = list(re.finditer(patron, texto))
    if len(cortes) == len(etapas):
        unidades = []
        for k, (m, etapa) in enumerate(zip(cortes, etapas)):
            fin = cortes[k + 1].start() if k + 1 < len(cortes) else len(texto)
            unidades.append({"id": etapa, "etapa": etapa,
                             "texto": texto[m.end():fin].strip(),
                             "meta": {"marcador": m.group(0).strip()}})
        if verbose:
            print("Momentos detectados en el resumen:")
            for u in unidades:
                print(f"  {u['etapa']}: '{u['meta']['marcador']}' -> {len(u['texto'].split()):,} palabras")
    else:
        if verbose:
            print(f"AVISO: se encontraron {len(cortes)} marcadores de momento (se esperaban "
                  f"{len(etapas)}); se corta el resumen por posicion 25/50/25.")
        partes = etapas_de_lista(texto.split())
        unidades = [{"id": e, "etapa": e, "texto": " ".join(p), "meta": {"marcador": "(por posicion)"}}
                    for e, p in zip(etapas, partes)]
        if verbose:
            for u in unidades:
                print(f"  {u['etapa']}: {len(u['texto'].split()):,} palabras")

    vacias = [u["etapa"] for u in unidades if not u["texto"].strip()]
    if vacias:
        raise ValueError(f"Momentos vacios tras el corte: {vacias}. Revisa `patron`.")
    return unidades


def dividir_en_fragmentos(texto, max_words=2000):
    """[(indice_desde_1, texto), ...] con fragmentos parejos de ~max_words palabras."""
    palabras = texto.split()
    if not palabras:
        return []
    tam = math.ceil(len(palabras) / math.ceil(len(palabras) / max_words))
    return [(k + 1, " ".join(palabras[i:i + tam])) for k, i in enumerate(range(0, len(palabras), tam))]


def _equiespaciados(lista, cantidad):
    if cantidad <= 0 or not lista:
        return []
    if len(lista) <= cantidad:
        return lista
    if cantidad == 1:
        return [lista[0]]
    paso = (len(lista) - 1) / (cantidad - 1)
    return [lista[round(i * paso)] for i in range(cantidad)]


def seleccionar_fragmentos(fragmentos, n_inicio=5, n_desarrollo=10, n_final=5):
    """
    Si hay mas fragmentos que la muestra pedida, corta el libro en Inicio /
    Desarrollo / Final con las proporciones de la muestra (5-10-5 -> 25/50/25,
    el mismo corte que `etapas_de_lista`) y toma muestras equiespaciadas de cada
    etapa; si no, devuelve todos.
    """
    total = n_inicio + n_desarrollo + n_final
    if total == 0 or len(fragmentos) <= total:
        return fragmentos
    ini, des, fin = etapas_de_lista(fragmentos, n_inicio / total, n_final / total)
    return (_equiespaciados(ini, n_inicio) + _equiespaciados(des, n_desarrollo)
            + _equiespaciados(fin, n_final))


def unidades_libro(texto, max_words=2000, seleccion=(5, 10, 5), verbose=True):
    """Divide el libro, selecciona la muestra y etiqueta cada fragmento con su etapa."""
    todos = dividir_en_fragmentos(texto, max_words)
    elegidos = seleccionar_fragmentos(todos, *seleccion) if seleccion else todos
    etapa_de = etapa_por_indice([i for i, _ in elegidos])
    if verbose:
        print(f"{len(elegidos)} fragmentos seleccionados de {len(todos)} (~{max_words} palabras c/u)")
    return [{"id": i, "etapa": etapa_de[i], "texto": t, "meta": {}} for i, t in elegidos]
