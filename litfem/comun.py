"""Utilidades compartidas por todos los modulos (sin dependencias internas).

- MOMENTOS / etapas_de_lista : el corte Inicio-Desarrollo-Final que usan todos los modos.
- es_error / a_numero        : helpers minimos de resultados y puntajes.
- _md / _display             : salida en notebook con fallback a print.
"""

__all__ = ["MOMENTOS", "etapas_de_lista", "etapa_por_indice", "es_error", "a_numero"]

# Los "momentos" del resumen (Moment_A/B/C) y las "etapas" del libro se nombran
# igual para que las tablas de todos los modos sean comparables celda a celda.
MOMENTOS = ["A", "B", "C"]


def etapas_de_lista(lista, prop_inicio=0.25, prop_final=0.25):
    """
    Corta una lista cronologica en Inicio / Desarrollo / Final segun proporciones
    (25% - 50% - 25% por defecto: con 20 elementos queda 5 - 10 - 5).
    Cada elemento cae en exactamente una etapa; con menos de 3 elementos las
    etapas sobrantes quedan vacias.
    """
    n = len(lista)
    if n == 0:
        return [], [], []
    n_ini = max(1, int(n * prop_inicio))
    n_fin = min(max(1, int(n * prop_final)), n - n_ini)
    return lista[:n_ini], lista[n_ini:n - n_fin], lista[n - n_fin:]


def etapa_por_indice(indices, prop_inicio=0.25, prop_final=0.25):
    """{indice: 'A'|'B'|'C'} para una lista de indices de fragmento (se ordena antes)."""
    idxs = sorted(indices)
    return {i: et for et, grupo in zip(MOMENTOS, etapas_de_lista(idxs, prop_inicio, prop_final))
            for i in grupo}


def es_error(res):
    """True si el resultado de una llamada no es un dict valido o trae 'error'."""
    return not isinstance(res, dict) or "error" in res


def a_numero(score):
    """float(score) o None si es N/A / Error / texto."""
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def _display(obj):
    try:
        from IPython.display import display
        display(obj)
    except ImportError:
        print(obj)


def _md(texto):
    try:
        from IPython.display import display, Markdown
        display(Markdown(texto))
    except ImportError:
        print(texto)
