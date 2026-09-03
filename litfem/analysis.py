"""Orquestacion del analisis PTM-V (una dimension o las 6 en paralelo).

Los prompts NO viven aca: se pasan desde el notebook, para poder experimentar
con ellos sin tocar la libreria.
  - `instructions`: template con los placeholders
      {dimensiones_texto}, {character}, {dimension}, {fragment}
  - `dimensiones`: dict {clave -> texto de las preguntas de esa dimension}
"""

from concurrent.futures import ThreadPoolExecutor

from .llm import llm_function, MODELO_POR_DEFECTO
from .schemas import PTMVResult
from .display import mostrar_resultados_bonitos
from .export import exportar_resultados_excel

__all__ = ["analizar_dimension_ptmv", "analizar_personaje_completo"]


def analizar_dimension_ptmv(character, dimension, fragment, instructions, dimensiones,
                            formato=PTMVResult, model=MODELO_POR_DEFECTO, temperature=0.1):
    """
    Arma el prompt para UNA dimension (ej: "P", "T", "M", "Vn", "Vr" o "Vp")
    y devuelve el dict con los scores de los 3 momentos (A, B, C).
    """
    prompt = instructions.format(
        dimensiones_texto=dimensiones[dimension],
        character=character,
        dimension=dimension,
        fragment=fragment,
    )
    return llm_function(prompt, formato, model=model, temperature=temperature)


def analizar_personaje_completo(character, fragment, instructions, dimensiones,
                                formato=PTMVResult, model=MODELO_POR_DEFECTO,
                                temperature=0.1, max_workers=6, excel=False,
                                mostrar=True):
    """
    Analiza todas las dimensiones EN PARALELO (las llamadas al LLM son I/O, asi
    que los hilos sirven perfecto: 6 dimensiones tardan lo que la mas lenta).
    Los resultados se muestran despues, en el orden fijo de `dimensiones`, para
    que la salida no se mezcle entre hilos.

    excel : False           -> no exporta
            True            -> exporta con nombre automatico
            "ruta/x.xlsx"   -> exporta con ese nombre

    Devuelve: dict {dimension -> resultado}
    """
    claves = list(dimensiones.keys())

    def _una(dim):
        try:
            return dim, analizar_dimension_ptmv(
                character, dim, fragment, instructions, dimensiones,
                formato=formato, model=model, temperature=temperature,
            )
        except Exception as e:
            return dim, {"error": f"{type(e).__name__}: {e}"}

    print(f"Analizando {len(claves)} dimensiones en paralelo...")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pares = list(pool.map(_una, claves))  # map conserva el orden

    resultados_totales = dict(pares)

    if mostrar:
        for dim in claves:
            print(f"DIMENSION {dim}...")
            mostrar_resultados_bonitos(resultados_totales[dim])

    if excel:
        exportar_resultados_excel(
            resultados_totales, character,
            nombre_archivo=excel if isinstance(excel, str) else None,
        )

    return resultados_totales
