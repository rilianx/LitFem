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
                            formato=PTMVResult, model=MODELO_POR_DEFECTO, temperature=0.1,
                            mostrar=True):
    """
    Arma el prompt para UNA dimension (ej: "P", "T", "M", "Vn", "Vr" o "Vp")
    y devuelve el dict con los scores de los 3 momentos (A, B, C).

    mostrar=True (por defecto): imprime toda la informacion de esa dimension
    (tabla, reasoning de las 5 preguntas en los 3 momentos, promedios, analysis
    y conclusion). El analisis de las 6 dimensiones lo desactiva y usa la vista
    compacta.
    """
    prompt = instructions.format(
        dimensiones_texto=dimensiones[dimension],
        character=character,
        dimension=dimension,
        fragment=fragment,
    )
    resultado = llm_function(prompt, formato, model=model, temperature=temperature)
    if mostrar:
        mostrar_resultados_bonitos(resultado, completo=True)
    return resultado


def analizar_personaje_completo(character, fragment, instructions, dimensiones,
                                formato=PTMVResult, model=MODELO_POR_DEFECTO,
                                temperature=0.1, max_workers=6, excel=False,
                                mostrar=True, completo=False):
    """
    Analiza todas las dimensiones EN PARALELO (las llamadas al LLM son I/O, asi
    que los hilos sirven perfecto: 6 dimensiones tardan lo que la mas lenta).
    Los resultados se muestran despues, en el orden fijo de `dimensiones`, para
    que la salida no se mezcle entre hilos.

    completo=True muestra tambien el reasoning de todas las preguntas, el
    analysis y la conclusion de cada dimension (por defecto, vista compacta).

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
                mostrar=False,   # se muestra despues, ordenado y fuera de los hilos
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
            mostrar_resultados_bonitos(resultados_totales[dim], completo=completo)

    if excel:
        exportar_resultados_excel(
            resultados_totales, character,
            nombre_archivo=excel if isinstance(excel, str) else None,
        )

    return resultados_totales
