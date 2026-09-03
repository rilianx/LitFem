"""litfem — analisis literario PTM-V con LLM (funciones genericas).

Los prompts y las preguntas de cada dimension se definen fuera de la libreria
(en el notebook) y se pasan como argumento.
"""

from .llm import configurar_api_key, llm_function, MODELO_POR_DEFECTO, SYSTEM_STRUCTURED
from .schemas import PTMVResult, Answer, Moment, Scores, Puntaje
from .analysis import analizar_dimension_ptmv, analizar_personaje_completo
from .display import mostrar_resultados_bonitos, mostrar_resumen_global, promedios_por_momento
from .export import exportar_resultados_excel

__version__ = "0.1.0"

__all__ = [
    "configurar_api_key",
    "llm_function",
    "MODELO_POR_DEFECTO",
    "SYSTEM_STRUCTURED",
    "PTMVResult",
    "Answer",
    "Moment",
    "Scores",
    "Puntaje",
    "analizar_dimension_ptmv",
    "analizar_personaje_completo",
    "mostrar_resultados_bonitos",
    "mostrar_resumen_global",
    "promedios_por_momento",
    "exportar_resultados_excel",
]
