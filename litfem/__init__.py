"""litfem — analisis literario PTM-V con LLM.

UN solo camino: se construyen UNIDADES, se puntuan con TODAS las preguntas
barajadas (una llamada por unidad), y el resultado ya es el formato largo que
consume `consolidar`. Los tres modos solo difieren en las unidades:

    A · resumen   unidades_resumen(texto)                ->  3 llamadas
    B · libro     unidades_libro(texto)                  -> 20 llamadas
    C · evidencia extraer_evidencia(unidades_libro(...)) -> 20 llamadas
                  + unidades_evidencia(df)               ->  3 llamadas

Los prompts y las preguntas por dimension viven en el notebook.
"""

from .comun import MOMENTOS, etapas_de_lista, etapa_por_indice
from .llm import configurar_api_key, llm_function, MODELO_POR_DEFECTO, SYSTEM_STRUCTURED
from .schemas import Answer, Puntaje
from .preguntas import parsear_preguntas, barajar, formatear_preguntas, esquema_puntajes
from .unidades import (unidades_resumen, unidades_libro, dividir_en_fragmentos,
                       seleccionar_fragmentos, PATRON_MOMENTOS)
from .puntuar import puntuar_unidades, iterar, COLUMNAS_LARGO
from .evidencia import (Evidence, extraer_evidencia, resumen_evidencia, mostrar_evidencia,
                        formatear_evidencia, unidades_evidencia, agregar_jackknife)
from .display import (tabla_cruda, mostrar_razonamientos, mostrar_na,
                      residuos_posicion, efecto_posicion)
from .agregacion import (unir_largos, agregar_celda, combinar_preguntas, consolidar,
                         formatear_con_error, mostrar_consolidado, comparar_consolidados,
                         comparar_modos)
from .acuerdo import acuerdo, grafico_acuerdo
from .calibracion import pares_de_consolidados, calibrar, MODELOS
from .validacion import (repetibilidad, fiabilidad_preguntas, curva_unidades,
                         comparar_agregadores)
from .export import exportar_largo_excel, exportar_consolidado_excel

__version__ = "0.10.0"

__all__ = [n for n in dir() if not n.startswith("_") and n not in
           ("comun", "llm", "schemas", "preguntas", "unidades", "puntuar",
            "evidencia", "display", "agregacion", "validacion", "calibracion", "export")]
