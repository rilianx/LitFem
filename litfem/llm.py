"""Cliente LLM con salida estructurada (LangChain + OpenAI)."""

import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

__all__ = ["configurar_api_key", "llm_function", "SYSTEM_STRUCTURED", "MODELO_POR_DEFECTO"]

MODELO_POR_DEFECTO = "gpt-5.4-mini"

# El esquema viaja por la API (Structured Outputs), NO dentro del prompt:
# el system message solo recuerda que no deje campos vacios.
SYSTEM_STRUCTURED = (
    "You are a literary analysis judge. Fill in every field of the required "
    "output schema; never leave a field empty."
)

_LLM_CACHE = {}


def configurar_api_key(var="OPENAI_API_KEY", interactivo=True):
    """
    Carga la API key desde (en orden): variable de entorno ya definida,
    Colab Secrets, archivo .env y, como ultimo recurso, getpass.

    Nunca escribe la clave en el notebook. Devuelve True si quedo cargada.
    """
    if os.getenv(var):
        return True

    # 1) Colab Secrets (icono de llave)
    try:
        from google.colab import userdata  # type: ignore

        valor = userdata.get(var)
        if valor:
            os.environ[var] = valor
            return True
    except Exception:
        pass

    # 2) archivo .env
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except ImportError:
        pass

    # 3) fallback interactivo
    if not os.getenv(var) and interactivo:
        from getpass import getpass

        os.environ[var] = getpass(f"{var}: ")

    return bool(os.getenv(var))


def _get_llm(model, temperature=0.1):
    """Reutiliza el mismo cliente para no recrearlo en cada llamada."""
    clave = (model, temperature)
    if clave not in _LLM_CACHE:
        _LLM_CACHE[clave] = ChatOpenAI(model=model, temperature=temperature)
    return _LLM_CACHE[clave]


def llm_function(prompt, esquema, model=MODELO_POR_DEFECTO, temperature=0.1,
                 system=SYSTEM_STRUCTURED):
    """
    Llama al LLM y OBLIGA la salida al formato pedido usando la funcion nativa
    de LangChain: `llm.with_structured_output(...)`.

    `esquema` puede ser un JSON Schema (dict), una clase Pydantic o un TypedDict.
    Con method="json_schema" (Structured Outputs) el esquema se manda por la API
    y el formato lo garantiza el decodificador del modelo: no hace falta
    describir la estructura en el prompt, ni parsear a mano, ni reintentar.

    Devuelve: dict.
    """
    llm = _get_llm(model, temperature)
    try:
        estructurado = llm.with_structured_output(esquema, method="json_schema", strict=True)
    except Exception:
        # Si el modelo/endpoint no soporta json_schema estricto, se usa el esquema
        # como definicion de tool (tampoco toca el prompt).
        estructurado = llm.with_structured_output(esquema, method="function_calling")

    salida = estructurado.invoke([
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ])
    return salida.model_dump() if hasattr(salida, "model_dump") else salida
