"""Cliente LLM con salida estructurada (LangChain + OpenAI)."""

import os
import threading

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

__all__ = ["configurar_api_key", "llm_function", "SYSTEM_STRUCTURED", "MODELO_POR_DEFECTO"]

MODELO_POR_DEFECTO = "gpt-5.4-mini"

# El esquema viaja por la API (Structured Outputs), NO dentro del prompt: el
# system message solo pide completar todos los campos. Es neutro para servir
# tanto al juez como al extractor de evidencia.
SYSTEM_STRUCTURED = (
    "You are a careful literary analysis assistant. Fill in every field of the "
    "required output schema; never leave a field empty."
)

_LLM_CACHE = {}
_LOCK = threading.Lock()


def configurar_api_key(var="OPENAI_API_KEY", interactivo=True):
    """
    Carga la API key desde (en orden): variable de entorno ya definida,
    Colab Secrets, archivo .env y, como ultimo recurso, getpass.
    Nunca escribe la clave en el notebook. Devuelve True si quedo cargada.
    """
    if os.getenv(var):
        return True

    try:  # 1) Colab Secrets (icono de llave)
        from google.colab import userdata  # type: ignore
        valor = userdata.get(var)
        if valor:
            os.environ[var] = valor
            return True
    except Exception:
        pass

    try:  # 2) archivo .env
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except ImportError:
        pass

    if not os.getenv(var) and interactivo:  # 3) fallback interactivo
        from getpass import getpass
        os.environ[var] = getpass(f"{var}: ")

    return bool(os.getenv(var))


def _get_llm(model, temperature):
    """Reutiliza el mismo cliente por (modelo, temperatura); seguro entre hilos."""
    clave = (model, temperature)
    with _LOCK:
        if clave not in _LLM_CACHE:
            _LLM_CACHE[clave] = ChatOpenAI(model=model, temperature=temperature)
        return _LLM_CACHE[clave]


def llm_function(prompt, esquema, model=MODELO_POR_DEFECTO, temperature=0.1,
                 system=SYSTEM_STRUCTURED):
    """
    Llama al LLM y OBLIGA la salida al esquema (clase Pydantic, TypedDict o
    JSON Schema dict) con `with_structured_output(method="json_schema", strict=True)`.
    Si el modelo/endpoint rechaza json_schema estricto, reintenta una vez con
    method="function_calling" (el esquema viaja como definicion de tool).

    Devuelve: dict.
    """
    llm = _get_llm(model, temperature)
    mensajes = [SystemMessage(content=system), HumanMessage(content=prompt)]
    try:
        salida = llm.with_structured_output(esquema, method="json_schema", strict=True).invoke(mensajes)
    except Exception as e_json:
        try:
            salida = llm.with_structured_output(esquema, method="function_calling").invoke(mensajes)
        except Exception:
            raise e_json
    return salida.model_dump() if hasattr(salida, "model_dump") else salida
