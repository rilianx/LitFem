"""Banco de preguntas: parseo, barajado ciego y esquema de salida dinamico.

En cada llamada se envian TODAS las preguntas juntas, pero:
  - sin decir a que dimension pertenece cada una,
  - con etiquetas neutras S1..Sn asignadas por el barajado,
  - en un orden distinto por semilla / unidad / iteracion.

Asi el modelo no puede agrupar mentalmente por dimension ni arrastrar el mismo
sesgo de posicion en todas las llamadas, y como cada fila guarda la `posicion`
en que se pregunto, el efecto de fatiga se puede medir despues
(`litfem.display.efecto_posicion`).
"""

import random
import re
from functools import lru_cache

from pydantic import Field, create_model

from .schemas import Answer

__all__ = ["parsear_preguntas", "barajar", "formatear_preguntas", "esquema_puntajes"]

_RE_ID = re.compile(r"^\s*([A-Za-z]+?)(\d+)\s*$")
# "P1: texto...", "Vr3: texto..." (formato antiguo, un bloque por dimension)
_RE_LINEA = re.compile(r"^\s*([A-Za-z]+?)(\d+)\s*:\s*(.+)$")


def _una(identificador, texto):
    m = _RE_ID.match(identificador)
    if not m:
        raise ValueError(f"Id de pregunta no reconocido: {identificador!r}. Se espera 'P1', 'Vr3'...")
    dim, num = m.group(1), int(m.group(2))
    return {"id": f"{dim}{num}", "dimension": dim, "pregunta": f"Q{num}", "texto": texto.strip()}


def parsear_preguntas(preguntas):
    """
    Normaliza el banco de preguntas a [{"id", "dimension", "pregunta", "texto"}, ...],
    conservando el orden. La DIMENSION sale del prefijo del id ("Vr3" -> "Vr").

    Acepta:
      - lista de pares  [("P1", "enunciado..."), ...]        <- forma recomendada
      - lista de dicts  [{"id": "P1", "texto": "..."}, ...]
      - dict por dimension {"P": "P1: ...\nP2: ..."}          <- formato antiguo
    """
    if isinstance(preguntas, dict):
        salida = []
        for dim, bloque in preguntas.items():
            for linea in bloque.splitlines():
                m = _RE_LINEA.match(linea)
                if m:
                    salida.append(_una(f"{dim}{int(m.group(2))}", m.group(3)))
    else:
        salida = []
        for item in preguntas:
            if isinstance(item, dict):
                salida.append(_una(item["id"], item.get("texto") or item["text"]))
            else:
                salida.append(_una(*item))

    if not salida:
        raise ValueError("Banco de preguntas vacio.")
    repetidos = {q["id"] for q in salida if [x["id"] for x in salida].count(q["id"]) > 1}
    if repetidos:
        raise ValueError(f"Ids repetidos en el banco de preguntas: {sorted(repetidos)}")
    return salida


def barajar(preguntas, semilla):
    """
    Devuelve [(etiqueta_neutra, pregunta), ...] en orden aleatorio reproducible.
    `semilla` puede ser cualquier objeto (se usa su repr), tipicamente una tupla
    (semilla_global, id_unidad, iteracion).
    """
    orden = list(preguntas)
    random.Random(repr(semilla)).shuffle(orden)
    return [(f"S{i}", q) for i, q in enumerate(orden, start=1)]


def formatear_preguntas(orden):
    """Bloque para el prompt: solo etiqueta neutra y enunciado, sin dimension."""
    return "\n".join(f"{etiqueta}: {q['texto']}" for etiqueta, q in orden)


@lru_cache(maxsize=None)
def esquema_puntajes(n):
    """
    Modelo Pydantic con los campos S1..Sn (todos obligatorios, cada uno un
    `Answer` con reasoning + score). Al ser obligatorios, el Structured Output
    garantiza que el modelo responda TODAS las preguntas.
    """
    campos = {f"S{i}": (Answer, Field(description=f"Answer to question S{i}")) for i in range(1, n + 1)}
    return create_model(f"Puntajes{n}", **campos)
