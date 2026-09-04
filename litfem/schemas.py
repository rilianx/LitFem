"""Esquema de una respuesta. Los bloques de N respuestas se generan en
`preguntas.esquema_puntajes(n)`; la evidencia, en `evidencia.esquema_evidencia()`.

Restringir el score a un Literal impide que el modelo devuelva "4.5" ni "Agree".
"""

from typing import Literal

from pydantic import BaseModel, Field

__all__ = ["Puntaje", "Answer"]

Puntaje = Literal["1", "2", "3", "4", "5", "N/A"]


class Answer(BaseModel):
    reasoning: str = Field(description="step-by-step logical deduction BEFORE giving a score")
    score: Puntaje = Field(description="1-5, or N/A if it cannot be reliably determined")
