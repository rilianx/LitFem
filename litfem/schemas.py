"""Esquema de salida PTM-V declarado con Pydantic.

Ventaja sobre un string de ejemplo: los tipos quedan validados tambien en
Python y el score se restringe a valores concretos (1-5 o N/A), de modo que el
modelo NO puede devolver "4.5" ni "Agree".
"""

from typing import Literal

from pydantic import BaseModel, Field

__all__ = ["Puntaje", "Answer", "Moment", "Scores", "PTMVResult"]

Puntaje = Literal["1", "2", "3", "4", "5", "N/A"]


class Answer(BaseModel):
    reasoning: str = Field(description="step-by-step logical deduction BEFORE giving a score")
    score: Puntaje = Field(description="1-5, or N/A if it cannot be reliably determined")


class Moment(BaseModel):
    Q1: Answer
    Q2: Answer
    Q3: Answer
    Q4: Answer
    Q5: Answer


class Scores(BaseModel):
    Moment_A: Moment
    Moment_B: Moment
    Moment_C: Moment


class PTMVResult(BaseModel):
    character: str = Field(description="Name of the character")
    dimension_analyzed: str = Field(description="Name of the dimension evaluated")
    book: str = Field(description="Title/Fragment")
    analysis: str = Field(description="deep analysis across moments A, B and C")
    scores: Scores
    conclusion: str = Field(description="key scenes or textual elements supporting the scores")
