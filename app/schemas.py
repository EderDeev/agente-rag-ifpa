"""Contratos de dados (Pydantic) da API e da resposta do agente."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

StatusResposta = Literal["respondida", "sem_evidencia", "fora_do_escopo", "bloqueada", "erro"]


class Citacao(BaseModel):
    documento: str = Field(..., description="Nome do documento oficial do IFPA")
    pagina: int | None = Field(None, description="Página do PDF onde o trecho aparece")
    trecho: str = Field("", description="Trecho literal curto que fundamenta a resposta")

    @field_validator("trecho")
    @classmethod
    def encurtar(cls, v: str) -> str:
        v = " ".join(v.split())
        return v if len(v) <= 400 else v[:397] + "..."


class RespostaAgente(BaseModel):
    """Formato JSON que o agente deve produzir na resposta final."""

    resposta: str = Field(..., min_length=1, description="Resposta ao usuário em português")
    citacoes: list[Citacao] = Field(default_factory=list)
    status: StatusResposta = "respondida"
    confianca: Literal["alta", "media", "baixa"] = "media"


class PassoReAct(BaseModel):
    tipo: Literal["thought", "action", "observation"]
    conteudo: str
    ferramenta: str | None = None


class PerguntaIn(BaseModel):
    pergunta: str = Field(..., min_length=1)
    sessao_id: str = Field(..., min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class RespostaOut(RespostaAgente):
    sessao_id: str
    passos: list[PassoReAct] = Field(default_factory=list)
    avisos: list[str] = Field(default_factory=list)
