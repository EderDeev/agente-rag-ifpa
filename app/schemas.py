"""Contratos de dados (Pydantic) da API e da resposta do agente."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

StatusResposta = Literal["respondida", "sem_evidencia", "fora_do_escopo", "bloqueada", "erro"]


class Citacao(BaseModel):
    documento: str = Field(..., description="Nome do documento de onde veio o trecho")
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


class Custo(BaseModel):
    """Custo de API gerado por uma pergunta (todas as chamadas do ciclo ReAct)."""

    modelo: str
    chamadas_llm: int = 0
    tokens_entrada: int = 0
    tokens_entrada_cache: int = 0
    tokens_saida: int = 0
    tokens_embedding: int = 0
    custo_usd: float = 0.0
    custo_brl: float = 0.0
    cotacao_usd_brl: float


ID_PATTERN = r"^[A-Za-z0-9_-]+$"


class PerguntaIn(BaseModel):
    pergunta: str = Field(..., min_length=1)
    sessao_id: str = Field(..., min_length=8, max_length=64, pattern=ID_PATTERN)
    # Espaço de documentos do usuário (fica igual entre conversas no mesmo navegador).
    espaco_id: str | None = Field(None, min_length=8, max_length=64, pattern=ID_PATTERN)


class DocumentoOut(BaseModel):
    documento: str
    arquivo: str
    exemplo: bool = False
    trechos: int = 0


class DocumentosOut(BaseModel):
    espaco_id: str
    documentos: list[DocumentoOut] = Field(default_factory=list)
    mensagem: str = ""


class RespostaOut(RespostaAgente):
    sessao_id: str
    passos: list[PassoReAct] = Field(default_factory=list)
    avisos: list[str] = Field(default_factory=list)
    custo: Custo | None = None
