"""Cálculo do custo de API de cada pergunta (FinOps).

Fontes dos números:
- Tokens do LLM: `usage_metadata` devolvido pela OpenAI em cada AIMessage do turno
  (inclui todas as chamadas do ciclo ReAct, não só a resposta final).
- Tokens de embedding: as consultas enviadas à ferramenta de busca, contadas com
  tiktoken (mesmo tokenizador do text-embedding-3-small).
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from langchain_core.messages import AIMessage, BaseMessage

from app.config import settings
from app.schemas import Custo

log = logging.getLogger("custos")

# Preços em US$ por 1 milhão de tokens: (entrada, entrada em cache, saída).
PRECOS_LLM = {
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5-mini": (0.25, 0.025, 2.00),
}
PRECOS_EMBEDDING = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}


def _preco_llm(modelo: str) -> tuple[float, float, float]:
    # Permite sobrescrever pelo ambiente caso a OpenAI mude a tabela.
    if os.getenv("PRICE_INPUT_PER_1M") and os.getenv("PRICE_OUTPUT_PER_1M"):
        entrada = float(os.environ["PRICE_INPUT_PER_1M"])
        return entrada, float(os.getenv("PRICE_CACHED_PER_1M", entrada)), float(os.environ["PRICE_OUTPUT_PER_1M"])
    # "gpt-4o-mini-2024-07-18" -> usa a entrada mais específica que for prefixo
    for nome in sorted(PRECOS_LLM, key=len, reverse=True):
        if modelo.startswith(nome):
            return PRECOS_LLM[nome]
    log.warning("Modelo %s sem preço cadastrado; usando o de gpt-4o-mini", modelo)
    return PRECOS_LLM["gpt-4o-mini"]


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def contar_tokens(texto: str) -> int:
    try:
        return len(_encoder().encode(texto))
    except Exception:  # sem tiktoken/cache offline: estimativa de ~4 caracteres por token
        return max(1, len(texto) // 4)


def calcular_custo(mensagens_do_turno: list[BaseMessage]) -> Custo:
    entrada = cache = saida = chamadas = 0
    tokens_embedding = 0

    for msg in mensagens_do_turno:
        if not isinstance(msg, AIMessage):
            continue
        uso = msg.usage_metadata or {}
        if uso:
            chamadas += 1
            entrada += uso.get("input_tokens", 0)
            saida += uso.get("output_tokens", 0)
            cache += (uso.get("input_token_details") or {}).get("cache_read", 0) or 0
        for call in msg.tool_calls or []:
            if call.get("name") == "buscar_normas_ifpa":
                tokens_embedding += contar_tokens(str((call.get("args") or {}).get("consulta", "")))

    p_in, p_cache, p_out = _preco_llm(settings.chat_model)
    p_emb = PRECOS_EMBEDDING.get(settings.embedding_model, 0.02)
    usd = ((entrada - cache) * p_in + cache * p_cache + saida * p_out + tokens_embedding * p_emb) / 1_000_000

    return Custo(
        modelo=settings.chat_model,
        chamadas_llm=chamadas,
        tokens_entrada=entrada,
        tokens_entrada_cache=cache,
        tokens_saida=saida,
        tokens_embedding=tokens_embedding,
        custo_usd=round(usd, 6),
        custo_brl=round(usd * settings.usd_brl, 6),
        cotacao_usd_brl=settings.usd_brl,
    )
