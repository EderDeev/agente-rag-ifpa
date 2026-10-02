"""Configuração central do agente.

Todos os parâmetros vêm de variáveis de ambiente (arquivo .env localmente ou
painel "Variables" no Railway). Nenhuma chave de API fica no código-fonte.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _int(nome: str, padrao: int) -> int:
    try:
        return int(os.getenv(nome, padrao))
    except ValueError:
        return padrao


def _float(nome: str, padrao: float) -> float:
    try:
        return float(os.getenv(nome, padrao))
    except ValueError:
        return padrao


@dataclass(frozen=True)
class Settings:
    # --- Credenciais (somente via ambiente) ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # --- Modelos (FinOps: os mais econômicos da OpenAI com bom tool calling) ---
    # gpt-4o-mini: US$ 0,15 / 1M tokens de entrada e US$ 0,60 / 1M de saída.
    chat_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # text-embedding-3-small: US$ 0,02 / 1M tokens, 1536 dimensões.
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    temperature: float = _float("LLM_TEMPERATURE", 0.0)
    max_output_tokens: int = _int("LLM_MAX_TOKENS", 900)
    request_timeout: int = _int("LLM_TIMEOUT", 45)
    max_retries: int = _int("LLM_MAX_RETRIES", 2)

    # --- Fatiamento (justificativa FinOps no README / relatório) ---
    chunk_size: int = _int("CHUNK_SIZE", 1200)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 200)

    # --- Recuperação ---
    top_k: int = _int("RETRIEVER_TOP_K", 4)
    # Distância de cosseno máxima aceita (0 = idêntico, 2 = oposto).
    # Trechos acima desse limiar são descartados como "sem evidência".
    max_distance: float = _float("RETRIEVER_MAX_DISTANCE", 0.75)

    # --- Agente ---
    recursion_limit: int = _int("AGENT_RECURSION_LIMIT", 12)

    # --- Blindagem de entrada ---
    max_question_chars: int = _int("MAX_QUESTION_CHARS", 1000)
    rate_limit_per_minute: int = _int("RATE_LIMIT_PER_MINUTE", 15)

    # --- Caminhos ---
    pdf_dir: Path = Path(os.getenv("PDF_DIR", BASE_DIR / "data" / "pdfs"))
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
    collection_name: str = os.getenv("CHROMA_COLLECTION", "normas_ifpa")


settings = Settings()


def exigir_chave_api() -> None:
    """Falha cedo, com mensagem clara, se a chave não estiver configurada."""
    if not settings.openai_api_key or not settings.openai_api_key.startswith("sk-"):
        raise RuntimeError(
            "OPENAI_API_KEY não configurada. Localmente, copie .env.example para .env "
            "e preencha a chave; no Railway, cadastre-a em Variables."
        )
