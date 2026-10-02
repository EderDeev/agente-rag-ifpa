"""Regras de blindagem do agente.

Camadas de defesa (da entrada até a saída):

1. Entrada       -> tamanho, caracteres de controle, tentativas de prompt injection
2. Tráfego       -> limite de requisições por cliente (protege a cota da API)
3. System prompt -> escopo restrito, proibição de alucinar, citação obrigatória (agent.py)
4. Recuperação   -> limiar de similaridade, "SEM_EVIDENCIA" (tools.py)
5. Saída         -> JSON validado com Pydantic + json-repair e checagem de
                    que toda citação veio de um trecho realmente recuperado
"""

from __future__ import annotations

import re
import threading
import time
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass

from app.config import settings

# Padrões típicos de prompt injection / jailbreak (pt-BR e inglês).
PADROES_INJECAO = [
    r"ignore (as|todas as|suas)? ?(instru[cç][oõ]es|regras)",
    r"esque[cç]a (as|todas as|suas)? ?(instru[cç][oõ]es|regras)",
    r"desconsidere (as|todas as|suas)? ?(instru[cç][oõ]es|regras)",
    r"ignore (all |any |the )?(previous|prior|above) (instructions|rules|prompts?)",
    r"disregard (all |the )?(previous|prior|above)",
    r"(mostre|revele|exiba|imprima|repita).{0,30}(prompt|instru[cç][oõ]es) (do )?sistema",
    r"(show|reveal|print|repeat).{0,30}system prompt",
    r"system prompt",
    r"voc[eê] agora [eé]",
    r"you are now",
    r"modo (desenvolvedor|dev|deus|irrestrito)",
    r"(developer|dan|god) mode",
    r"jailbreak",
    r"finja (que|ser)",
    r"pretend (to be|you are)",
    r"aja como (se|um|uma)",
    r"<\s*/?\s*(system|assistant|instructions?)\s*>",
    r"\[\s*(system|inst)\s*\]",
]
_REGEX_INJECAO = re.compile("|".join(PADROES_INJECAO), re.IGNORECASE)


@dataclass
class ResultadoValidacao:
    ok: bool
    texto: str
    motivo: str = ""


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKC", texto)
    # remove caracteres de controle e de largura zero (usados para esconder instruções)
    texto = "".join(c for c in texto if unicodedata.category(c)[0] != "C" or c in "\n\t")
    return re.sub(r"\s+", " ", texto).strip()


def validar_pergunta(pergunta: str) -> ResultadoValidacao:
    texto = _normalizar(pergunta or "")
    if not texto:
        return ResultadoValidacao(False, texto, "A pergunta está vazia.")
    if len(texto) > settings.max_question_chars:
        return ResultadoValidacao(
            False,
            texto,
            f"A pergunta excede {settings.max_question_chars} caracteres. Seja mais objetivo(a).",
        )
    if _REGEX_INJECAO.search(texto):
        return ResultadoValidacao(
            False,
            texto,
            "Sua mensagem contém instruções para alterar o comportamento do assistente. "
            "Sou um assistente acadêmico do IFPA e só respondo dúvidas sobre normas, "
            "curso e calendário com base nos documentos oficiais.",
        )
    return ResultadoValidacao(True, texto)


class LimitadorDeTaxa:
    """Janela deslizante de 60 s por cliente (IP). Protege a cota da OpenAI."""

    def __init__(self, limite_por_minuto: int):
        self.limite = limite_por_minuto
        self._hist: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def permitir(self, cliente: str) -> bool:
        agora = time.monotonic()
        with self._lock:
            fila = self._hist[cliente]
            while fila and agora - fila[0] > 60:
                fila.popleft()
            if len(fila) >= self.limite:
                return False
            fila.append(agora)
            return True


limitador = LimitadorDeTaxa(settings.rate_limit_per_minute)


def _chave(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", texto)


def citacao_foi_recuperada(documento: str, pagina: int | None, observacoes: list[str]) -> bool:
    """Confere se (documento, página) apareceu em algum retorno da ferramenta.

    Impede que o modelo "invente" uma fonte que não foi de fato consultada.
    """
    doc = _chave(documento)
    if not doc:
        return False
    for obs in observacoes:
        for cab in re.findall(r"^\[\d+\] (.+?) \| página (\d+)", obs, flags=re.MULTILINE):
            nome, pag = cab
            nome_k = _chave(nome)
            mesmo_doc = doc in nome_k or nome_k in doc
            if mesmo_doc and (pagina is None or int(pag) == int(pagina)):
                return True
    return False
