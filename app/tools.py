"""Ferramentas (Tool Calling) expostas ao agente ReAct.

A docstring de cada @tool é enviada ao modelo como contrato: ela descreve
QUANDO usar, O QUE passar e O QUE esperar de volta. Por isso é detalhada.

O espaço do usuário (quais documentos ele enviou) NÃO é um argumento que o
modelo preenche: ele chega pelo `config` da execução, via ToolRuntime. Assim o
modelo não consegue, nem por engano nem por prompt injection, consultar os
documentos de outra pessoa.
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from functools import lru_cache

from langchain.tools import ToolRuntime, tool
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.config import settings
from app.ingest import ESPACO_EXEMPLOS, abrir_vectorstore, documentos_do_espaco

log = logging.getLogger("tools")

SEM_EVIDENCIA = "SEM_EVIDENCIA"

# Palavras muito frequentes que não ajudam a busca lexical.
STOPWORDS = set(
    "a o as os um uma de da do das dos em no na nos nas por para com sem que se ao aos e ou "
    "é ser sua seu suas seus pelo pela pelos pelas como qual quais quando onde posso pode "
    "caso sobre mais menos esse essa este esta isso isto".split()
)

# Muda a cada upload/remoção e invalida o cache dos índices BM25.
_versao = {"n": 0}
_trava = threading.Lock()


def marcar_base_alterada() -> None:
    with _trava:
        _versao["n"] += 1


@lru_cache(maxsize=1)
def _vectorstore() -> Chroma:
    return abrir_vectorstore()


def _espacos_visiveis(espaco: str) -> list[str]:
    espacos = [espaco] if espaco else []
    if settings.incluir_exemplos or not espacos:
        espacos.append(ESPACO_EXEMPLOS)
    return espacos


def _filtro(espaco: str) -> dict:
    espacos = _espacos_visiveis(espaco)
    return {"espaco": espacos[0]} if len(espacos) == 1 else {"espaco": {"$in": espacos}}


def _tokenizar(texto: str) -> list[str]:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]+", texto) if t not in STOPWORDS and len(t) > 1]


@lru_cache(maxsize=64)
def _bm25(espaco: str, _versao_base: int) -> BM25Retriever | None:
    """Índice lexical BM25 em memória com os trechos visíveis para um espaço."""
    dados = _vectorstore()._collection.get(where=_filtro(espaco), include=["documents", "metadatas"])
    docs = [
        Document(page_content=t, metadata=m or {}, id=i)
        for t, m, i in zip(dados["documents"], dados["metadatas"], dados["ids"])
    ]
    if not docs:
        return None
    retriever = BM25Retriever.from_documents(docs, preprocess_func=_tokenizar)
    retriever.k = settings.bm25_k
    return retriever


def busca_hibrida(consulta: str, espaco: str) -> list[tuple[Document, str]]:
    """Busca vetorial (semântica) + BM25 (lexical), sem duplicatas.

    A busca vetorial decide SE há evidência (limiar de distância); a lexical só
    complementa com trechos que contêm os termos exatos (nomes, siglas, números),
    que embeddings às vezes deixam escapar.
    """
    vetoriais = _vectorstore().similarity_search_with_score(consulta, k=settings.top_k, filter=_filtro(espaco))
    relevantes = [(d, f"semântica {round(1 - dist, 3)}") for d, dist in vetoriais if dist <= settings.max_distance]
    if not relevantes:
        return []

    vistos = {(d.metadata.get("espaco"), d.metadata.get("arquivo"), d.metadata.get("start_index")) for d, _ in relevantes}
    try:
        lexical = _bm25(espaco, _versao["n"])
        for d in lexical.invoke(consulta) if lexical else []:
            chave = (d.metadata.get("espaco"), d.metadata.get("arquivo"), d.metadata.get("start_index"))
            if chave not in vistos:
                vistos.add(chave)
                relevantes.append((d, "lexical"))
    except Exception:  # a busca lexical é complementar: falha nela não derruba a resposta
        log.exception("Falha na busca BM25")
    return relevantes


def _espaco_da_execucao(runtime: ToolRuntime | None) -> str:
    try:
        return str((runtime.config.get("configurable") or {}).get("espaco_id") or "")
    except Exception:
        return ""


@tool
def buscar_documentos(consulta: str, pensamento: str, runtime: ToolRuntime) -> str:
    """Busca trechos nos documentos do usuário (os que ele enviou e os de exemplo), com busca híbrida: semântica + palavras-chave.

    Use SEMPRE esta ferramenta antes de responder qualquer pergunta sobre o
    conteúdo dos documentos — qualquer tema: definições, resumos, datas,
    valores, regras, procedimentos, nomes, conclusões etc.

    Args:
        consulta: A pergunta reescrita com as palavras que provavelmente
            aparecem no documento (termos técnicos, nomes, sinônimos). Ex.:
            "definição de rede neural artificial" em vez de "o que é isso de
            rede neural?". Para perguntas com várias partes, chame a
            ferramenta uma vez para cada parte.
        pensamento: Uma frase curta explicando POR QUE esta busca é
            necessária e o que você espera encontrar (fica registrado como
            "Thought" no log de raciocínio ReAct).

    Returns:
        Até 7 trechos numerados no formato
        "[n] <documento> | página <p> | relevância <r>" seguido do texto (em
        normas jurídicas pode vir também "| Art. <a>"). Cite exatamente o
        documento e a página informados. Se o retorno começar com
        "SEM_EVIDENCIA", os documentos NÃO tratam do assunto e você não deve
        inventar uma resposta.
    """
    consulta = (consulta or "").strip()
    if not consulta:
        return f"{SEM_EVIDENCIA}: consulta vazia."
    espaco = _espaco_da_execucao(runtime)

    try:
        resultados = busca_hibrida(consulta, espaco)
    except Exception as exc:
        log.exception("Falha na busca vetorial")
        return f"ERRO_BUSCA: a base vetorial está indisponível no momento ({type(exc).__name__})."

    # Guardrail de recuperação: sem trecho semanticamente próximo, não há
    # evidência — o modelo não deve "esticar" um contexto irrelevante.
    if not resultados:
        return (
            f"{SEM_EVIDENCIA}: nenhum trecho dos documentos disponíveis é suficientemente "
            f"relacionado a '{consulta}'. Tente reformular uma vez com outros termos ou "
            "informe ao usuário que os documentos não tratam desse assunto."
        )

    blocos = []
    for i, (doc, relevancia) in enumerate(resultados, start=1):
        m = doc.metadata
        artigo = f" | Art. {m['artigo']}" if m.get("artigo") else ""
        blocos.append(
            f"[{i}] {m.get('fonte', '?')} | página {m.get('pagina', '?')}{artigo} | relevância {relevancia}\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(blocos)


@tool
def listar_documentos(runtime: ToolRuntime) -> str:
    """Lista os documentos disponíveis para este usuário: os que ele enviou e os de exemplo.

    Use quando o usuário perguntar quais documentos/fontes você conhece, quando
    ele se referir a "o documento que eu enviei" sem dizer qual, ou para
    decidir se um assunto está coberto pela base antes de responder. Não
    recebe argumentos.

    Returns:
        Um documento por linha, indicando se foi enviado pelo usuário ou se é de exemplo.
    """
    espaco = _espaco_da_execucao(runtime)
    try:
        docs = documentos_do_espaco(espaco, incluir_exemplos=settings.incluir_exemplos) if espaco else documentos_do_espaco(
            ESPACO_EXEMPLOS, incluir_exemplos=False
        )
    except Exception as exc:
        log.exception("Falha ao listar documentos")
        return f"ERRO_BUSCA: não foi possível listar os documentos ({type(exc).__name__})."
    if not docs:
        return "Nenhum documento disponível. Peça ao usuário para enviar um arquivo PDF, TXT ou MD."
    return "\n".join(f"- {d['documento']} ({'exemplo' if d['exemplo'] else 'enviado pelo usuário'})" for d in docs)


FERRAMENTAS = [buscar_documentos, listar_documentos]
