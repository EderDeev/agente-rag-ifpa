"""Ferramentas (Tool Calling) expostas ao agente ReAct.

A docstring de cada @tool é enviada ao modelo como contrato: ela descreve
QUANDO usar, O QUE passar e O QUE esperar de volta. Por isso é detalhada.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.tools import tool

from app.config import settings
from app.ingest import abrir_vectorstore

log = logging.getLogger("tools")

SEM_EVIDENCIA = "SEM_EVIDENCIA"

# Palavras muito frequentes que não ajudam a busca lexical.
STOPWORDS = set(
    "a o as os um uma de da do das dos em no na nos nas por para com sem que se ao aos e ou "
    "é ser sua seu suas seus pelo pela pelos pelas como qual quais quando onde posso pode "
    "caso sobre mais menos esse essa este esta isso isto".split()
)


@lru_cache(maxsize=1)
def _vectorstore() -> Chroma:
    return abrir_vectorstore()


def _tokenizar(texto: str) -> list[str]:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]+", texto) if t not in STOPWORDS and len(t) > 1]


@lru_cache(maxsize=1)
def _bm25() -> BM25Retriever:
    """Índice lexical BM25 montado em memória a partir dos chunks do ChromaDB."""
    dados = _vectorstore()._collection.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=t, metadata=m or {}, id=i)
        for t, m, i in zip(dados["documents"], dados["metadatas"], dados["ids"])
    ]
    retriever = BM25Retriever.from_documents(docs, preprocess_func=_tokenizar)
    retriever.k = settings.bm25_k
    return retriever


def busca_hibrida(consulta: str) -> list[tuple[Document, str]]:
    """Busca vetorial (semântica) + BM25 (lexical), sem duplicatas.

    A busca vetorial decide SE há evidência (limiar de distância); a lexical só
    complementa com trechos que contêm os termos exatos (ex.: "abono", "Art. 88"),
    que embeddings às vezes deixam escapar.
    """
    vetoriais = _vectorstore().similarity_search_with_score(consulta, k=settings.top_k)
    relevantes = [(d, f"semântica {round(1 - dist, 3)}") for d, dist in vetoriais if dist <= settings.max_distance]
    if not relevantes:
        return []

    vistos = {(d.metadata.get("arquivo"), d.metadata.get("start_index")) for d, _ in relevantes}
    try:
        for d in _bm25().invoke(consulta):
            chave = (d.metadata.get("arquivo"), d.metadata.get("start_index"))
            if chave not in vistos:
                vistos.add(chave)
                relevantes.append((d, "lexical"))
    except Exception:  # a busca lexical é complementar: falha nela não derruba a resposta
        log.exception("Falha na busca BM25")
    return relevantes


@lru_cache(maxsize=1)
def _documentos_indexados() -> list[str]:
    dados = _vectorstore()._collection.get(include=["metadatas"])
    return sorted({m.get("fonte", "?") for m in dados.get("metadatas") or []})


@tool
def buscar_normas_ifpa(consulta: str, pensamento: str) -> str:
    """Busca nos documentos OFICIAIS do IFPA indexados no ChromaDB (busca híbrida: semântica + palavras-chave).

    Use SEMPRE esta ferramenta antes de responder qualquer pergunta sobre
    regras acadêmicas do IFPA: médias e aprovação, frequência mínima, faltas e
    abono, reprovação, prova final, recuperação, trancamento, cancelamento de
    matrícula, aproveitamento de estudos, estágio, TCC, carga horária,
    disciplinas do curso de Ciência da Computação, datas do calendário
    acadêmico, direitos e deveres dos estudantes.

    Args:
        consulta: Pergunta reformulada em linguagem normativa, com os termos
            que provavelmente aparecem no documento. Ex.: "frequência mínima
            exigida para aprovação em componente curricular" em vez de
            "quantas faltas posso ter?". Para perguntas com várias partes,
            chame a ferramenta uma vez para cada parte.
        pensamento: Uma frase curta explicando POR QUE esta busca é
            necessária e o que você espera encontrar (fica registrado como
            "Thought" no log de raciocínio ReAct).

    Returns:
        Até 7 trechos numerados no formato
        "[n] <documento> | página <p> | Art. <a> | relevância <r>" seguido do
        texto. "(continuação do Art. N)" indica que o trecho pertence ao Art. N.
        Cite exatamente o documento, a página e o artigo informados. Se o
        retorno começar com "SEM_EVIDENCIA", os documentos NÃO tratam do
        assunto e você não deve inventar uma resposta.
    """
    consulta = (consulta or "").strip()
    if not consulta:
        return f"{SEM_EVIDENCIA}: consulta vazia."

    try:
        resultados = busca_hibrida(consulta)
    except Exception as exc:
        log.exception("Falha na busca vetorial")
        return f"ERRO_BUSCA: a base vetorial está indisponível no momento ({type(exc).__name__})."

    # Guardrail de recuperação: sem trecho semanticamente próximo, não há
    # evidência — o modelo não deve "esticar" um contexto irrelevante.
    if not resultados:
        return (
            f"{SEM_EVIDENCIA}: nenhum trecho dos documentos oficiais é suficientemente "
            f"relacionado a '{consulta}'. Tente reformular uma vez com outros termos ou "
            "informe ao usuário que não há base documental."
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
def listar_documentos_oficiais() -> str:
    """Lista os documentos oficiais do IFPA disponíveis na base de conhecimento.

    Use quando o usuário perguntar quais documentos/fontes o assistente
    conhece, ou para decidir se um assunto está coberto pela base antes de
    responder. Não recebe argumentos.

    Returns:
        Lista com um documento por linha.
    """
    try:
        docs = _documentos_indexados()
    except Exception as exc:
        log.exception("Falha ao listar documentos")
        return f"ERRO_BUSCA: não foi possível listar os documentos ({type(exc).__name__})."
    if not docs:
        return "Nenhum documento indexado."
    return "\n".join(f"- {d}" for d in docs)


FERRAMENTAS = [buscar_normas_ifpa, listar_documentos_oficiais]
