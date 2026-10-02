"""Ferramentas (Tool Calling) expostas ao agente ReAct.

A docstring de cada @tool é enviada ao modelo como contrato: ela descreve
QUANDO usar, O QUE passar e O QUE esperar de volta. Por isso é detalhada.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.tools import tool

from app.config import settings
from app.ingest import abrir_vectorstore

log = logging.getLogger("tools")

SEM_EVIDENCIA = "SEM_EVIDENCIA"


@lru_cache(maxsize=1)
def _vectorstore() -> Chroma:
    return abrir_vectorstore()


@lru_cache(maxsize=1)
def _documentos_indexados() -> list[str]:
    dados = _vectorstore()._collection.get(include=["metadatas"])
    return sorted({m.get("fonte", "?") for m in dados.get("metadatas") or []})


@tool
def buscar_normas_ifpa(consulta: str, pensamento: str) -> str:
    """Busca semântica nos documentos OFICIAIS do IFPA indexados no ChromaDB.

    Use SEMPRE esta ferramenta antes de responder qualquer pergunta sobre
    regras acadêmicas do IFPA: médias e aprovação, frequência mínima,
    reprovação, exame final, recuperação, trancamento, cancelamento de
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
        Até 4 trechos numerados no formato
        "[n] <documento> | página <p> | relevância <r>" seguido do texto.
        Cite exatamente o documento e a página informados. Se o retorno
        começar com "SEM_EVIDENCIA", os documentos NÃO tratam do assunto e
        você não deve inventar uma resposta.
    """
    consulta = (consulta or "").strip()
    if not consulta:
        return f"{SEM_EVIDENCIA}: consulta vazia."

    try:
        resultados = _vectorstore().similarity_search_with_score(consulta, k=settings.top_k)
    except Exception as exc:
        log.exception("Falha na busca vetorial")
        return f"ERRO_BUSCA: a base vetorial está indisponível no momento ({type(exc).__name__})."

    # Guardrail de recuperação: descarta trechos pouco similares para que o
    # modelo não seja tentado a "esticar" um contexto irrelevante.
    relevantes = [(d, dist) for d, dist in resultados if dist <= settings.max_distance]
    if not relevantes:
        return (
            f"{SEM_EVIDENCIA}: nenhum trecho dos documentos oficiais é suficientemente "
            f"relacionado a '{consulta}'. Tente reformular uma vez com outros termos ou "
            "informe ao usuário que não há base documental."
        )

    blocos = []
    for i, (doc, dist) in enumerate(relevantes, start=1):
        m = doc.metadata
        relevancia = round(1 - dist, 3)
        blocos.append(
            f"[{i}] {m.get('fonte', '?')} | página {m.get('pagina', '?')} | relevância {relevancia}\n"
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
