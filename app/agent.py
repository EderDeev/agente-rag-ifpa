"""Agente autônomo ReAct (Thought -> Action -> Observation) com LangChain/LangGraph.

- Modelo: ChatOpenAI (gpt-4o-mini por padrão, configurável)
- Ferramentas: buscar_normas_ifpa, listar_documentos_oficiais
- Memória de sessão: InMemorySaver (checkpointer do LangGraph), uma thread por sessao_id
- Saída: JSON reparado com json-repair e validado com Pydantic
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Callable

import openai
from json_repair import repair_json
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from pydantic import ValidationError

from app.config import exigir_chave_api, settings
from app.guardrails import citacao_foi_recuperada, validar_pergunta
from app.prompts import SYSTEM_PROMPT
from app.schemas import PassoReAct, RespostaAgente, RespostaOut
from app.tools import FERRAMENTAS

log = logging.getLogger("agente")

# FinOps: o histórico enviado ao modelo é limitado a este orçamento de tokens.
# A conversa inteira continua salva no checkpointer; só o que vai para a API é aparado.
ORCAMENTO_HISTORICO_TOKENS = 6000


@wrap_model_call
def aparar_historico(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Envia ao modelo só as mensagens mais recentes que cabem no orçamento de tokens."""
    mensagens = request.messages
    if count_tokens_approximately(mensagens) > ORCAMENTO_HISTORICO_TOKENS:
        aparadas = trim_messages(
            mensagens,
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=ORCAMENTO_HISTORICO_TOKENS,
            start_on="human",  # nunca começa com ToolMessage órfã
            allow_partial=False,
        )
        if aparadas:
            request = request.override(messages=aparadas)
    return handler(request)


@lru_cache(maxsize=1)
def construir_agente():
    exigir_chave_api()
    modelo = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_output_tokens,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )
    return create_agent(
        model=modelo,
        tools=FERRAMENTAS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[aparar_historico],
        checkpointer=InMemorySaver(),
    )


# ---------------------------------------------------------------------------
# Extração do raciocínio ReAct (para logs e para a interface)
# ---------------------------------------------------------------------------

def _mensagens_do_turno(mensagens: list[BaseMessage]) -> list[BaseMessage]:
    """Retorna as mensagens geradas após a última pergunta do usuário."""
    for i in range(len(mensagens) - 1, -1, -1):
        if isinstance(mensagens[i], HumanMessage):
            return mensagens[i + 1 :]
    return mensagens


def _texto(msg: BaseMessage) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(p.get("text", "") for p in c if isinstance(p, dict))


def extrair_passos(mensagens: list[BaseMessage]) -> tuple[list[PassoReAct], list[str]]:
    passos: list[PassoReAct] = []
    observacoes: list[str] = []
    for msg in mensagens:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            if _texto(msg).strip():
                passos.append(PassoReAct(tipo="thought", conteudo=_texto(msg).strip()))
            for call in msg.tool_calls:
                args = dict(call.get("args") or {})
                pensamento = args.pop("pensamento", None)
                if pensamento:
                    passos.append(PassoReAct(tipo="thought", conteudo=pensamento))
                args_txt = ", ".join(f'{k}="{v}"' for k, v in args.items())
                passos.append(
                    PassoReAct(tipo="action", ferramenta=call["name"], conteudo=f"{call['name']}({args_txt})")
                )
        elif isinstance(msg, ToolMessage):
            obs = _texto(msg)
            observacoes.append(obs)
            resumo = obs if len(obs) <= 1500 else obs[:1500] + " [...]"
            passos.append(PassoReAct(tipo="observation", ferramenta=msg.name, conteudo=resumo))
    return passos, observacoes


# ---------------------------------------------------------------------------
# Saída estruturada tolerante a falhas
# ---------------------------------------------------------------------------

def interpretar_resposta_final(texto: str) -> tuple[RespostaAgente, list[str]]:
    """Converte a mensagem final do modelo em RespostaAgente.

    1. json-repair corrige JSON malformado (aspas, vírgulas, ```json``` etc.)
    2. Pydantic valida tipos e valores permitidos
    3. Se tudo falhar, devolve o texto cru com confiança baixa (nunca quebra a API)
    """
    avisos: list[str] = []
    try:
        dados = repair_json(texto, return_objects=True)
        if not isinstance(dados, dict) or "resposta" not in dados:
            raise ValueError("JSON sem o campo 'resposta'")
        return RespostaAgente.model_validate(dados), avisos
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
        log.warning("Saída fora do schema, usando fallback: %s", exc)
        avisos.append("A resposta do modelo não veio no formato estruturado; exibindo texto bruto.")
        return RespostaAgente(resposta=texto.strip() or "(sem resposta)", confianca="baixa"), avisos


def verificar_citacoes(resp: RespostaAgente, observacoes: list[str]) -> list[str]:
    """Guardrail de saída: remove citações que não vieram da busca vetorial."""
    avisos: list[str] = []
    verificadas = [c for c in resp.citacoes if citacao_foi_recuperada(c.documento, c.pagina, observacoes)]
    removidas = len(resp.citacoes) - len(verificadas)
    if removidas:
        avisos.append(f"{removidas} citação(ões) descartada(s) por não corresponder(em) aos trechos recuperados.")
    resp.citacoes = verificadas

    if resp.status == "respondida":
        if not observacoes:
            resp.confianca = "baixa"
            avisos.append("Resposta gerada sem consulta à base documental.")
        elif not verificadas:
            resp.confianca = "baixa"
            avisos.append("Nenhuma citação verificável: confira a informação na Secretaria Acadêmica.")
    else:
        resp.citacoes = []
    return avisos


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def _erro(sessao_id: str, mensagem: str, status: str = "erro") -> RespostaOut:
    return RespostaOut(sessao_id=sessao_id, resposta=mensagem, status=status, confianca="baixa")


def perguntar(pergunta: str, sessao_id: str) -> RespostaOut:
    validacao = validar_pergunta(pergunta)
    if not validacao.ok:
        log.info("[%s] entrada bloqueada: %s", sessao_id, validacao.motivo)
        return _erro(sessao_id, validacao.motivo, status="bloqueada")

    config = {"configurable": {"thread_id": sessao_id}, "recursion_limit": settings.recursion_limit}

    try:
        agente = construir_agente()
        estado = agente.invoke({"messages": [HumanMessage(validacao.texto)]}, config=config)
    except RuntimeError as exc:  # chave ausente
        log.error("Configuração inválida: %s", exc)
        return _erro(sessao_id, "O assistente não está configurado (chave de API ausente).")
    except openai.AuthenticationError:
        log.error("Chave da OpenAI inválida ou revogada")
        return _erro(sessao_id, "Falha de autenticação com o provedor de IA. Avise os administradores.")
    except openai.RateLimitError:
        log.warning("Cota/limite da OpenAI atingido")
        return _erro(sessao_id, "O limite de uso da API foi atingido. Tente novamente em alguns instantes.")
    except (openai.APITimeoutError, openai.APIConnectionError):
        log.warning("Timeout/conexão com a OpenAI")
        return _erro(sessao_id, "O provedor de IA não respondeu a tempo. Tente novamente.")
    except GraphRecursionError:
        log.warning("Limite de passos ReAct atingido")
        return _erro(
            sessao_id,
            "Não consegui chegar a uma resposta segura dentro do limite de passos. Reformule a pergunta de forma mais específica.",
        )
    except Exception:
        log.exception("Erro inesperado no agente")
        return _erro(sessao_id, "Ocorreu um erro interno ao processar sua pergunta.")

    turno = _mensagens_do_turno(estado["messages"])
    passos, observacoes = extrair_passos(turno)
    final = next((m for m in reversed(turno) if isinstance(m, AIMessage) and not m.tool_calls), None)

    resp, avisos = interpretar_resposta_final(_texto(final) if final else "")
    avisos += verificar_citacoes(resp, observacoes)

    for p in passos:
        log.info("[%s] %-11s %s", sessao_id, p.tipo.upper(), p.conteudo.replace("\n", " ")[:200])

    return RespostaOut(sessao_id=sessao_id, passos=passos, avisos=avisos, **resp.model_dump())
