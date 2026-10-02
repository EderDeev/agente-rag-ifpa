"""API HTTP (FastAPI) + interface web do assistente.

Local:    uvicorn app.api:app --reload
Railway:  uvicorn app.api:app --host 0.0.0.0 --port $PORT  (ver Dockerfile)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from app.agent import construir_agente, perguntar
from app.config import BASE_DIR, settings
from app.guardrails import limitador
from app.ingest import construir_indice, indice_existe
from app.schemas import PerguntaIn, RespostaOut

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
log = logging.getLogger("api")

STATIC = BASE_DIR / "static"
ESTADO = {"pronto": False, "erro": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Indexa os PDFs na primeira subida (ex.: novo deploy no Railway, cujo disco é efêmero).
    try:
        if not indice_existe():
            log.info("Índice vetorial não encontrado — construindo a partir de %s", settings.pdf_dir)
            await run_in_threadpool(construir_indice)
        construir_agente()
        ESTADO["pronto"] = True
        log.info("Agente pronto (modelo=%s, embeddings=%s)", settings.chat_model, settings.embedding_model)
    except Exception as exc:  # a API sobe mesmo assim, para o /health mostrar o problema
        ESTADO["erro"] = str(exc)
        log.exception("Falha ao inicializar o agente")
    yield


app = FastAPI(title="Assistente Acadêmico IFPA", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    corpo = {"status": "ok" if ESTADO["pronto"] else "degradado", "modelo": settings.chat_model}
    if ESTADO["erro"]:
        corpo["erro"] = ESTADO["erro"]
    return JSONResponse(corpo, status_code=200 if ESTADO["pronto"] else 503)


@app.post("/api/chat", response_model=RespostaOut)
async def chat(dados: PerguntaIn, request: Request) -> RespostaOut:
    cliente = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
    if not limitador.permitir(cliente.strip()):
        raise HTTPException(429, "Muitas perguntas em sequência. Aguarde um minuto.")
    if not ESTADO["pronto"]:
        raise HTTPException(503, "O assistente ainda está inicializando ou mal configurado.")
    # O agente é síncrono; rodar em threadpool evita travar o event loop.
    return await run_in_threadpool(perguntar, dados.pergunta, dados.sessao_id)


@app.get("/")
def index():
    return FileResponse(Path(STATIC / "index.html"))
