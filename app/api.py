"""API HTTP (FastAPI) + interface web do assistente.

Local:    uvicorn app.api:app --reload
Railway:  uvicorn app.api:app --host 0.0.0.0 --port $PORT  (ver Dockerfile)
"""

from __future__ import annotations

import io
import logging
import re
import shutil
import unicodedata
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import segno
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response

from app.agent import construir_agente, perguntar
from app.config import BASE_DIR, settings
from app.guardrails import LimitadorDeTaxa, limitador
from app.ingest import (
    EXTENSOES_ACEITAS,
    DocumentoInvalido,
    construir_indice,
    documentos_do_espaco,
    indexar_arquivo,
    indice_existe,
    remover_documentos,
    titulo_amigavel,
)
from app.schemas import DocumentoOut, DocumentosOut, PerguntaIn, RespostaOut
from app.tools import marcar_base_alterada

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


app = FastAPI(title="Converse com os seus documentos", version="2.0.0", lifespan=lifespan)
limitador_upload = LimitadorDeTaxa(settings.rate_limit_upload_por_minuto)


@app.get("/health")
def health():
    corpo = {"status": "ok" if ESTADO["pronto"] else "degradado", "modelo": settings.chat_model}
    if ESTADO["erro"]:
        corpo["erro"] = ESTADO["erro"]
    return JSONResponse(corpo, status_code=200 if ESTADO["pronto"] else 503)


def _cliente(request: Request) -> str:
    return request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()


@app.post("/api/chat", response_model=RespostaOut)
async def chat(dados: PerguntaIn, request: Request) -> RespostaOut:
    if not limitador.permitir(_cliente(request)):
        raise HTTPException(429, "Muitas perguntas em sequência. Aguarde um minuto.")
    if not ESTADO["pronto"]:
        raise HTTPException(503, "O assistente ainda está inicializando ou mal configurado.")
    # O agente é síncrono; rodar em threadpool evita travar o event loop.
    return await run_in_threadpool(perguntar, dados.pergunta, dados.sessao_id, dados.espaco_id)


# ---------------------------------------------------------------------------
# Documentos enviados pelo usuário
# ---------------------------------------------------------------------------

RE_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _validar_espaco(espaco_id: str) -> str:
    if not RE_ID.match(espaco_id or ""):
        raise HTTPException(422, "Identificador de espaço inválido.")
    return espaco_id


def _nome_seguro(nome: str) -> str:
    """Nome de arquivo sem caminho, acentos ou caracteres especiais (evita path traversal)."""
    nome = Path(nome or "").name
    nome = unicodedata.normalize("NFKD", nome)
    nome = "".join(c for c in nome if not unicodedata.combining(c))
    base, ext = Path(nome).stem, Path(nome).suffix.lower()
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_")[:80] or "documento"
    return base + ext


def _listar(espaco_id: str, mensagem: str = "") -> DocumentosOut:
    docs = documentos_do_espaco(espaco_id, incluir_exemplos=settings.incluir_exemplos)
    return DocumentosOut(espaco_id=espaco_id, documentos=[DocumentoOut(**d) for d in docs], mensagem=mensagem)


@app.get("/api/documentos", response_model=DocumentosOut)
async def listar_documentos(espaco_id: str = Query(...)) -> DocumentosOut:
    _validar_espaco(espaco_id)
    if not ESTADO["pronto"]:
        raise HTTPException(503, "O assistente ainda está inicializando ou mal configurado.")
    return await run_in_threadpool(_listar, espaco_id)


@app.post("/api/documentos", response_model=DocumentosOut)
async def enviar_documento(
    request: Request,
    espaco_id: str = Form(...),
    arquivo: UploadFile = File(...),
) -> DocumentosOut:
    _validar_espaco(espaco_id)
    if not ESTADO["pronto"]:
        raise HTTPException(503, "O assistente ainda está inicializando ou mal configurado.")
    if not limitador_upload.permitir(_cliente(request)):
        raise HTTPException(429, "Muitos envios em sequência. Aguarde um minuto.")

    nome = _nome_seguro(arquivo.filename or "")
    ext = Path(nome).suffix.lower()
    if ext not in EXTENSOES_ACEITAS:
        raise HTTPException(415, "Formato não suportado. Envie um arquivo PDF, TXT ou MD.")

    # Lê no máximo o limite + 1 byte: arquivo maior é recusado sem ocupar memória à toa.
    limite = settings.max_upload_mb * 1024 * 1024
    conteudo = await arquivo.read(limite + 1)
    if len(conteudo) > limite:
        raise HTTPException(413, f"O arquivo passa de {settings.max_upload_mb} MB.")
    if not conteudo:
        raise HTTPException(422, "O arquivo está vazio.")
    if ext == ".pdf" and not conteudo.startswith(b"%PDF"):
        raise HTTPException(415, "O arquivo não parece ser um PDF válido.")

    existentes = await run_in_threadpool(documentos_do_espaco, espaco_id, False)
    if nome not in {d["arquivo"] for d in existentes} and len(existentes) >= settings.max_documentos_por_espaco:
        raise HTTPException(409, f"Limite de {settings.max_documentos_por_espaco} documentos atingido. Remova um para enviar outro.")

    pasta = settings.upload_dir / espaco_id
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / nome
    destino.write_bytes(conteudo)
    titulo = titulo_amigavel(Path(Path(arquivo.filename or nome).name))

    try:
        trechos = await run_in_threadpool(indexar_arquivo, destino, espaco_id, titulo)
    except DocumentoInvalido as exc:
        destino.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        destino.unlink(missing_ok=True)
        log.exception("Falha ao indexar %s", nome)
        raise HTTPException(500, "Não foi possível processar o documento agora. Tente novamente.") from exc
    finally:
        marcar_base_alterada()

    return await run_in_threadpool(_listar, espaco_id, f"“{titulo}” foi lido e dividido em {trechos} trecho{'s' if trechos != 1 else ''}. Já pode perguntar!")


@app.delete("/api/documentos", response_model=DocumentosOut)
async def remover_documento(espaco_id: str = Query(...), arquivo: str | None = Query(None)) -> DocumentosOut:
    _validar_espaco(espaco_id)
    if not ESTADO["pronto"]:
        raise HTTPException(503, "O assistente ainda está inicializando ou mal configurado.")
    nome = _nome_seguro(arquivo) if arquivo else None
    try:
        await run_in_threadpool(remover_documentos, espaco_id, nome)
    except DocumentoInvalido as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        marcar_base_alterada()
    pasta = settings.upload_dir / espaco_id
    if nome:
        (pasta / nome).unlink(missing_ok=True)
    else:
        shutil.rmtree(pasta, ignore_errors=True)
    return await run_in_threadpool(_listar, espaco_id, "Documento removido." if nome else "Seus documentos foram removidos.")


def url_publica(request: Request) -> str:
    """URL que os alunos devem abrir (QR code). Prioriza a configurada no ambiente."""
    if settings.public_url:
        return settings.public_url.rstrip("/")
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
    return f"{proto}://{host}"


@lru_cache(maxsize=8)
def _qr_svg(url: str) -> bytes:
    buf = io.BytesIO()
    segno.make(url, error="m").save(buf, kind="svg", scale=8, border=2, dark="#111111", light="#ffffff", xmldecl=False)
    return buf.getvalue()


@app.get("/api/info")
def info(request: Request):
    return {
        "url": url_publica(request),
        "modelo": settings.chat_model,
        "cotacao_usd_brl": settings.usd_brl,
        "upload": {
            "extensoes": sorted(EXTENSOES_ACEITAS),
            "max_mb": settings.max_upload_mb,
            "max_paginas": settings.max_upload_paginas,
            "max_documentos": settings.max_documentos_por_espaco,
        },
    }


@app.get("/qr.svg")
def qr_code(request: Request):
    return Response(_qr_svg(url_publica(request)), media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=3600"})


ICONES = {
    "favicon.ico": "image/x-icon",
    "favicon.svg": "image/svg+xml",
    "apple-touch-icon.png": "image/png",
    "icon-512.png": "image/png",
}


@app.get("/{nome}", include_in_schema=False)
def icone(nome: str):
    if nome not in ICONES:
        raise HTTPException(404)
    return FileResponse(STATIC / nome, media_type=ICONES[nome], headers={"Cache-Control": "public, max-age=86400"})


@app.get("/")
def index():
    return FileResponse(Path(STATIC / "index.html"))
