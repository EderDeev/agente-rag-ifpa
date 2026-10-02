"""Pipeline de indexação RAG.

Etapas:
1. Ingestão documental  -> PyPDFLoader lê cada PDF oficial de data/pdfs/
2. Fatiamento           -> RecursiveCharacterTextSplitter com separadores jurídicos
3. Vetorização          -> OpenAI text-embedding-3-small
4. Armazenamento        -> ChromaDB persistido localmente em chroma_db/

Uso:
    python -m app.ingest            # indexa (pula se já existir índice)
    python -m app.ingest --rebuild  # apaga e reconstrói o índice
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import shutil
from bisect import bisect_right
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import exigir_chave_api, settings

log = logging.getLogger("ingest")

# Separadores em ordem de prioridade. Normas do IFPA são organizadas em
# Títulos > Capítulos > Seções > Artigos > Parágrafos/Incisos. Tentar quebrar
# primeiro nessas fronteiras mantém cada artigo inteiro dentro de um chunk,
# o que melhora a precisão da citação ("Art. 45 da Resolução 944/2023").
SEPARADORES_NORMATIVOS = [
    "\nTÍTULO ",
    "\nCAPÍTULO ",
    "\nSeção ",
    "\nSEÇÃO ",
    "\nArt. ",
    "\n§ ",
    "\n\n",
    "\n",
    ". ",
    "; ",
    " ",
    "",
]

# Nome amigável exibido nas citações, inferido pelo nome do arquivo.
TITULOS_CONHECIDOS = {
    "944": "Resolução CONSUP nº 944/2023 – Regulamento Didático-Pedagógico da Graduação",
    "945": "Resolução CONSUP nº 945/2023 – RDP do Ensino da Educação Básica",
    "ppc": "PPC – Bacharelado em Ciência da Computação (IFPA Ananindeua)",
    "guia": "Guia Acadêmico IFPA 2025",
    "calendario": "Calendário Acadêmico IFPA 2026",
}


def titulo_do_documento(caminho: Path) -> str:
    nome = caminho.stem.lower()
    for chave, titulo in TITULOS_CONHECIDOS.items():
        if chave in nome:
            return titulo
    return caminho.stem.replace("_", " ").replace("-", " ").strip()


def limpar_texto(texto: str) -> str:
    """Normaliza artefatos comuns de extração de PDF sem perder a estrutura."""
    texto = texto.replace("\x00", "")
    texto = re.sub(r"-\n(?=[a-zà-ú])", "", texto)  # hifenização no fim de linha
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def carregar_pdfs(pasta: Path) -> list[Document]:
    """Retorna UM Document por PDF, com o texto de todas as páginas concatenado.

    Fatiar o documento inteiro (e não página a página) evita cortar um artigo
    na virada de página. A página de cada chunk é recuperada depois pelos
    offsets guardados em metadata["inicios_pagina"].
    """
    pdfs = sorted(pasta.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"Nenhum PDF encontrado em {pasta}. Coloque os documentos oficiais do IFPA nessa pasta."
        )

    documentos: list[Document] = []
    for pdf in pdfs:
        try:
            paginas = PyPDFLoader(str(pdf)).load()
        except Exception as exc:  # PDF corrompido não derruba a indexação inteira
            log.error("Falha ao ler %s: %s", pdf.name, exc)
            continue

        partes: list[str] = []
        inicios: list[tuple[int, int]] = []  # (offset no texto completo, página humana)
        offset = 0
        for d in paginas:
            texto = limpar_texto(d.page_content)
            if len(texto) < 30:  # páginas em branco / só cabeçalho
                continue
            # PyPDFLoader numera a partir de 0; guardamos a página "humana".
            inicios.append((offset, int(d.metadata.get("page", 0)) + 1))
            partes.append(texto)
            offset += len(texto) + 1  # +1 pela quebra de linha do join

        log.info("  %-55s %3d páginas com texto", pdf.name, len(partes))
        if not partes:
            log.warning("  %s parece ser um PDF escaneado (sem camada de texto).", pdf.name)
            continue
        documentos.append(
            Document(
                page_content="\n".join(partes),
                metadata={"fonte": titulo_do_documento(pdf), "arquivo": pdf.name, "inicios_pagina": inicios},
            )
        )
    return documentos


RE_ARTIGO = re.compile(r"^\s*Art\.\s*(\d+)", re.MULTILINE)


def _pagina_do_offset(inicios: list[tuple[int, int]], pos: int) -> int:
    i = bisect_right([o for o, _ in inicios], pos) - 1
    return inicios[max(i, 0)][1]


def fatiar(documentos: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=SEPARADORES_NORMATIVOS,
        keep_separator="start",
        length_function=len,
        add_start_index=True,
    )
    chunks: list[Document] = []
    for doc in documentos:
        texto = doc.page_content
        inicios = doc.metadata["inicios_pagina"]
        artigos = [(m.start(), int(m.group(1))) for m in RE_ARTIGO.finditer(texto)]
        posicoes = [p for p, _ in artigos]

        for c in splitter.create_documents([texto]):
            inicio = c.metadata["start_index"]
            meta = {
                "fonte": doc.metadata["fonte"],
                "arquivo": doc.metadata["arquivo"],
                "pagina": _pagina_do_offset(inicios, inicio),
                "start_index": inicio,
            }
            # Artigo em vigor no início do chunk: último "Art. N" antes (ou no) início.
            i = bisect_right(posicoes, inicio + 2) - 1
            conteudo = c.page_content.strip()
            if i >= 0:
                meta["artigo"] = artigos[i][1]
                if not conteudo.startswith("Art."):
                    # Contexto explícito evita que o modelo "chute" o número do artigo
                    # quando o chunk começa no meio dele (ex.: "§ 2º ...").
                    conteudo = f"(continuação do Art. {artigos[i][1]}) {conteudo}"
            # ID determinístico: reindexar o mesmo conteúdo não duplica vetores.
            meta_id = f"{meta['arquivo']}|{inicio}"
            chunks.append(Document(page_content=conteudo, metadata=meta, id=hashlib.sha1(meta_id.encode()).hexdigest()))
    return chunks


def obter_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        max_retries=settings.max_retries,
        request_timeout=settings.request_timeout,
    )


def abrir_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=obter_embeddings(),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def indice_existe() -> bool:
    try:
        return abrir_vectorstore()._collection.count() > 0
    except Exception:
        return False


def construir_indice(rebuild: bool = False) -> int:
    exigir_chave_api()

    if rebuild and settings.chroma_dir.exists():
        log.info("Removendo índice anterior em %s", settings.chroma_dir)
        shutil.rmtree(settings.chroma_dir)

    if not rebuild and indice_existe():
        n = abrir_vectorstore()._collection.count()
        log.info("Índice já existe com %d chunks. Use --rebuild para recriar.", n)
        return n

    log.info("1/4 Ingestão de PDFs em %s", settings.pdf_dir)
    documentos = carregar_pdfs(settings.pdf_dir)

    log.info("2/4 Fatiamento (chunk_size=%d, chunk_overlap=%d)", settings.chunk_size, settings.chunk_overlap)
    chunks = fatiar(documentos)
    total_chars = sum(len(c.page_content) for c in chunks)
    tokens_estimados = total_chars // 4
    custo = tokens_estimados / 1_000_000 * 0.02
    log.info("    %d chunks | ~%d tokens | custo estimado de embedding ≈ US$ %.4f", len(chunks), tokens_estimados, custo)

    log.info("3/4 Vetorização com %s", settings.embedding_model)
    log.info("4/4 Persistindo no ChromaDB em %s", settings.chroma_dir)
    vs = abrir_vectorstore()
    lote = 200  # lotes pequenos evitam estourar o limite de tokens por requisição
    for i in range(0, len(chunks), lote):
        parte = chunks[i : i + lote]
        vs.add_documents(parte, ids=[c.id for c in parte])
        log.info("    %d/%d chunks indexados", min(i + lote, len(chunks)), len(chunks))

    return vs._collection.count()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Indexa os PDFs do IFPA no ChromaDB.")
    parser.add_argument("--rebuild", action="store_true", help="apaga e reconstrói o índice")
    args = parser.parse_args()
    try:
        total = construir_indice(rebuild=args.rebuild)
    except Exception as exc:
        log.error("Erro na indexação: %s", exc)
        sys.exit(1)
    log.info("Pronto: %d chunks no índice '%s'.", total, settings.collection_name)


if __name__ == "__main__":
    main()
