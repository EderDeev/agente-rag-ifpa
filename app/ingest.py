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
    pdfs = sorted(pasta.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"Nenhum PDF encontrado em {pasta}. Coloque os documentos oficiais do IFPA nessa pasta."
        )

    paginas: list[Document] = []
    for pdf in pdfs:
        try:
            docs = PyPDFLoader(str(pdf)).load()
        except Exception as exc:  # PDF corrompido não derruba a indexação inteira
            log.error("Falha ao ler %s: %s", pdf.name, exc)
            continue

        titulo = titulo_do_documento(pdf)
        uteis = 0
        for d in docs:
            d.page_content = limpar_texto(d.page_content)
            if len(d.page_content) < 30:  # páginas em branco / só cabeçalho
                continue
            d.metadata = {
                "fonte": titulo,
                "arquivo": pdf.name,
                # PyPDFLoader numera a partir de 0; exibimos a página "humana".
                "pagina": int(d.metadata.get("page", 0)) + 1,
            }
            paginas.append(d)
            uteis += 1
        log.info("  %-55s %3d páginas com texto", pdf.name, uteis)
        if uteis == 0:
            log.warning("  %s parece ser um PDF escaneado (sem camada de texto).", pdf.name)
    return paginas


def fatiar(paginas: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=SEPARADORES_NORMATIVOS,
        keep_separator="start",
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(paginas)
    for c in chunks:
        # ID determinístico: reindexar o mesmo conteúdo não duplica vetores.
        base = f"{c.metadata['arquivo']}|{c.metadata['pagina']}|{c.metadata.get('start_index')}"
        c.id = hashlib.sha1(base.encode()).hexdigest()
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
    paginas = carregar_pdfs(settings.pdf_dir)

    log.info("2/4 Fatiamento (chunk_size=%d, chunk_overlap=%d)", settings.chunk_size, settings.chunk_overlap)
    chunks = fatiar(paginas)
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
