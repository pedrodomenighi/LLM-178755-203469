"""
Ingestor de documentos — constrói a base de conhecimento (RAG).

Fluxo:
    1. Lê todos os PDFs de `docs/`.
    2. Extrai o texto página a página (preservando o número da página).
    3. Divide o texto em chunks com sobreposição.
    4. Gera embeddings locais (nomic-embed-text) e persiste no ChromaDB.

Cada chunk carrega metadados (fonte, titulo, categoria, pagina) que serão
usados pelas tools de busca e de metadados expostas via MCP.

Uso:
    python src/ingest.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Garante saída UTF-8 no console do Windows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from pypdf import PdfReader  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

import config  # noqa: E402
from retriever import get_vectorstore  # noqa: E402

console = Console()


def _carregar_pdf(caminho: Path) -> list[Document]:
    """Extrai texto de um PDF, um Document por página com metadados."""
    meta = config.doc_metadata(caminho.name)
    documentos: list[Document] = []
    try:
        reader = PdfReader(str(caminho))
    except Exception as exc:  # PDF corrompido / protegido
        console.print(f"  [red]✗ Falha ao abrir {caminho.name}: {exc}[/red]")
        return documentos

    for i, page in enumerate(reader.pages, start=1):
        texto = (page.extract_text() or "").strip()
        if not texto:
            continue
        documentos.append(
            Document(
                page_content=texto,
                metadata={**meta, "pagina": i},
            )
        )
    return documentos


def _dividir(documentos: list[Document]) -> list[Document]:
    """Divide os documentos de página em chunks menores para o RAG."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documentos)


def main() -> None:
    console.print(
        Panel.fit(
            "[bold]Ingestão da Base de Conhecimento[/bold]\n"
            "PDFs institucionais da UPF → embeddings locais → ChromaDB",
            border_style="cyan",
        )
    )

    # Usa o sufixo em minúsculas para não duplicar arquivos em filesystems
    # case-insensitive (Windows), onde *.pdf e *.PDF casariam os mesmos arquivos.
    pdfs = sorted(
        (p for p in config.DOCS_DIR.glob("*") if p.suffix.lower() == ".pdf"),
        key=lambda p: p.name,
    )
    if not pdfs:
        console.print(
            f"[red]Nenhum PDF encontrado em {config.DOCS_DIR}. "
            "Adicione os documentos e rode novamente.[/red]"
        )
        sys.exit(1)

    # Reconstrói a base do zero para tornar a ingestão idempotente.
    if config.CHROMA_DIR.exists():
        console.print("[dim]Removendo base vetorial anterior...[/dim]")
        shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)

    tabela = Table(title="Documentos processados", show_lines=False)
    tabela.add_column("Arquivo", style="cyan", no_wrap=False)
    tabela.add_column("Categoria", style="magenta")
    tabela.add_column("Páginas", justify="right")
    tabela.add_column("Chunks", justify="right", style="green")

    todos_chunks: list[Document] = []
    ignorados: list[str] = []

    for pdf in pdfs:
        console.print(f"[bold]→[/bold] Lendo [cyan]{pdf.name}[/cyan] ...")
        paginas = _carregar_pdf(pdf)
        if not paginas:
            ignorados.append(pdf.name)
            console.print(
                f"  [yellow]⚠ Nenhum texto extraído de {pdf.name} "
                "(provavelmente é um PDF digitalizado/imagem — exigiria OCR). "
                "Ignorando.[/yellow]"
            )
            continue
        chunks = _dividir(paginas)
        todos_chunks.extend(chunks)
        meta = config.doc_metadata(pdf.name)
        tabela.add_row(pdf.name, meta["categoria"], str(len(paginas)), str(len(chunks)))

    if not todos_chunks:
        console.print("[red]Nenhum texto extraível encontrado. Abortando.[/red]")
        sys.exit(1)

    console.print(tabela)
    console.print(
        f"\n[bold]Gerando embeddings de {len(todos_chunks)} chunks "
        f"com '{config.EMBED_MODEL}'...[/bold] [dim](pode levar alguns minutos)[/dim]"
    )

    vectordb = get_vectorstore()

    # Indexa em lotes para não sobrecarregar o servidor de embeddings.
    lote = 128
    with console.status("[cyan]Indexando no ChromaDB...[/cyan]") as status:
        for inicio in range(0, len(todos_chunks), lote):
            parte = todos_chunks[inicio : inicio + lote]
            vectordb.add_documents(parte)
            status.update(
                f"[cyan]Indexando no ChromaDB... "
                f"{min(inicio + lote, len(todos_chunks))}/{len(todos_chunks)}[/cyan]"
            )

    console.print(
        Panel.fit(
            f"[bold green]✓ Base construída com sucesso![/bold green]\n"
            f"Chunks indexados: [bold]{len(todos_chunks)}[/bold]\n"
            f"Coleção: [bold]{config.COLLECTION_NAME}[/bold]\n"
            f"Local: [dim]{config.CHROMA_DIR}[/dim]"
            + (
                f"\n[yellow]Ignorados (sem texto): {', '.join(ignorados)}[/yellow]"
                if ignorados
                else ""
            ),
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
