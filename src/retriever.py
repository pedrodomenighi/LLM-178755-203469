"""
Acesso ao armazenamento vetorial (ChromaDB) e aos modelos locais do Ollama.

Centraliza a criação dos objetos de embeddings, do LLM e da coleção vetorial,
de modo que o ingestor, o servidor MCP e os agentes compartilhem exatamente a
mesma configuração de RAG.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permite execução tanto como módulo (`python -m`) quanto como script solto,
# garantindo que `import config` resolva quando o servidor MCP for iniciado.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_chroma import Chroma  # noqa: E402
from langchain_ollama import ChatOllama, OllamaEmbeddings  # noqa: E402

import config  # noqa: E402


def get_embeddings() -> OllamaEmbeddings:
    """Modelo de embeddings local (nomic-embed-text via Ollama)."""
    return OllamaEmbeddings(
        model=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


def get_llm(temperature: float = 0.1) -> ChatOllama:
    """Modelo de linguagem local (llama3.1:8b via Ollama)."""
    return ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
    )


def get_vectorstore() -> Chroma:
    """
    Abre (ou cria) a coleção vetorial persistida em disco.

    Usa a mesma coleção e função de embeddings em todo o sistema, garantindo
    que a indexação feita pelo ingestor seja compatível com a busca dos agentes.
    """
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def vectorstore_existe() -> bool:
    """Indica se a base vetorial já foi construída pelo ingestor."""
    return config.CHROMA_DIR.exists() and any(config.CHROMA_DIR.iterdir())
