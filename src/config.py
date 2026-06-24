"""
Configuração central do Assistente Acadêmico Multiagente.

Reúne caminhos, nomes de modelos e parâmetros de RAG em um único lugar,
lendo valores do arquivo `.env` quando presente (com padrões sensatos).
Também define o mapeamento de metadados dos documentos institucionais.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Carrega variáveis do .env (se existir) sem sobrescrever o ambiente real.
load_dotenv()

# Caminhos do projeto
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DOCS_DIR: Path = ROOT_DIR / "docs"
CHROMA_DIR: Path = ROOT_DIR / "chroma_db"
MCP_SERVER_PATH: Path = Path(__file__).resolve().parent / "mcp_server.py"

#  Modelos locais (Ollama) 
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.1:8b")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")

#  Armazenamento vetorial 
COLLECTION_NAME: str = "upf_docs"

#  Parâmetros de RAG / chunking 
TOP_K: int = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

#  Fluxo multiagente 
# Número máximo de vezes que o Verificador pode pedir reprocessamento.
MAX_ITER: int = int(os.getenv("MAX_ITER", "2"))


#  Metadados dos documentos institucionais 
# Títulos e categorias amigáveis por arquivo. Arquivos não listados recebem
# uma categoria inferida do nome (ver `doc_metadata`).
DOC_TITLES: dict[str, str] = {
    "guia_academico.pdf": "Guia Acadêmico da UPF",
    "DispositivosRegimentais.pdf": "Dispositivos Regimentais e Legais (Graduação)",
    "Edital_Transferencia_2025_2_Medicina.pdf": "Edital de Transferência 2025/2 — Medicina",
    "EDITAL BOLSAS 2026_2.pdf": "Edital de Bolsas 2026/2",
    "EDITAL SELECAO 2026_2.pdf": "Edital de Seleção / Processo Seletivo 2026/2",
    "8e6698b0-bd40-4e9a-a0cd-5c0384d3e937.pdf": "Edital de Processo Seletivo (Inverno 2025)",
}


def infer_categoria(filename: str) -> str:
    """Infere a categoria de um documento a partir do nome do arquivo."""
    nome = filename.lower()
    if "edital" in nome or nome.startswith("8e6698b0"):
        return "Edital"
    if "regiment" in nome or "dispositiv" in nome or "resolu" in nome:
        return "Norma / Regulamento"
    if "guia" in nome:
        return "Guia Acadêmico"
    if "calendar" in nome:
        return "Calendário Acadêmico"
    if "estagio" in nome or "estágio" in nome:
        return "Normas de Estágio"
    return "Documento Institucional"


def doc_metadata(filename: str) -> dict[str, str]:
    """Retorna {titulo, categoria, fonte} para um arquivo da base documental."""
    titulo = DOC_TITLES.get(filename, filename.replace("_", " ").replace(".pdf", ""))
    return {
        "fonte": filename,
        "titulo": titulo,
        "categoria": infer_categoria(filename),
    }
