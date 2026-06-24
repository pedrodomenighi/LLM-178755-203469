"""
Servidor MCP (Model Context Protocol) — expõe a base de conhecimento da UPF
como ferramentas padronizadas, acessíveis pelos agentes.

Implementado com FastMCP (SDK oficial do MCP). O servidor roda como um processo
separado e se comunica por stdio; os agentes (no grafo LangGraph) carregam estas
tools via `langchain-mcp-adapters`.

Tools expostas:
    • busca_semantica   — Tool 1: consulta a base vetorial (RAG).
    • consulta_metadados — Tool 2: lista/descreve os documentos institucionais.
    • resumo_documento   — Tool 3: gera um resumo de um documento da base.

Execução direta (para teste manual do servidor):
    python src/mcp_server.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

import config  # noqa: E402
from retriever import get_llm, get_vectorstore, vectorstore_existe  # noqa: E402

# Silencia logs verbosos (cada requisição MCP / HTTP do Ollama) para não poluir
# o terminal do usuário. Erros e avisos continuam visíveis.
for _noisy in ("httpx", "mcp", "mcp.server", "mcp.server.lowlevel"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

mcp = FastMCP("upf-base-documental")

# A base vetorial e o LLM são carregados uma única vez (lazy) e reaproveitados
# entre chamadas, mantendo o processo do servidor "quente".
_vectordb = None
_llm = None


def _vs():
    """Acesso preguiçoso ao vetor store, com mensagem clara se não existir."""
    global _vectordb
    if not vectorstore_existe():
        raise RuntimeError(
            "Base vetorial não encontrada. Rode `python src/ingest.py` primeiro."
        )
    if _vectordb is None:
        _vectordb = get_vectorstore()
    return _vectordb


def _modelo():
    global _llm
    if _llm is None:
        _llm = get_llm(temperature=0.1)
    return _llm


# ─── Tool 1 — Busca Semântica ──────────────────────────────────────────────
@mcp.tool()
def busca_semantica(query: str, k: int = 5) -> str:
    """
    Consulta a base vetorial institucional e retorna os trechos mais
    relevantes para a pergunta, com seus metadados (documento, página, score).

    Args:
        query: pergunta ou termo de busca em linguagem natural.
        k: quantidade de trechos a recuperar (padrão 5).

    Returns:
        JSON (string) com uma lista de trechos: conteudo, fonte, titulo,
        categoria, pagina e score de distância.
    """
    try:
        resultados = _vs().similarity_search_with_score(query, k=k)
    except Exception as exc:
        return json.dumps({"erro": str(exc)}, ensure_ascii=False)

    payload = []
    for doc, score in resultados:
        meta = doc.metadata or {}
        payload.append(
            {
                "conteudo": doc.page_content.strip(),
                "fonte": meta.get("fonte", "desconhecido"),
                "titulo": meta.get("titulo", meta.get("fonte", "desconhecido")),
                "categoria": meta.get("categoria", "Documento Institucional"),
                "pagina": meta.get("pagina"),
                "score": round(float(score), 4),
            }
        )
    return json.dumps(payload, ensure_ascii=False)


# ─── Tool 2 — Consulta de Metadados ────────────────────────────────────────
@mcp.tool()
def consulta_metadados(fonte: str | None = None) -> str:
    """
    Lista os documentos institucionais indexados e seus metadados. Se `fonte`
    for informado, retorna apenas os metadados daquele documento.

    Args:
        fonte: nome do arquivo (ex.: "guia_academico.pdf"). Opcional.

    Returns:
        JSON (string) com: fonte, titulo, categoria, paginas (nº de páginas
        com texto) e chunks (nº de trechos indexados).
    """
    try:
        registros = _vs()._collection.get(include=["metadatas"])
    except Exception as exc:
        return json.dumps({"erro": str(exc)}, ensure_ascii=False)

    agregado: dict[str, dict] = {}
    for meta in registros.get("metadatas", []) or []:
        if not meta:
            continue
        f = meta.get("fonte", "desconhecido")
        if fonte and f != fonte:
            continue
        info = agregado.setdefault(
            f,
            {
                "fonte": f,
                "titulo": meta.get("titulo", f),
                "categoria": meta.get("categoria", "Documento Institucional"),
                "paginas": set(),
                "chunks": 0,
            },
        )
        info["chunks"] += 1
        if meta.get("pagina") is not None:
            info["paginas"].add(meta["pagina"])

    saida = [
        {
            "fonte": v["fonte"],
            "titulo": v["titulo"],
            "categoria": v["categoria"],
            "paginas": len(v["paginas"]),
            "chunks": v["chunks"],
        }
        for v in agregado.values()
    ]
    saida.sort(key=lambda d: d["titulo"])
    return json.dumps(saida, ensure_ascii=False)


# ─── Tool 3 — Resumo de Documento ──────────────────────────────────────────
@mcp.tool()
def resumo_documento(fonte: str, max_chars: int = 6000) -> str:
    """
    Gera um resumo objetivo de um documento institucional indexado, usando o
    modelo local. Útil para dar ao usuário/agente uma visão geral do conteúdo.

    Args:
        fonte: nome do arquivo a resumir (ex.: "DispositivosRegimentais.pdf").
        max_chars: tamanho máximo de texto-fonte considerado (padrão 6000).

    Returns:
        JSON (string) com: fonte, titulo e resumo.
    """
    try:
        registros = _vs()._collection.get(
            where={"fonte": fonte}, include=["documents", "metadatas"]
        )
    except Exception as exc:
        return json.dumps({"erro": str(exc)}, ensure_ascii=False)

    docs = registros.get("documents", []) or []
    metas = registros.get("metadatas", []) or []
    if not docs:
        return json.dumps(
            {"erro": f"Documento '{fonte}' não encontrado na base."},
            ensure_ascii=False,
        )

    # Ordena os trechos por página para um resumo coerente.
    pares = sorted(zip(docs, metas), key=lambda p: (p[1] or {}).get("pagina", 0))
    titulo = (pares[0][1] or {}).get("titulo", fonte)
    texto = "\n".join(d for d, _ in pares)[:max_chars]

    prompt = (
        "Você é um assistente que resume documentos institucionais de uma "
        "universidade. Resuma o conteúdo abaixo em português, de forma objetiva, "
        "em até 8 linhas, destacando temas, prazos e regras principais. "
        "Não invente informações que não estejam no texto.\n\n"
        f"DOCUMENTO: {titulo}\n\nCONTEÚDO:\n{texto}\n\nRESUMO:"
    )
    try:
        resposta = _modelo().invoke(prompt)
        resumo = resposta.content if hasattr(resposta, "content") else str(resposta)
    except Exception as exc:
        return json.dumps({"erro": f"Falha ao gerar resumo: {exc}"}, ensure_ascii=False)

    return json.dumps(
        {"fonte": fonte, "titulo": titulo, "resumo": resumo.strip()},
        ensure_ascii=False,
    )


if __name__ == "__main__":
    # Comunicação por stdio (padrão do MCP para servidores locais).
    mcp.run(transport="stdio")
