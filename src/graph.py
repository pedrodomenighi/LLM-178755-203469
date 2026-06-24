"""
Orquestração multiagente com LangGraph.

Modela o fluxo de colaboração como um grafo de estados:

        START → consultor → verificador → (condicional)
                    ↑__________________________│
                       (REPROVADA e dentro do limite de iterações)

O Verificador pode reprovar a resposta e devolver o controle ao Consultor para
um novo ciclo de recuperação (mais abrangente), até `MAX_ITER` vezes.

Também concentra a configuração do cliente MCP, que inicia o servidor
`mcp_server.py` por stdio e disponibiliza suas tools aos agentes.
"""

from __future__ import annotations

import sys

from langgraph.graph import END, START, StateGraph
from langchain_mcp_adapters.client import MultiServerMCPClient

import config
from agents import AgentState, make_consultor, make_verificador


def criar_mcp_client() -> MultiServerMCPClient:
    """Cria o cliente MCP que sobe o servidor da base documental via stdio."""
    return MultiServerMCPClient(
        {
            "upf": {
                "command": sys.executable,
                "args": [str(config.MCP_SERVER_PATH)],
                "transport": "stdio",
            }
        }
    )


def _decidir(state: AgentState) -> str:
    """Aresta condicional após o Verificador: reprocessar ou encerrar."""
    reprovada = state.get("veredito") == "REPROVADA"
    dentro_limite = state.get("iteracoes", 0) < config.MAX_ITER
    if reprovada and dentro_limite:
        return "consultor"
    return END


def build_graph(tools: dict, llm):
    """Monta e compila o grafo de estados com os dois agentes."""
    consultor = make_consultor(tools, llm)
    verificador = make_verificador(tools, llm)

    grafo = StateGraph(AgentState)
    grafo.add_node("consultor", consultor)
    grafo.add_node("verificador", verificador)

    grafo.add_edge(START, "consultor")
    grafo.add_edge("consultor", "verificador")
    grafo.add_conditional_edges(
        "verificador",
        _decidir,
        {"consultor": "consultor", END: END},
    )

    return grafo.compile()
