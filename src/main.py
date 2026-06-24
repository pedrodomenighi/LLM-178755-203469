"""
Interface de terminal (CLI) do Assistente Acadêmico Multiagente da UPF.

Inicia o cliente MCP (que sobe o servidor da base documental), carrega as tools,
constrói o grafo LangGraph e abre um laço interativo onde o estudante faz
perguntas e acompanha a atuação de cada agente.

Comandos:
    /docs              lista os documentos institucionais indexados
    /resumo <arquivo>  gera um resumo de um documento (ex.: /resumo guia_academico.pdf)
    /ajuda             mostra a ajuda
    /sair              encerra

Uso:
    python src/main.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Garante saída UTF-8 no console do Windows (Rich usa setas, emojis e caixas
# que não existem no code page legado cp1252).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from langchain_mcp_adapters.tools import load_mcp_tools  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.prompt import Prompt  # noqa: E402
from rich.table import Table  # noqa: E402

import config  # noqa: E402
from agents import _tool_text  # noqa: E402
from graph import build_graph, criar_mcp_client  # noqa: E402
from retriever import get_llm, vectorstore_existe  # noqa: E402

# Silencia os logs HTTP do cliente Ollama para manter a CLI limpa.
for _noisy in ("httpx", "mcp"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

console = Console()


def _cabecalho() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]Assistente Acadêmico Multiagente — UPF[/bold cyan]\n"
            "[dim]Consultor + Verificador · RAG · LLM local (Ollama) · MCP[/dim]\n\n"
            f"Modelo: [bold]{config.LLM_MODEL}[/bold]   "
            f"Embeddings: [bold]{config.EMBED_MODEL}[/bold]\n"
            "Digite sua dúvida acadêmica ou um comando ([bold]/ajuda[/bold]).",
            border_style="cyan",
        )
    )


def _ajuda() -> None:
    tabela = Table(title="Comandos", show_header=True, header_style="bold")
    tabela.add_column("Comando", style="cyan")
    tabela.add_column("Descrição")
    tabela.add_row("/docs", "Lista os documentos institucionais indexados")
    tabela.add_row("/resumo <arquivo>", "Resume um documento (Tool 3 via MCP)")
    tabela.add_row("/ajuda", "Mostra esta ajuda")
    tabela.add_row("/sair", "Encerra o assistente")
    console.print(tabela)


async def _cmd_docs(tools: dict) -> None:
    bruto = await tools["consulta_metadados"].ainvoke({})
    try:
        docs = json.loads(_tool_text(bruto))
    except json.JSONDecodeError:
        console.print("[red]Não foi possível ler os metadados.[/red]")
        return
    if isinstance(docs, dict) and docs.get("erro"):
        console.print(f"[red]{docs['erro']}[/red]")
        return

    tabela = Table(title="Base documental indexada", show_lines=False)
    tabela.add_column("Título", style="cyan")
    tabela.add_column("Categoria", style="magenta")
    tabela.add_column("Arquivo", style="dim")
    tabela.add_column("Pág.", justify="right")
    tabela.add_column("Chunks", justify="right", style="green")
    for d in docs:
        tabela.add_row(
            d.get("titulo", ""),
            d.get("categoria", ""),
            d.get("fonte", ""),
            str(d.get("paginas", "")),
            str(d.get("chunks", "")),
        )
    console.print(tabela)


async def _cmd_resumo(tools: dict, fonte: str) -> None:
    if not fonte:
        console.print("[yellow]Uso: /resumo <arquivo.pdf>[/yellow]")
        return
    with console.status(f"[cyan]Resumindo {fonte}...[/cyan]"):
        bruto = await tools["resumo_documento"].ainvoke({"fonte": fonte})
    try:
        dados = json.loads(_tool_text(bruto))
    except json.JSONDecodeError:
        console.print("[red]Falha ao gerar o resumo.[/red]")
        return
    if dados.get("erro"):
        console.print(f"[red]{dados['erro']}[/red]")
        return
    console.print(
        Panel(
            Markdown(dados.get("resumo", "")),
            title=f"Resumo — {dados.get('titulo', fonte)}",
            border_style="green",
        )
    )


async def _responder(graph, pergunta: str) -> None:
    """Executa o grafo e mostra a atuação de cada agente em tempo real."""
    estado_final: dict = {}
    entrada = {"pergunta": pergunta, "iteracoes": 0}

    async for passo in graph.astream(entrada, stream_mode="updates"):
        for no, update in passo.items():
            if no == "consultor":
                n = len(update.get("contexto", []))
                console.print(
                    f"[cyan]🔎 Agente Consultor[/cyan] "
                    f"[dim]recuperou {n} trecho(s) e elaborou a resposta preliminar.[/dim]"
                )
            elif no == "verificador":
                veredito = update.get("veredito", "—")
                cor = "green" if veredito == "APROVADA" else "yellow"
                console.print(
                    f"[magenta]✅ Agente Verificador[/magenta] "
                    f"[dim]validou as evidências →[/dim] [{cor}]{veredito}[/{cor}]"
                )
            estado_final.update(update)

    resposta = estado_final.get("resposta_final") or estado_final.get(
        "resposta_preliminar", "(sem resposta)"
    )
    console.print(
        Panel(Markdown(resposta), title="Resposta", border_style="cyan")
    )

    fontes = estado_final.get("fontes") or []
    if fontes:
        console.print(
            "[dim]Documentos consultados:[/dim] " + ", ".join(f"[cyan]{f}[/cyan]" for f in fontes)
        )


async def run() -> None:
    _cabecalho()

    if not vectorstore_existe():
        console.print(
            Panel.fit(
                "[red]Base vetorial não encontrada.[/red]\n"
                "Rode a ingestão primeiro:\n\n"
                "   [bold]python src/ingest.py[/bold]",
                border_style="red",
            )
        )
        return

    llm = get_llm(temperature=0.1)
    client = criar_mcp_client()

    # Mantém uma única sessão MCP aberta (servidor "quente") durante todo o uso.
    try:
        async with client.session("upf") as session:
            tools_list = await load_mcp_tools(session)
            tools = {t.name: t for t in tools_list}
            graph = build_graph(tools, llm)

            console.print(
                f"[dim]Tools MCP carregadas: "
                f"{', '.join(sorted(tools)) or 'nenhuma'}[/dim]\n"
            )

            while True:
                try:
                    entrada = await asyncio.to_thread(
                        Prompt.ask, "[bold green]você[/bold green]"
                    )
                except (EOFError, KeyboardInterrupt):
                    break

                entrada = entrada.strip()
                if not entrada:
                    continue

                if entrada in ("/sair", "/exit", "/quit"):
                    break
                if entrada == "/ajuda":
                    _ajuda()
                    continue
                if entrada == "/docs":
                    await _cmd_docs(tools)
                    continue
                if entrada.startswith("/resumo"):
                    partes = entrada.split(maxsplit=1)
                    await _cmd_resumo(tools, partes[1] if len(partes) > 1 else "")
                    continue
                if entrada.startswith("/"):
                    console.print("[yellow]Comando desconhecido. Use /ajuda.[/yellow]")
                    continue

                try:
                    await _responder(graph, entrada)
                except Exception as exc:  # noqa: BLE001
                    console.print(
                        f"[red]Erro ao processar a pergunta:[/red] {exc}\n"
                        "[dim]Verifique se o Ollama está em execução "
                        "(`ollama serve`) e se os modelos foram baixados.[/dim]"
                    )
    except Exception as exc:  # noqa: BLE001
        console.print(
            Panel.fit(
                f"[red]Falha ao iniciar o servidor MCP / sessão.[/red]\n{exc}\n\n"
                "[dim]Confira se as dependências foram instaladas e se o Ollama "
                "está ativo.[/dim]",
                border_style="red",
            )
        )

    console.print("\n[dim]Até mais! 👋[/dim]")


if __name__ == "__main__":
    asyncio.run(run())
