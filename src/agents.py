"""
Agentes do sistema multiagente: Consultor e Verificador.

Cada agente é um nó do grafo LangGraph. As funções `make_consultor` e
`make_verificador` recebem as tools (carregadas via MCP) e o LLM local, e
devolvem a função-nó correspondente.

    • Agente Consultor  — interpreta a pergunta, recupera evidências via RAG
      (tool `busca_semantica`) e elabora uma resposta preliminar.
    • Agente Verificador — valida a resposta contra as evidências, consulta os
      metadados oficiais (tool `consulta_metadados`) e produz a resposta final
      com as referências. Pode reprovar a resposta e pedir reprocessamento.
"""

from __future__ import annotations

import json
from typing import Any, Callable, TypedDict

import config


# Estado compartilhado do grafo 
class AgentState(TypedDict, total=False):
    pergunta: str
    contexto: list[dict]       # trechos recuperados (conteudo + metadados)
    fontes: list[str]          # documentos citados
    resposta_preliminar: str   # produzida pelo Consultor
    resposta_final: str        # produzida pelo Verificador
    veredito: str              # "APROVADA" | "REPROVADA"
    iteracoes: int             # quantas vezes o Consultor já rodou


# ─── Utilidades ────────────────────────────────────────────────────────────
def _tool_text(resultado: Any) -> str:
    """Normaliza o retorno de uma tool MCP para texto puro."""
    if isinstance(resultado, str):
        return resultado
    if isinstance(resultado, (list, tuple)) and resultado:
        primeiro = resultado[0]
        if isinstance(primeiro, str):
            return primeiro
        if isinstance(primeiro, dict):
            return primeiro.get("text", json.dumps(primeiro, ensure_ascii=False))
    return str(resultado)


def _formatar_evidencias(contexto: list[dict]) -> str:
    """Formata os trechos recuperados em uma lista numerada para o prompt."""
    linhas = []
    for i, trecho in enumerate(contexto, start=1):
        titulo = trecho.get("titulo", trecho.get("fonte", "documento"))
        pagina = trecho.get("pagina")
        ref = f"{titulo}" + (f", p.{pagina}" if pagina else "")
        linhas.append(f"[{i}] ({ref})\n{trecho.get('conteudo', '').strip()}")
    return "\n\n".join(linhas)


# ─── Agente Consultor ──────────────────────────────────────────────────────
def make_consultor(tools: dict, llm) -> Callable:
    busca = tools["busca_semantica"]

    async def consultor_node(state: AgentState) -> dict:
        pergunta = state["pergunta"]
        iteracoes = state.get("iteracoes", 0) + 1

        # Na reexecução solicitada pelo Verificador, amplia a recuperação.
        k = config.TOP_K + (2 if iteracoes > 1 else 0)

        bruto = await busca.ainvoke({"query": pergunta, "k": k})
        try:
            dados = json.loads(_tool_text(bruto))
        except (json.JSONDecodeError, TypeError):
            dados = []

        if isinstance(dados, dict) and dados.get("erro"):
            return {
                "contexto": [],
                "fontes": [],
                "resposta_preliminar": (
                    "Não foi possível acessar a base documental "
                    f"({dados['erro']})."
                ),
                "iteracoes": iteracoes,
            }

        contexto: list[dict] = dados if isinstance(dados, list) else []
        if not contexto:
            return {
                "contexto": [],
                "fontes": [],
                "resposta_preliminar": (
                    "Não encontrei informações sobre essa pergunta nos documentos "
                    "institucionais disponíveis. Recomendo consultar diretamente a "
                    "secretaria ou a coordenação do curso."
                ),
                "iteracoes": iteracoes,
            }

        fontes = list(dict.fromkeys(t.get("titulo", t.get("fonte")) for t in contexto))
        evidencias = _formatar_evidencias(contexto)

        prompt = (
            "Você é o Agente Consultor de um assistente acadêmico da Universidade "
            "de Passo Fundo (UPF). Responda à pergunta do estudante usando "
            "EXCLUSIVAMENTE as evidências abaixo, extraídas de documentos oficiais. "
            "Não invente regras, prazos ou números que não estejam nas evidências. "
            "Se as evidências forem insuficientes, diga isso claramente. "
            "Cite as evidências usadas no formato [n].\n\n"
            f"PERGUNTA:\n{pergunta}\n\n"
            f"EVIDÊNCIAS:\n{evidencias}\n\n"
            "RESPOSTA (clara, objetiva, em português):"
        )
        resposta = await llm.ainvoke(prompt)
        texto = resposta.content if hasattr(resposta, "content") else str(resposta)

        return {
            "contexto": contexto,
            "fontes": fontes,
            "resposta_preliminar": texto.strip(),
            "iteracoes": iteracoes,
        }

    return consultor_node


# ─── Agente Verificador ────────────────────────────────────────────────────
def make_verificador(tools: dict, llm) -> Callable:
    consulta_metadados = tools["consulta_metadados"]

    async def verificador_node(state: AgentState) -> dict:
        pergunta = state["pergunta"]
        preliminar = state.get("resposta_preliminar", "")
        contexto = state.get("contexto", [])

        # Sem contexto não há o que verificar: finaliza com a resposta segura.
        if not contexto:
            return {
                "resposta_final": preliminar,
                "veredito": "APROVADA",
            }

        # Tool 2 (MCP): obtém a lista oficial de documentos para as referências.
        try:
            meta_bruto = await consulta_metadados.ainvoke({})
            documentos = json.loads(_tool_text(meta_bruto))
            if isinstance(documentos, dict):
                documentos = []
        except (json.JSONDecodeError, TypeError):
            documentos = []

        catalogo = "\n".join(
            f"- {d.get('titulo')} ({d.get('categoria')})" for d in documentos
        ) or "(catálogo indisponível)"

        evidencias = _formatar_evidencias(contexto)
        prompt = (
            "Você é o Agente Verificador de um assistente acadêmico da UPF. Sua "
            "função é auditar a resposta preliminar produzida pelo Agente Consultor, "
            "verificando se TODA afirmação está fundamentada nas evidências.\n\n"
            "Faça o seguinte:\n"
            "1. Remova ou corrija qualquer afirmação não suportada pelas evidências.\n"
            "2. Mantenha a resposta clara e em português.\n"
            "3. Ao final, acrescente uma seção 'Fontes consultadas:' listando os "
            "documentos oficiais efetivamente usados (com página quando houver).\n"
            "4. Na ÚLTIMA linha, escreva exatamente 'VEREDITO: APROVADA' se a "
            "resposta preliminar já estava bem fundamentada, ou 'VEREDITO: REPROVADA' "
            "se foi necessário corrigir algo relevante.\n\n"
            f"PERGUNTA:\n{pergunta}\n\n"
            f"RESPOSTA PRELIMINAR:\n{preliminar}\n\n"
            f"EVIDÊNCIAS:\n{evidencias}\n\n"
            f"CATÁLOGO DE DOCUMENTOS OFICIAIS:\n{catalogo}\n\n"
            "RESPOSTA FINAL VALIDADA:"
        )
        saida = await llm.ainvoke(prompt)
        texto = (saida.content if hasattr(saida, "content") else str(saida)).strip()

        # Extrai o veredito e o remove do corpo da resposta final.
        veredito = "APROVADA"
        linhas = texto.splitlines()
        if linhas and "VEREDITO:" in linhas[-1].upper():
            veredito = "REPROVADA" if "REPROVADA" in linhas[-1].upper() else "APROVADA"
            texto = "\n".join(linhas[:-1]).strip()

        return {
            "resposta_final": texto,
            "veredito": veredito,
        }

    return verificador_node
