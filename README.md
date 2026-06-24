# Assistente Acadêmico Multiagente — UPF

Sistema **multiagente** baseado em **LLMs locais** que responde dúvidas acadêmicas
de estudantes da Universidade de Passo Fundo (UPF) consultando documentos
institucionais oficiais (regulamentos, editais, guias) por meio de **RAG**
(Retrieval-Augmented Generation), com ferramentas expostas via **MCP**
(Model Context Protocol).

> Trabalho final da disciplina de Inteligência Artificial — Profs. Diego A. Lusa e
> Roberto Rabello.

---

## Integrantes da equipe

- Carlos Henrique Güllich Scherer
- Pedro Rafael Domenighi

---

## Problema e objetivo

Universidades publicam muitos documentos institucionais (regulamentos acadêmicos,
editais, normas de estágio, calendários, resoluções). Embora públicos, encontrar
uma resposta específica costuma ser demorado: exige ler e interpretar documentos
extensos e frequentemente atualizados. Isso gera dúvidas recorrentes e aumenta a
demanda por atendimento administrativo.

**Objetivo:** oferecer um assistente de terminal que recupera trechos relevantes
dos documentos oficiais e responde em linguagem natural, **sempre fundamentado nas
fontes** e com uma etapa explícita de **validação** que reduz alucinações e cita as
referências utilizadas.

---

## Arquitetura multiagente

O sistema usa **dois agentes especializados** orquestrados com **LangGraph**, que
modela o fluxo como um grafo de estados. A separação de papéis torna o sistema mais
confiável que um agente único: um agente **gera** a resposta a partir do contexto e
o outro **audita** essa resposta contra as evidências antes de entregá-la.

```
        ┌─────────────┐      ┌──────────────┐      ┌────────────────┐
 USUÁRIO│   Pergunta  │ ───▶ │  Consultor   │ ───▶ │  Verificador   │ ───▶ Resposta final
  (CLI) └─────────────┘      │  (RAG + LLM) │      │ (validação +   │       + fontes
                             └──────┬───────┘      │  referências)  │
                                    │              └───────┬────────┘
                              busca_semantica               │ REPROVADA (reprocessa,
                              (Tool MCP)                     │ até MAX_ITER vezes)
                                    │                        ▼
                              ┌─────┴───────┐          volta ao Consultor
                              │  ChromaDB   │
                              │ (vetores)   │
                              └─────────────┘
```

### Papel de cada agente

| Agente | Responsabilidade | Entradas | Saídas |
|--------|------------------|----------|--------|
| **Consultor** | Interpreta a pergunta, recupera evidências via RAG (tool `busca_semantica`) e elabora a resposta preliminar usando **somente** o contexto recuperado. | Pergunta do usuário; trechos da base vetorial; metadados. | Resposta preliminar; evidências; lista de documentos. |
| **Verificador** | Audita a resposta preliminar contra as evidências, remove/corrige afirmações não suportadas, adiciona a seção de fontes (tool `consulta_metadados`) e emite um veredito. Pode **reprovar** e devolver ao Consultor para um novo ciclo. | Resposta preliminar; evidências; catálogo de documentos. | Resposta final validada; referências; veredito. |

O fluxo condicional (`REPROVADA → reprocessa`) é implementado com
`add_conditional_edges` do LangGraph e limitado por `MAX_ITER` para evitar loops.

---

## Tools dos agentes (expostas via MCP)

As ferramentas ficam em um **servidor MCP** (`src/mcp_server.py`, FastMCP) e são
carregadas pelos agentes através de `langchain-mcp-adapters`.

| Tool | Função | Usada por |
|------|--------|-----------|
| **`busca_semantica(query, k)`** | Consulta a base vetorial e retorna os `k` trechos mais relevantes com metadados e score. | Agente Consultor |
| **`consulta_metadados(fonte?)`** | Lista os documentos indexados (título, categoria, nº de páginas e de chunks). | Agente Verificador / comando `/docs` |
| **`resumo_documento(fonte, max_chars)`** | Gera, com o LLM local, um resumo objetivo de um documento. | Comando `/resumo` |

---

## Como o MCP foi utilizado

O **Model Context Protocol** padroniza o acesso ao recurso externo do sistema — a
**base documental institucional**. Em vez de os agentes acessarem o ChromaDB
diretamente, eles consomem **tools padronizadas** publicadas por um servidor MCP:

- `src/mcp_server.py` é um **servidor MCP** (FastMCP) que expõe a base como três
  tools e roda como **processo independente**, comunicando-se por **stdio**.
- O orquestrador (`src/graph.py`) cria um **cliente MCP** (`MultiServerMCPClient`)
  que inicia esse servidor e carrega as tools como ferramentas LangChain via
  `langchain-mcp-adapters`.
- A sessão MCP é mantida aberta durante toda a execução da CLI, mantendo o servidor
  "quente" (modelos e índice já carregados).

Isso desacopla os agentes da implementação concreta da base: trocar ChromaDB por
outro mecanismo não exigiria mudar os agentes, apenas o servidor MCP.

---

## Estratégia de RAG

1. **Ingestão** (`src/ingest.py`): cada PDF de `docs/` é lido página a página
   (preservando o número da página), o texto é dividido em **chunks** de
   ~1000 caracteres com sobreposição de 150 (`RecursiveCharacterTextSplitter`), e
   cada chunk recebe metadados: `fonte`, `titulo`, `categoria`, `pagina`.
2. **Indexação**: os chunks são convertidos em **embeddings locais**
   (`nomic-embed-text` via Ollama) e persistidos no **ChromaDB**.
3. **Recuperação**: a tool `busca_semantica` embeda a pergunta e retorna os trechos
   mais próximos por similaridade, com seus metadados.
4. **Geração fundamentada**: o Consultor responde **apenas** com base nesses
   trechos e os cita como `[n]`; o Verificador confere essa fundamentação.

Quando a recuperação não encontra contexto, o sistema declara que não há base
documental para a pergunta — em vez de inventar uma resposta.

---

## Base de conhecimento

- **Origem:** documentos públicos da UPF (portal e `download.upf.br`).
- **Natureza:** PDFs institucionais — Guia Acadêmico, Dispositivos Regimentais e
  Legais (graduação), e editais recentes (bolsas, seleção, transferência).
- **Localização:** pasta [`docs/`](docs/). Para ampliar a base, basta adicionar
  novos PDFs nessa pasta e rodar a ingestão novamente.

> PDFs digitalizados (somente imagem) são ignorados na ingestão com um aviso, pois
> exigiriam OCR.

---

## Tecnologias

| Camada | Tecnologia | Por quê |
|--------|-----------|---------|
| Orquestração | **LangGraph** | Modela o fluxo Consultor → Verificador como grafo de estados, com fluxo condicional explícito e fácil de justificar. |
| Modelo local | **Ollama + `llama3.1:8b`** | Boa qualidade de compreensão/geração rodando localmente (~8 GB), sem APIs pagas. |
| Embeddings | **`nomic-embed-text`** (via Ollama) | Embeddings locais e gratuitos, integrados ao mesmo runtime do LLM. |
| Vetores | **ChromaDB** | Armazenamento vetorial persistente e embutido, sem servidor separado. |
| Tools / Protocolo | **MCP (FastMCP)** + `langchain-mcp-adapters` | Padroniza o acesso à base como ferramentas. |
| Interface | **Rich** | CLI formatada para uma demonstração clara. |
| Linguagem | **Python 3.11+** | Stack inteiro em Python. |

---

## Estrutura do projeto

```
.
├── docs/                  # Base de conhecimento (PDFs institucionais)
├── src/
│   ├── config.py          # Configuração central (modelos, caminhos, parâmetros)
│   ├── retriever.py       # Acesso ao Ollama (LLM/embeddings) e ao ChromaDB
│   ├── ingest.py          # Ingestão: PDF → chunks → embeddings → ChromaDB
│   ├── mcp_server.py      # Servidor MCP (FastMCP) com as 3 tools
│   ├── agents.py          # Agentes Consultor e Verificador (nós do grafo)
│   ├── graph.py           # Orquestração LangGraph + cliente MCP
│   └── main.py            # Interface de terminal (CLI)
├── chroma_db/             # Base vetorial gerada (não versionada)
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Instalação e execução

### Pré-requisitos

- **Python 3.11+**
- **Ollama** instalado e em execução — <https://ollama.com/download>

### 1. Instalar o Ollama e baixar os modelos

```bash
ollama serve            # inicia o servidor (se ainda não estiver rodando)
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 2. Criar o ambiente Python e instalar dependências

```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. (Opcional) Configurar variáveis

```bash
cp .env.example .env      # ajuste modelos
```

### 4. Construir a base de conhecimento (RAG)

```bash
python src/ingest.py
```

### 5. Executar o assistente

```bash
python src/main.py
```

---

## Exemplos de uso (terminal)

```text
você: Quais são os requisitos para me transferir para medicina na UPF?
Agente Consultor recuperou 5 trecho(s) e elaborou a resposta preliminar.
Agente Verificador validou as evidências → APROVADA

╭───────────────────────────── Resposta ─────────────────────────────╮
│ Para transferir para a Medicina na UPF, é necessário...                │
│                                                                     │
│ Fontes consultadas:                                                 │
│ - Guia Acadêmico da UPF, p.34                                       │
╰─────────────────────────────────────────────────────────────────────╯
Documentos consultados: Edital de Transferência 2025/2 — Medicina, Guia Acadêmico da UPF
```

Comandos disponíveis:

```text
você: /docs                      # lista os documentos indexados
você: /resumo guia_academico.pdf # resume um documento (Tool 3 via MCP)
você: /ajuda                     # ajuda
você: /sair                      # encerra
```

---

## Reprodutibilidade

- A base vetorial (`chroma_db/`) **não é versionada**: reconstrua com
  `python src/ingest.py`.
- Parâmetros (modelos, `TOP_K`, tamanho de chunk, `MAX_ITER`) ficam em
  `src/config.py` / `.env`.

---

## Observações

- O sistema opera **100% com modelos locais** (Ollama); nenhuma API paga é
  necessária.
- A primeira execução do `ingest.py` pode levar alguns minutos, dependendo do
  hardware, pois gera os embeddings de todos os documentos.
