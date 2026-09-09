# AskYourDocs

Um assistente de RAG (Retrieval-Augmented Generation) que responde perguntas com base em
PDFs que você mesmo envia — com fontes citadas (arquivo + página) e sem inventar resposta
quando não há contexto suficiente.

Este é um projeto de portfólio focado em **arquitetura de backend**, não em "chamar a API
da OpenAI". O objetivo é mostrar como estruturar um pipeline de ingestão de documentos,
busca vetorial, orquestração de prompt e geração de resposta como um sistema real —
com separação de camadas, tratamento de erro explícito, cache e testes.

## O problema que ele resolve

LLMs "cruas" não sabem nada sobre documentos privados (contratos, manuais internos,
políticas de RH) e, quando forçadas a responder sobre eles, tendem a alucinar. RAG resolve
isso separando duas responsabilidades:

1. **Recuperação (Retrieval):** encontrar os trechos dos seus documentos que são
   realmente relevantes para a pergunta, usando busca por similaridade vetorial.
2. **Geração (Generation):** dar esses trechos como contexto para o LLM e pedir para ele
   responder **apenas** com base neles — citando de onde tirou a informação.

O diferencial deste projeto é que, quando a recuperação não encontra nada relevante o
suficiente, o sistema **se recusa a responder** em vez de deixar o LLM "chutar" (ver seção
[Por que não alucinar](#tratamento-de-erros-e-anti-alucinação)).

## Arquitetura

```
                         ┌─────────────────────┐
                         │   Frontend (HTML/JS) │
                         │  upload PDF + chat    │
                         └──────────┬───────────┘
                                    │ HTTP/JSON
                                    ▼
                         ┌─────────────────────┐
                         │     FastAPI (API)    │
                         │  routes_documents /   │
                         │     routes_chat       │
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
    ┌──────────────────┐ ┌────────────────────┐ ┌──────────────────┐
    │ ingestion/        │ │ retrieval/          │ │ cache/            │
    │ pdf_loader.py      │ │ vector_store.py      │ │ query_cache.py    │
    │ chunking.py        │ │ service.py (RAG)     │ │ TTL cache +       │
    │ pipeline.py        │ │                     │ │ rate limiter      │
    └─────────┬─────────┘ └──────────┬──────────┘ └──────────────────┘
              │                      │
              ▼                      ▼
    ┌───────────────────────────────────────┐      ┌───────────────────┐
    │              llm/                       │      │        db/         │
    │  embeddings.py  (OpenAI | local)         │◄────►│ models.py           │
    │  providers.py   (OpenAI | Anthropic)     │      │ session.py          │
    │  prompts.py                              │      │ Postgres + pgvector │
    └───────────────────────────────────────┘      └───────────────────┘
```

**Fluxo de upload (`POST /documents`):**
`PDF` → `pdf_loader` extrai texto por página → `chunking` quebra cada página em
trechos com overlap → `embeddings` gera um vetor por trecho (em lote) → tudo é salvo em
`documents` + `chunks` (Postgres/pgvector), guardando `filename` e `page_number` por trecho.

**Fluxo de pergunta (`POST /chat`):**
Pergunta → cache (se já foi perguntada, responde na hora) → embedding da pergunta →
busca por similaridade de cosseno no pgvector (top-k trechos) → **gate de relevância**
(se o melhor trecho está abaixo de um score mínimo, recusa a responder) → monta prompt
com os trechos como contexto → chama o LLM configurado → retorna resposta + lista de
fontes (arquivo + página) deduplicada.

### Camadas e por que essa separação

- **`api/`** — só HTTP: valida request, injeta dependências, traduz exceções de domínio em
  respostas JSON. Não tem lógica de negócio.
- **`ingestion/` e `retrieval/`** — a lógica de negócio (RAG em si). Não sabem nada sobre
  FastAPI nem sobre qual provedor de LLM está configurado além da interface abstrata.
- **`llm/`** — toda a integração com IA (embeddings e geração de texto) atrás de
  interfaces (`EmbeddingProvider`, `LLMProvider`). Trocar de OpenAI para Anthropic, ou de
  API paga para modelo local, é trocar uma variável de ambiente — nenhum outro módulo
  precisa saber a diferença.
- **`db/`** — modelos SQLAlchemy e sessão. Nenhuma query SQL vaza para fora de
  `retrieval/vector_store.py`.
- **`cache/`** — cross-cutting concern isolado, injetado via dependência do FastAPI.

## Tratamento de erros e anti-alucinação

Cada situação de erro tem uma exceção de domínio própria (`app/core/exceptions.py`),
capturada por um `exception_handler` dedicado no FastAPI, retornando um JSON claro:

| Situação | Status | Código |
|---|---|---|
| Nenhum trecho relevante o suficiente (`similaridade < MIN_RELEVANCE_SCORE`) | 422 | `no_relevant_context` |
| Nenhum documento foi enviado ainda | 409 | `empty_knowledge_base` |
| Provedor de LLM/embeddings falhou (erro de API, key inválida, timeout) | 502 | `llm_provider_error` |
| PDF inválido, criptografado ou sem texto extraível | 422 | `unsupported_file` |
| Limite de requisições excedido | 429 | `rate_limit_exceeded` (com header `Retry-After`) |

O ponto central é o **gate de relevância** em `app/retrieval/service.py`: a similaridade
de cosseno do melhor trecho encontrado é comparada com `MIN_RELEVANCE_SCORE` *antes* de
qualquer chamada ao LLM. Se não passar, a pergunta nem chega a ser enviada ao modelo —
isso economiza tokens e evita que o LLM tente "adivinhar" uma resposta plausível a partir
de contexto fraco.

## Cache e rate limiting

- **Cache (`TTLCache`):** perguntas idênticas (normalizadas por case/espaço) reutilizam a
  resposta já gerada, evitando gastar tokens de embedding + LLM de novo. TTL e tamanho
  máximo configuráveis via env.
- **Rate limiting (`SlidingWindowRateLimiter`):** limite de requisições por IP numa janela
  deslizante, para conter abuso do endpoint de chat.

Ambos são implementações **em memória**, propositalmente simples — adequadas para uma
demo/portfólio rodando em um único processo. Ver [Decisões técnicas](#decisões-técnicas-e-porquês)
para a nota sobre como isso evoluiria em produção.

## Como rodar localmente

### Opção 1 — Docker Compose (recomendado)

```bash
cp backend/.env.example backend/.env
# edite backend/.env com suas chaves de API (ou deixe EMBEDDING_PROVIDER=local para não precisar de nenhuma)
docker compose up --build
```

- API em `http://localhost:8000` (docs interativas em `/docs`)
- Postgres com pgvector já configurado, extensão criada automaticamente no startup

Abra `frontend/index.html` diretamente no navegador (é só HTML+JS estático, sem build).

### Opção 2 — manual

```bash
# Postgres com pgvector rodando localmente, então:
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env  # ajuste DATABASE_URL e as chaves de API
uvicorn app.main:app --reload
```

### Rodando os testes

```bash
cd backend
pip install -r requirements-dev.txt
pytest -v
```

Os testes **não** precisam de um Postgres real nem de chaves de API: o endpoint de chat é
testado com o LLM e o embedding provider trocados por fakes via `dependency_overrides` do
FastAPI, e a busca vetorial é mockada com `monkeypatch`.

## Configuração (variáveis de ambiente)

Veja `backend/.env.example` para a lista completa. As mais importantes:

| Variável | Valores | Efeito |
|---|---|---|
| `EMBEDDING_PROVIDER` | `local` \| `openai` | `local` usa `sentence-transformers` (grátis, offline). `openai` usa `text-embedding-3-small`. |
| `LLM_PROVIDER` | `openai` \| `anthropic` | Qual API gera a resposta final. |
| `MIN_RELEVANCE_SCORE` | `0.0`–`1.0` | Quão exigente é o gate anti-alucinação. |
| `CHUNK_SIZE_CHARS` / `CHUNK_OVERLAP_CHARS` | inteiros | Tamanho/overlap dos trechos na ingestão. |

> Atenção: `EMBEDDING_DIMENSION` precisa bater com o modelo escolhido (384 para o modelo
> local MiniLM, 1536 para `text-embedding-3-small`). Ela define o tipo da coluna
> `Vector(...)` no Postgres, então mudar o provedor de embeddings depois de já ter
> documentos indexados exige reindexar tudo.

## Decisões técnicas e porquês

**Por que pgvector em vez de um vector DB dedicado (Pinecone, Weaviate, Qdrant)?**
Para este caso de uso (um portfólio, volume pequeno/médio de documentos), rodar a busca
vetorial dentro do mesmo Postgres que já guarda metadados elimina um serviço inteiro da
arquitetura: sem sincronizar dois bancos, sem uma segunda API para aprender, sem custo
extra. `pgvector` também torna trivial fazer joins entre `chunks` e `documents` numa única
query SQL (impossível de forma nativa num vector DB separado). A troca só valeria a pena
em escala bem maior (dezenas de milhões de vetores) ou se precisasse de filtros/algoritmos
de indexação (HNSW/IVF) mais sofisticados do que o Postgres oferece hoje.

**Por que embeddings locais como padrão, com OpenAI como opção?**
Um projeto de portfólio precisa ser fácil de rodar por qualquer avaliador, sem exigir uma
chave de API paga só para testar o upload de um PDF. `sentence-transformers` roda 100%
local e offline. A abstração `EmbeddingProvider` deixa claro que trocar para
`text-embedding-3-small` é uma decisão de configuração, não de código.

> **Nota de deploy real:** `sentence-transformers` traz PyTorch como dependência, que
> sozinho já passa de 500MB de RAM ao carregar o modelo — isso derruba (OOM kill) hosts
> free-tier pequenos como o plano gratuito do Render (512MB). Por isso o PyTorch foi
> movido para um arquivo de dependências separado (`requirements-local-embeddings.txt`,
> só instalado se você realmente for usar `EMBEDDING_PROVIDER=local`), e o deploy gratuito
> sugerido abaixo usa `EMBEDDING_PROVIDER=openai` em produção — local fica ótimo para rodar
> na sua própria máquina, onde RAM não é um problema.

**Por que endpoints síncronos (não `async def`) no FastAPI?**
As chamadas de I/O aqui (Postgres via `psycopg`, SDKs da OpenAI/Anthropic) são todas
síncronas. O FastAPI já executa handlers síncronos numa threadpool automaticamente, então
não há ganho real em reescrever tudo com clientes assíncronos para o volume de tráfego que
este projeto precisa suportar — só complexidade extra (sessões async do SQLAlchemy,
clientes async dos SDKs). Se o tráfego crescesse a ponto de a threadpool virar gargalo,
essa é uma migração isolada (camada `db/` e `llm/`), não uma reescrita do projeto.

**Por que chunking por caracteres com fallback de palavra/frase, em vez de um chunker
baseado em tokens (tiktoken)?**
É previsível, testável sem dependências pesadas e suficiente para o objetivo (não cortar
ideias/palavras no meio). Contar tokens exatos importa mais quando se está otimizando
custo de contexto agressivamente — não é o gargalo aqui.

**Por que cache/rate-limit em memória em vez de Redis?**
Simplicidade de deploy: um processo só, sem infraestrutura extra para rodar/testar
localmente. A interface (`TTLCache`, `SlidingWindowRateLimiter`) foi desenhada para ser
trocada por uma implementação com Redis sem tocar em `retrieval/service.py` ou nas rotas —
seria a primeira coisa a mudar antes de rodar múltiplas instâncias do backend atrás de um
load balancer.

**Por que `Base.metadata.create_all()` no startup em vez de Alembic?**
Reduz o setup para rodar o projeto localmente/em demo. Em um serviço real eu usaria
Alembic desde o primeiro dia para migrações versionadas e reversíveis — é a limitação mais
consciente deste projeto e a primeira coisa que eu adicionaria antes de ir para produção
de verdade.

## Deploy gratuito (sugestão)

1. **Banco (Postgres + pgvector):** crie um projeto no [Neon](https://neon.tech) ou
   [Supabase](https://supabase.com) (ambos têm tier gratuito com pgvector já disponível).
   Copie a connection string para `DATABASE_URL` (ajuste o driver para
   `postgresql+psycopg://...`).
2. **Backend (FastAPI):** crie um Web Service no [Render](https://render.com) apontando
   para a pasta `backend/` (ele detecta o `Dockerfile` automaticamente). Configure as
   variáveis de ambiente do `.env.example` no painel do Render, incluindo o
   `DATABASE_URL` do passo 1. **Use `EMBEDDING_PROVIDER=openai`** (com `EMBEDDING_DIMENSION=1536`
   e um `OPENAI_API_KEY`) em vez de `local` — o plano gratuito do Render só tem 512MB de
   RAM, insuficiente para o PyTorch que o modelo local exige (ver nota técnica acima).
3. **Frontend:** publique a pasta `frontend/` no [Vercel](https://vercel.com) (ou Netlify)
   como site estático. Defina `window.ASKYOURDOCS_API_BASE` (no `index.html`, antes de
   carregar `app.js`) para a URL pública do backend no Render.
4. **CORS:** adicione a URL do frontend publicado em `CORS_ORIGINS` nas envs do backend.
5. Teste o fluxo completo: upload de um PDF pequeno primeiro, depois uma pergunta — os
   planos gratuitos do Render "dormem" após inatividade, então a primeira requisição pode
   demorar alguns segundos para "acordar" o serviço.

## Limitações conhecidas

- PDFs escaneados (imagem, sem texto) não são suportados — não há OCR.
- Cache e rate limit são por processo (não compartilhados entre múltiplas instâncias).
- Sem autenticação: qualquer pessoa com acesso à API pode enviar documentos e perguntar.
  Para um deploy público real, seria o próximo item da lista (API key simples ou JWT).
