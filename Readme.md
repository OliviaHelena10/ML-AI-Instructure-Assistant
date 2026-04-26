# AI Personal Agent Documentation

## Sumário

- [Visão geral](#visão-geral)
- [Objetivo do projeto](#objetivo-do-projeto)
- [Arquitetura e componentes](#arquitetura-e-componentes)
- [Fluxo de dados](#fluxo-de-dados)
- [Como rodar o projeto](#como-rodar-o-projeto)
- [API](#api)
- [Dependências](#dependências)
- [Estrutura de arquivos](#estrutura-de-arquivos)
- [Observações importantes](#observações-importantes)
- [Limitações](#limitações)
- [O que foi atualizado](#o-que-foi-atualizado)

## Visão geral

Este projeto é um assistente de aprendizado baseado em Inteligência Artificial que:

- lê PDFs de duas pastas separadas (`ML-AI-Files` e `HTL-Files`)
- limpa e enriquece o texto com Spark
- transforma o conteúdo em documentos LangChain
- cria um índice de similaridade FAISS
- expõe uma API FastAPI para responder perguntas usando um modelo de linguagem local

## Objetivo do projeto

Construir um pipeline completo de RAG (retrieval-augmented generation) para:

- separar conteúdo técnico de conteúdo pedagógico
- enriquecer o texto com metadata útil
- permitir respostas mais didáticas e contextualizadas
- servir um endpoint de pergunta/resposta via FastAPI

## Arquitetura e componentes

### `data_pipeline.py`

Responsável por:

- carregar PDFs de `./files/ML-AI-Files` e `./files/HTL-Files`
- criar registros por página usando `PyPDFLoader`
- aplicar limpeza e enriquecimento via Spark
- gerar `Document` do LangChain com metadata
- dividir cada documento em chunks de 1000 caracteres com 200 de overlap
- salvar `processed_docs.pkl`

Dados extraídos por página incluem:

- `source`: `pdf_machine_learning` ou `pdf_htl`
- `content_type`: tipo de conteúdo
- `file_name`: nome do arquivo PDF
- `page`: número da página

O texto processado também é enriquecido com:

- nome da fonte
- número da página

### `indexing.py`

Responsável por:

- carregar `processed_docs.pkl`
- gerar embeddings usando `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")`
- construir um índice FAISS com `FAISS.from_documents`
- salvar o índice local em `faiss_index`

### `api/main.py`

Define a API FastAPI do projeto:

- `GET /`: health check simples
- `POST /ask`: recebe pergunta e retorna resposta do agente

### `api/schemas.py`

Define os modelos Pydantic usados pela API:

- `QuestionRequest` com campo `question`
- `AnswerResponse` com campo `answer`

### `api/services.py`

Define a lógica de resposta:

- carrega o índice FAISS local com `allow_dangerous_deserialization=True`
- configura embeddings e retriever
- usa `ChatOllama(model="llama3", temperature=0)` como LLM
- separa documentos retornados por metadata de source
- monta contexto técnico e didático
- gera o prompt de pergunta com regras didáticas
- retorna a resposta do modelo

### `retrievel.py`

Tem um loader auxiliar para:

- carregar o mesmo `faiss_index`
- criar um retriever com `k=3`

Esse arquivo funciona como um helper de recuperação e validação independente da API.

## Fluxo de dados

1. `data_pipeline.py` processa PDFs em `processed_docs.pkl`
2. `indexing.py` gera `faiss_index/` a partir de `processed_docs.pkl`
3. `api/services.py` carrega `faiss_index/` e cria um retriever
4. `api/main.py` expõe `/ask` para responder perguntas

## Como rodar o projeto

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. Gere os documentos processados:

```bash
python data_pipeline.py
```

3. Crie o índice FAISS:

```bash
python indexing.py
```

4. Inicie a API:

```bash
uvicorn api.main:app --reload
```

5. Envie uma pergunta:

```bash
curl -X POST "http://127.0.0.1:8000/ask"   -H "Content-Type: application/json"   -d '{"question":"Explique o que é machine learning."}'
```

## API

### `GET /`

Retorna um health check:

```json
{ "status": "ok" }
```

### `POST /ask`

Entrada JSON:

```json
{ "question": "..." }
```

Resposta JSON:

```json
{ "answer": "..." }
```

## Dependências

O arquivo `requirements.txt` já lista as dependências do projeto. As principais são:

- `fastapi`
- `uvicorn`
- `pyspark`
- `langchain`
- `langchain-community`
- `langchain-huggingface`
- `langchain-classic`
- `sentence-transformers`
- `faiss-cpu`

## Estrutura de arquivos

- `data_pipeline.py` — pipeline de ingestão, limpeza e chunking de PDFs
- `indexing.py` — criação do índice FAISS
- `api/main.py` — aplicação FastAPI
- `api/schemas.py` — modelos de request/response Pydantic
- `api/services.py` — lógica de resposta e prompt do modelo
- `retrievel.py` — helper para carregar o índice e criar retriever
- `requirements.txt` — dependências do projeto
- `files/ML-AI-Files/` — PDFs de Machine Learning e AI
- `files/HTL-Files/` — PDFs de conteúdo pedagógico
- `faiss_index/` — índice FAISS salvo localmente
- `processed_docs.pkl` — documentos processados e chunkificados

## Observações importantes

- O projeto atual não usa `langchain_openai`.
- A integração atual utiliza `langchain-huggingface`, `langchain-community` e `ChatOllama`.
- `api/services.py` assume que um modelo compatível com `ChatOllama(model="llama3")` está disponível no ambiente.
- `FAISS.load_local` no serviço usa `allow_dangerous_deserialization=True` para carregar o índice salvo.
- `answer_question` usa `retriever.invoke(question)` em vez de `get_relevant_documents`.

## Limitações

- O pipeline converte o DataFrame Spark para pandas em memória.
- O `data_pipeline.py` falha se não houver PDFs válidos nas pastas de entrada.
- A API depende da existência do diretório `faiss_index/`.
- A montagem de prompt e contexto está codificada em `api/services.py`.
- O modelo local deve suportar `ChatOllama` e o nome do modelo `llama3`.

## O que foi atualizado

Este README agora documenta corretamente:

- o uso real de `data_pipeline.py`, `indexing.py`, `api/main.py`, `api/schemas.py`, `api/services.py` e `retrievel.py`
- o fluxo de geração de `processed_docs.pkl` e `faiss_index/`
- os metadados extraídos e o enriquecimento do texto
- a separação entre conteúdo técnico (`pdf_machine_learning`) e didático (`pdf_htl`)
- as dependências presentes em `requirements.txt`
- a API FastAPI e seus endpoints

## Resumo final

O projeto é uma solução de RAG voltada para perguntas sobre conteúdo técnico e pedagógico.
Ele combina:

- extração de PDFs
- limpeza de texto com Spark
- chunking de conteúdo
- indexação com FAISS
- respostas com um modelo de linguagem local via FastAPI
