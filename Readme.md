# Enrichment Pipeline Documentation

## Sumário

- [Objetivo](#objetivo)
- [Visão geral do pipeline](#visão-geral-do-pipeline)
- [Por que enriquecimento é importante?](#por-que-enriquecimento-é-importante)
- [O que o código faz](#o-que-o-código-faz)
- [Estrutura do código e comentários](#estrutura-do-código-e-comentários)
- [Por que separar didático e conteúdo no metadata?](#por-que-separar-didático-e-conteúdo-no-metadata)
- [Como usar isso depois na recuperação](#como-usar-isso-depois-na-recuperação)
- [Por que essa abordagem é melhor?](#por-que-essa-abordagem-é-melhor)
- [Dependências e execução](#dependências-e-execução)
- [Aviso sobre tamanho de dados](#aviso-sobre-tamanho-de-dados)
- [Resumo final](#resumo-final)

## Objetivo

Este documento explica detalhadamente o que o script `code.py` faz e por quê.
O foco é transformar PDFs em documentos enriquecidos para uso em embeddings, busca semântica e RAG.

## Visão geral do pipeline

O fluxo principal é:

1. Carregar PDFs de pastas locais com `PyPDFLoader`
2. Transformar as páginas em registros estruturados
3. Limpar o texto com Spark
4. Enriquecer o texto com contexto adicional
5. Manter metadata separada para filtros e lógica de agente
6. Converter tudo em `Document` do LangChain
7. Divide os documentos em chunks para melhorar a recuperação

## Por que enriquecimento é importante?

Embeddings não entendem por si só:
- de onde veio o texto
- qual o tipo de documento
- contexto implícito

Por isso, precisamos injetar manualmente essas informações.

### Dois lugares para enriquecimento

1) Dentro do texto (`page_content`)

Exemplo:

```text
pdf_machine_learning | Página: 3 | o modelo apresentou overfitting
```

Isso influencia:
- embeddings
- busca semântica

2) Dentro do metadata

Exemplo:

```python
metadata = {
    "source": "pdf_machine_learning"
}
```

Isso influencia:
- filtros
- lógica do agente
- organização

## O que o código faz

### 1) Carregar os PDFs

O código procura arquivos PDF nas pastas configuradas:
- `./files/ML-AI-Files`
- `./files/HTL-Files`

Cada PDF é carregado com `PyPDFLoader` e transformado em registros por página com metadados:
- `source` (como `pdf_machine_learning` ou `pdf_htl`)
- `content_type` (por exemplo `material_estudo`)
- `file_name`
- `page`

Isso dá um registro estruturado para cada página do PDF.

### 2) Limpeza de texto com Spark

O texto de cada página passa por várias transformações:
- remove quebras de linha repetidas (`\n+`)
- remove hifenização de palavras quebradas (`-\s+`)
- normaliza espaços (`\s+`)
- converte para minúsculas
- remove rodapés como `Page 1`
- remove textos curtos com menos de 80 caracteres

Essas etapas deixam o texto mais limpo e melhoram a qualidade dos embeddings.

### 3) Enriquecimento dentro do texto

Depois da limpeza, o script adiciona informação no próprio campo de texto:

- `source` (por exemplo `pdf_machine_learning`)
- `Página: N`

Isso garante que os embeddings carreguem o contexto de origem e a posição do conteúdo.

### 4) Enriquecimento em metadata

Além disso, cada documento mantém metadata separada:

```python
metadata={
    "source": record["source"],
    "content_type": record["content_type"],
    "file_name": record["file_name"],
    "page": record.get("page"),
}
```

Isso permite uso posterior em filtros, agrupamentos, decisão de agente e montagem de prompt.

## Estrutura do código e comentários

O código está dividido em seções claras com comentários explicativos:

- **Imports e visão geral**: Explica o pipeline em alto nível e os dois tipos de enriquecimento.
- **Funções auxiliares**: `list_pdf_files` e `load_pdf_records` com docstrings.
- **Configuração**: Pastas de entrada e labels de source.
- **Carregamento**: Loop para carregar PDFs e criar registros.
- **Pipeline Spark**: Limpeza e enriquecimento do texto.
- **Conversão para LangChain**: Criação de objetos `Document` com metadata separada.
- **Chunking**: Divisão em chunks para melhor recuperação.

Comentários adicionados incluem:
- Explicação de por que Spark é usado para limpeza.
- Detalhes sobre cada transformação de texto.
- Motivo para manter metadata separada.
- Aviso sobre coleta de dados em memória.

## Por que separar didático e conteúdo no metadata?

A pergunta chave é:

> "Eu não deveria separar o que é didático e o que é conteúdo?"

Resposta:

- Sim, você deve separar isso
- Mas NÃO no próprio texto enriquecido

Você já está fazendo certo ao diferenciar `source` no metadata:
- `pdf_htl` vs `pdf_machine_learning`

Se você tentar pré-classificar demais no texto, perde flexibilidade.

## Como usar isso depois na recuperação

O caminho ideal é:

1) Buscar documentos relevantes

```python
docs = retriever.get_relevant_documents(pergunta)
```

2) Separar por tipo usando metadata

```python
ml_docs = [d for d in docs if d.metadata["source"] == "pdf_machine_learning"]
htl_docs = [d for d in docs if d.metadata["source"] == "pdf_htl"]
```

3) Montar contexto estruturado

```text
CONTEÚDO TÉCNICO:
... (texto do ml_docs)

FORMA DE ENSINAR:
... (texto do htl_docs)
```

4) Enviar ao modelo com instrução

```python
prompt = f"Use esse contexto para responder...

{conteudo_tecnico}

{forma_de_ensinar}"
```

### Exemplo completo de uso

Aqui está um exemplo mais completo de como integrar isso em um sistema RAG:

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Assumindo que 'chunked_documents' foi criado pelo script
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunked_documents, embeddings)
retriever = vectorstore.as_retriever()

# Busca por uma pergunta
pergunta = "Como ensinar machine learning?"
docs = retriever.get_relevant_documents(pergunta)

# Separa por source
ml_docs = [d for d in docs if d.metadata["source"] == "pdf_machine_learning"]
htl_docs = [d for d in docs if d.metadata["source"] == "pdf_htl"]

# Monta contexto
conteudo_tecnico = "\n".join([d.page_content for d in ml_docs])
forma_de_ensinar = "\n".join([d.page_content for d in htl_docs])

# Cria prompt estruturado
prompt = f"""
Use o contexto abaixo para responder à pergunta: {pergunta}

CONTEÚDO TÉCNICO:
{conteudo_tecnico}

FORMA DE ENSINAR:
{forma_de_ensinar}

Resposta:
"""

# Envia para o modelo (exemplo com OpenAI)
from langchain.llms import OpenAI
llm = OpenAI()
resposta = llm(prompt)
```

## Por que essa abordagem é melhor?

Se você decide o papel de cada documento antes da busca:
- perde flexibilidade
- limita a combinação de informações
- piora a recuperação

Se você decide isso após a busca:
- mantém controle total
- consegue montar respostas mais precisas
- constrói um sistema mais inteligente

## Dependências e execução

### Dependências

Para executar o script, você precisa:

- Python 3.8+
- PySpark
- LangChain
- PyPDFLoader (parte do LangChain)

Instale com:

```bash
pip install pyspark langchain pypdf
```

### Como executar

1. Coloque seus PDFs nas pastas `./files/ML-AI-Files` e `./files/HTL-Files`
2. Execute o script:

```bash
python code.py
```

O script irá imprimir o número de documentos carregados.

## Aviso sobre tamanho de dados

O script converte o DataFrame Spark em `pandas`:

```python
df.toPandas()
```

Isso funciona bem para conjuntos de dados moderados.
Se o dataset for gigante, você precisará de uma solução distribuída ou outra estratégia de processamento.

## Resumo final

- O enriquecimento é feito em dois lugares: texto e metadata
- Texto enriquecido ajuda embeddings e busca semântica
- Metadata separada ajuda filtros e lógica de agente
- A decisão sobre "didático vs conteúdo" deve ficar para depois da busca, não no dado bruto
- O pipeline atual segue essa ideia de forma clara e pedagógica

