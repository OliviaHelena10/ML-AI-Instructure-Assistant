from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from typer.cli import docs
from langchain_huggingface import HuggingFaceEmbeddings

# carregar tudo uma vez (evita custo por request)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "faiss_index", 
    embeddings,
    allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOllama(
    model="llama3",   # ou llama3, dependendo do que você baixou
    temperature=0
)

def split_docs(docs):
    ml_docs = [d for d in docs if d.metadata["source"] == "pdf_machine_learning"]
    htl_docs = [d for d in docs if d.metadata["source"] == "pdf_htl"]
    return ml_docs, htl_docs


def build_context(ml_docs, htl_docs):
    ml_text = "\n".join([d.page_content for d in ml_docs])
    htl_text = "\n".join([d.page_content for d in htl_docs])

    return f"""
    CONTEÚDO TÉCNICO:
    {ml_text}

    ESTRATÉGIA DIDÁTICA:
    {htl_text}
    """


def answer_question(question: str, mode="tecnico-didatico") -> str:
    docs = retriever.invoke(question)

    ml_docs, htl_docs = split_docs(docs)
    context = build_context(ml_docs[:5], htl_docs[:2])

    prompt = f"""
    Você é um tutor especialista em Machine Learning.

    OBJETIVO:
    Ensinar de forma clara e didática usando o contexto fornecido.

    REGRAS:
    - Foque na clareza e didática, não apenas na precisão técnica
    - Explique como se fosse para alguém aprendendo
    - Priorize o conteúdo técnico, mas use a estratégia didática para guiar a resposta
    - Faça analogias e exemplos para facilitar o entendimento
    - Estimule o pensamento crítico fazendo perguntas de acompanhamento
    - Use exemplos quando possível
    - NÃO invente conteúdo fora do contexto

    Pergunta: {question}

    Contexto:
    {context}
    """

    print(f"\nPergunta: {question}")
    print(f"Docs retornados: {len(docs)}")

    response = llm.invoke(prompt)
    return response.content