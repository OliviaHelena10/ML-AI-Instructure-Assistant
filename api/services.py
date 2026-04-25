from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# carregar tudo uma vez (evita custo por request)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("faiss_index", embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(temperature=0)


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


def answer_question(question: str) -> str:
    docs = retriever.get_relevant_documents(question)

    ml_docs, htl_docs = split_docs(docs)
    context = build_context(ml_docs, htl_docs)

    prompt = f"""
    Você é um assistente de aprendizado.

    Regras:
    - Use o conteúdo técnico para explicar conceitos
    - Use a estratégia didática para ensinar de forma clara
    - Combine os dois de forma objetiva

    Pergunta: {question}

    Contexto:
    {context}
    """

    response = llm.predict(prompt)
    return response