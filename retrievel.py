from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.load_local("faiss_index", embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})