import pickle
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# carregar docs processados
with open("processed_docs.pkl", "rb") as f:
    chunked_documents = pickle.load(f)

# criar embeddings
embeddings = OpenAIEmbeddings()

# criar index
vectorstore = FAISS.from_documents(chunked_documents, embeddings)

# salvar index
vectorstore.save_local("faiss_index")