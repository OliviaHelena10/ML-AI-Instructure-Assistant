import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# carregar docs processados
with open("processed_docs.pkl", "rb") as f:
    chunked_documents = pickle.load(f)

# criar embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# criar index
vectorstore = FAISS.from_documents(chunked_documents, embeddings)

# salvar index
vectorstore.save_local("faiss_index")