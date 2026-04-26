import os
import pickle
from langchain_classic import text_splitter
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, length, lit, lower, regexp_replace, when
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Separating files types into different folders to apply different rules for ML and AI files
# vs HTL files. This allows for more tailored processing and enrichment based on the content type.
    
def list_pdf_files(folder_path):
    if not os.path.isdir(folder_path):
        return []
    return [
        os.path.join(folder_path, file_name)
        for file_name in os.listdir(folder_path)
        if file_name.lower().endswith(".pdf")
    ]

def load_pdf_records(pdf_path, source_label, content_type="material_estudo"):
    """Load a PDF safely and return structured records."""

    print(f"📄 Processando: {pdf_path}")

    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.lazy_load()  # 👈 evita travamento pesado
    except Exception as e:
        print(f"❌ Erro ao abrir {pdf_path}: {e}")
        return []

    records = []

    try:
        for doc in docs:
            if not doc.page_content:
                continue

            records.append({
                "page_content": doc.page_content,
                "source": source_label,
                "content_type": content_type,
                "file_name": os.path.basename(pdf_path),
                "page": doc.metadata.get("page"),
            })

    except Exception as e:
        print(f"❌ Erro ao processar {pdf_path}: {e}")
        return []

    return records


# ---------------------------------------------------------------------
# Configure input folders and source labels
# ---------------------------------------------------------------------
ml_ai_path = "./files/ML-AI-Files"
htl_path = "./files/HTL-Files"

pdf_sources = [
    (ml_ai_path, "pdf_machine_learning"),
    (htl_path, "pdf_htl"),
]


# ---------------------------------------------------------------------
# Load PDFs and convert them into structured records
# ---------------------------------------------------------------------
all_records = []
for folder_path, source_label in pdf_sources:
    for pdf_file in list_pdf_files(folder_path):
        records = load_pdf_records(pdf_file, source_label)
        all_records.extend(records)

if not all_records:
    raise RuntimeError("❌ Nenhum PDF válido encontrado.")


print(f"✅ Total de páginas carregadas: {len(all_records)}")


# ---------------------------------------------------------------------
# Spark pipeline: text cleaning and enrichment
# ---------------------------------------------------------------------
# Spark lets us apply transformations consistently across many rows.
spark = SparkSession.builder.appName("pdf_enrichment_pipeline").getOrCreate()
df = spark.createDataFrame(all_records)

df = df.withColumn("text", regexp_replace(col("page_content"), r"\n+", " "))
df = df.withColumn("text", regexp_replace(col("text"), r"-\s+", ""))
df = df.withColumn("text", regexp_replace(col("text"), r"\s+", " "))
df = df.withColumn("text", lower(col("text")))
df = df.filter(~col("text").rlike("Page \\d+"))
df = df.filter(length(col("text")) > 80)

page_label = when(
    col("page").isNotNull(),
    concat_ws(" ", lit("Página:"), col("page").cast("string")),
).otherwise(lit(None))

df = df.withColumn("text", concat_ws(" | ", col("source"), page_label, col("text")))


# ---------------------------------------------------------------------
# Convert cleaned Spark rows back to LangChain Document objects
# ---------------------------------------------------------------------
# This step collects the Spark DataFrame into local Python memory.

records = df.select("text", "source", "content_type", "file_name", "page") \
    .toPandas() \
    .to_dict(orient="records")

documents = [
    Document(
        page_content=record["text"],
        metadata={
            "source": record["source"],
            "content_type": record["content_type"],
            "file_name": record["file_name"],
            "page": record.get("page"),
        },
    )
    for record in records
]

print(f"📚 Documentos processados: {len(documents)}")


# ---------------------------------------------------------------------
# Optional chunking for more effective embeddings and retrieval
# ---------------------------------------------------------------------
# Splitting text into smaller chunks helps retrieval systems match precise
# information and reduces the chance of irrelevant long passages dominating.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunked_documents = []

for document in documents:
    chunks = text_splitter.split_text(document.page_content)
    for chunk in chunks:
        chunked_documents.append(
            Document(
                page_content=chunk,
                metadata=document.metadata
            )
        )

print(f"✂️ Chunks gerados: {len(chunked_documents)}")


# ---------------------------------------------------------------------
# Save processed docs (SEM FAISS aqui)
# ---------------------------------------------------------------------

with open("processed_docs.pkl", "wb") as f:
    pickle.dump(chunked_documents, f)

print("💾 processed_docs.pkl salvo com sucesso!")