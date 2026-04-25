import os
import pickle
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, length, lit, lower, regexp_replace, when
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

 
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
    """Load a PDF with PyPDFLoader and return page-level records with metadata."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    records = []

    for doc in docs:
        records.append({
            "page_content": doc.page_content,
            "source": source_label,
            "content_type": content_type,
            "file_name": os.path.basename(pdf_path),
            "page": doc.metadata.get("page"),
        })

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
        # Load each PDF and store its page content plus metadata.
        all_records.extend(load_pdf_records(pdf_file, source_label))

if not all_records:
    raise RuntimeError("No PDF files found in the configured folders.")


# ---------------------------------------------------------------------
# Spark pipeline: text cleaning and enrichment
# ---------------------------------------------------------------------
# Spark lets us apply transformations consistently across many rows.
spark = SparkSession.builder.appName("pdf_enrichment_pipeline").getOrCreate()
df = spark.createDataFrame(all_records)

# Remove line breaks from the raw page text.
df = df.withColumn("text", regexp_replace(col("page_content"), r"\n+", " "))

# Remove hyphenation artifacts from broken words.
df = df.withColumn("text", regexp_replace(col("text"), r"-\s+", ""))

# Normalize whitespace so the text is cleaner for embeddings.
df = df.withColumn("text", regexp_replace(col("text"), r"\s+", " "))

# Convert text to lowercase for more consistent representation.
df = df.withColumn("text", lower(col("text")))

# Remove common footer markers such as 'Page 1'.
df = df.filter(~col("text").rlike("Page \\d+"))

# Drop very short text segments that usually do not help embeddings.
df = df.filter(length(col("text")) > 80)

# Add enrichment context directly into the text string.
# This makes embeddings aware of source and page without requiring external metadata.
page_label = when(
    col("page").isNotNull(),
    concat_ws(" ", lit("Página:"), col("page").cast("string")),
).otherwise(lit(None))

df = df.withColumn("text", concat_ws(" | ", col("source"), page_label, col("text")))


# ---------------------------------------------------------------------
# Convert cleaned Spark rows back to LangChain Document objects
# ---------------------------------------------------------------------
# This step collects the Spark DataFrame into local Python memory.
records = df.select("text", "source", "content_type", "file_name", "page").toPandas().to_dict(orient="records")

documents = [
    Document(
        page_content=record["text"],
        metadata={
            # Keep source/type metadata distinct from the enriched text.
            # Metadata is used later for filters, document grouping, and agent logic.
            "source": record["source"],
            "content_type": record["content_type"],
            "file_name": record["file_name"],
            "page": record.get("page"),
        },
    )
    for record in records
]


# ---------------------------------------------------------------------
# Optional chunking for more effective embeddings and retrieval
# ---------------------------------------------------------------------
# Splitting text into smaller chunks helps retrieval systems match precise
# information and reduces the chance of irrelevant long passages dominating.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunked_documents = []
for document in documents:
    for chunk in text_splitter.split_text(document.page_content):
        chunked_documents.append(Document(page_content=chunk, metadata=document.metadata))

print(f"Loaded {len(documents)} enriched documents and {len(chunked_documents)} chunked documents.")


with open("processed_docs.pkl", "wb") as f:
    pickle.dump(chunked_documents, f)