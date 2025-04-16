# 1) Install dependencies if needed:
# !pip install chromadb sentence-transformers

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# 2) Load a local model:
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 3) Create an embedding function:
def embedding_fn(texts):
    # The .encode() method returns NumPy arrays;
    # Chroma expects lists of lists, so convert them.
    return embedding_model.encode(texts).tolist()

# 4) Create a Chroma client and collection:
chroma_client = chromadb.Client(Settings(
    # e.g., specify a persist directory if you like
    persist_directory="path/to/chroma_data"  
))

collection = chroma_client.create_collection(
    name="my_collection",
    embedding_function=embedding_fn
)

# 5) Add documents on the fly:
documents = ["This is my first document.", "Here is my second document."]
ids = ["doc1", "doc2"]  # unique IDs for each document

collection.add(documents=documents, ids=ids)

# 6) Query by embedding:
query_texts = ["What is in my first doc?"]
results = collection.query(
    query_texts=query_texts,
    n_results=1  # how many matches to return
)

print(results)
