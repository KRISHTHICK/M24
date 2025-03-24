from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample data
documents = ["This is a sample document.", "Another document for testing.", "Yet another document."]
doc_embeddings = model.encode(documents)

# Create FAISS index
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

# Save the index and model
faiss.write_index(index, "faiss_index.bin")
model.save("sentence_transformer_model")
