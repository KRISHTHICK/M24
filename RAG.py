from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# Load RAG model and tokenizer
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="custom", passages_path="faiss_index.bin")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

# Save the RAG model
model.save_pretrained("rag_model")
tokenizer.save_pretrained("rag_tokenizer")
