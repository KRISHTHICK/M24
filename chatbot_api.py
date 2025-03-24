from flask import Flask, request, jsonify

app = Flask(__name__)

# Load tokenizer, retriever, and model
tokenizer = RagTokenizer.from_pretrained("rag_tokenizer")
retriever = RagRetriever.from_pretrained("rag_model", index_name="custom", passages_path="faiss_index.bin")
model = RagSequenceForGeneration.from_pretrained("rag_model", retriever=retriever)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['input']
    inputs = tokenizer(user_input, return_tensors="pt")
    generated_ids = model.generate(**inputs)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
