@app.route('/update_data', methods=['POST'])
def update_data():
    new_documents = request.json['documents']
    new_embeddings = model.encode(new_documents)
    index.add(np.array(new_embeddings))
    faiss.write_index(index, "faiss_index.bin")
    return jsonify({'status': 'index updated'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
