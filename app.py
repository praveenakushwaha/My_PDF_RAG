from flask import Flask, render_template, request, jsonify
import os
# Ensure relative pathing works if your documents folder is outside or inside package paths
from rag import initialize_rag, ask_llm

app = Flask(__name__, template_folder='../templates')

# Initialize RAG on application startup
# Dynamically mapping target PDF path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "../documents/sampleTest.pdf")
db = initialize_rag(PDF_PATH)

@app.route('/')
def home():
    return render_template('index.html')
                                
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    question = data['question']
    
    try:
        # Retrieve context matches from the vector database
        retrieved_chunks = db.search(question)
        context = "\n".join(retrieved_chunks)
        
        # Get response from Qwen LLM
        answer = ask_llm(context, question)
        return jsonify({'answer': answer})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)