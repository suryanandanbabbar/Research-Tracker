from flask import Flask, render_template, request, jsonify
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

template = """
Answer the question below as a researcher who is helping others in research tracking and productivity in their research work.

Here is the conversation history: {context}

Question: {question}

Answer:
"""
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Conversation memory
conversation_history = ""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    global conversation_history
    user_input = request.json.get("question", "")
    if not user_input:
        return jsonify({"error": "Empty input"}), 400

    result = chain.invoke({"context": conversation_history, "question": user_input})
    conversation_history += f"User: {user_input}\nAssistant: {result}\n"
    return jsonify({"response": result})

if __name__ == "__main__":
    app.run(debug=True)
