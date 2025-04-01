from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
Answer the question below as a researcher who is helping others in research tracking and productivity in their research work.

Here is the conversation history: {context}

Question: {question}

Answer:
"""
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_coversions():
    context = ""
    print("Welcome to the Research Assistant! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        result = chain.invoke({"context": context, "question": user_input})
        print("Assistant:", result)
        context += f"User: {user_input}\nAssistant: {result}\n"

if __name__ == "__main__":
    handle_coversions()