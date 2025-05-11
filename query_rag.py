import google.generativeai as genai
from retriever import load_and_split_docs, build_faiss_index, load_faiss_index
from dotenv import load_dotenv
import os
from tools import get_definition
import json

load_dotenv()
api_key = st.secrets["GEMINI_API_KEY"]
os.environ["GEMINI_API_KEY"] = api_key  

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.0-flash")

index_filename = "faiss_index.index"


import json

def classify_query(query):
    routing_prompt = f"""
You are a smart routing assistant.

Given a user question, classify it into one of the following categories:
1. definition — if the user wants to know the meaning of a term.
2. calculation — if the user wants to calculate something (math-related).
3. rag — if the question requires searching through research papers or broader explanation.

If the category is:
- **definition**: set "target" as the term to define, and "programmatic_expression" as null.
- **calculation**: set "target" as the natural language math query, and "programmatic_expression" as a Python-evaluable string (e.g., "4 - (1 + 1)").
- **rag**: leave both "target" and "programmatic_expression" as null.

Respond ONLY in JSON format:
{{
  "category": "...",
  "target": "...",
  "programmatic_expression": "..."
}}

Question: "{query}"
"""

    response = model.generate_content(routing_prompt)
    
    cleaned_response = response.text.strip('`').strip()
    if cleaned_response.startswith("json"):
        cleaned_response = cleaned_response[4:].strip()
    try:
        parsed = json.loads(cleaned_response)
        try:
            target = parsed.get("target", "").strip()
        except:
            target = parsed.get("target", "")
        try:
            prg_exp = parsed.get("programmatic_expression", "").strip()
        except:
            prg_exp = parsed.get("programmatic_expression", "")


        return (
                parsed.get("category", "rag"),
                target,
                prg_exp
                )
    except Exception as e:
        return "rag", ""



def rag_query(vectordb, query):
    # Retrieving top 3 chunks
    docs = vectordb.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""You are an academic assistant helping explain AI research papers.
Use the following context to answer the question.

Context:
{context}

Question: {query}
Answer:"""

    # Generate the answer using Gemini
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    try:
        vectordb = load_faiss_index(index_filename)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        chunks = load_and_split_docs()
        vectordb = build_faiss_index(chunks)

    print("Welcome to the Research Paper Q&A Assistant!")
    print("Ask a question (or type 'exit' to quit):")

    while True:
        query = input(">> ").strip()
        if query.lower() == "exit":
            break

        category, target, program_expr = classify_query(query)
        print(f"[Agent] Category: {category} | Target: {target}")

        if category == "definition":
            print("[Agent] Routing to Dictionary Tool")
            print(get_definition(target))

        elif category == "calculation":
            print("[Agent] Routing to Calculator")
            try:
                result = eval(program_expr)
                print("Result:", result)
            except Exception as e:
                print("Could not compute the result:", e)


        else:
            print("[Agent] Routing to RAG Pipeline")
            answer = rag_query(vectordb, query)
            print("Answer:", answer)

        print("-" * 50)