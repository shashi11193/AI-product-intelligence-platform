import sys
import os
import time
import re
from datetime import datetime

# -----------------------------
# PATH SETUP
# -----------------------------
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
)

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from src.observability.logger import log_query

DB_PATH = "data/chroma_db"


# -----------------------------
# LOAD MODELS ONCE
# -----------------------------
print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading vector DB...")
db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

print("Loading LLM (Ollama - phi3:mini)...")
llm = Ollama(model="phi3:mini")

print("System ready 🚀")


# -----------------------------
# EVALUATION UTILITIES
# -----------------------------
def normalize(text):
    return re.sub(r"\s+", " ", text.lower())


def compute_faithfulness(answer, context_chunks):
    """
    Measures overlap between answer tokens and context tokens.
    Simple but effective for portfolio-level evaluation.
    """
    answer_tokens = set(normalize(answer).split())
    context_tokens = set(normalize(" ".join(context_chunks)).split())

    if not answer_tokens:
        return 0.0

    overlap = answer_tokens.intersection(context_tokens)

    return round(len(overlap) / len(answer_tokens), 2)


def detect_hallucination(answer, context_chunks):
    """
    Flags sentences in answer that do not appear in context.
    """
    context_text = normalize(" ".join(context_chunks))
    answer_text = normalize(answer)

    sentences = answer_text.split(".")

    hallucinated = []

    for s in sentences:
        s = s.strip()
        if len(s) < 5:
            continue

        if s not in context_text:
            hallucinated.append(s)

    return len(hallucinated) > 0, hallucinated


# -----------------------------
# QUERY ENGINE
# -----------------------------
def ask_question(question):

    total_start = time.time()

    # -------------------------
    # RETRIEVAL
    # -------------------------
    retrieval_start = time.time()

    docs = db.similarity_search(question, k=3)

    retrieval_time = time.time() - retrieval_start

    context_chunks = []
    context = ""

    print("\n--- Retrieved Context ---")

    for i, doc in enumerate(docs):
        chunk = doc.page_content

        context_chunks.append(chunk)

        print(f"\nChunk {i+1}:\n{chunk}")

        context += f"\n[CHUNK {i+1}]\n{chunk}\n"

    # -------------------------
    # PROMPT
    # -------------------------
    prompt = f"""
You are a strict retrieval-based AI assistant.

RULES:
1. Use ONLY the provided context.
2. If answer is not in context, say:
   "Not found in documents."
3. Be precise and concise.

CONTEXT:
{context}

QUESTION:
{question}

Return format:

ANSWER:
<final answer>

SOURCES:
- List chunk numbers used (Chunk 1, Chunk 2, etc.)
"""

    # -------------------------
    # GENERATION
    # -------------------------
    generation_start = time.time()

    response = llm.invoke(prompt)

    generation_time = time.time() - generation_start

    total_time = time.time() - total_start

    # -------------------------
    # EVALUATION LAYER (PHASE 4)
    # -------------------------
    faithfulness = compute_faithfulness(response, context_chunks)

    hallucination_flag, hallucinated_parts = detect_hallucination(
        response,
        context_chunks
    )

    # -------------------------
    # LOGGING
    # -------------------------
    log_query(
        timestamp=datetime.now(),
        question=question,
        answer=response,
        latency=total_time,
        chunks_retrieved=len(docs)
    )

    # -------------------------
    # OUTPUT METRICS
    # -------------------------
    print("\n--- FINAL ANSWER ---")
    print(response)

    print("\n--- PERFORMANCE METRICS ---")
    print(f"Retrieval Time: {retrieval_time:.2f} sec")
    print(f"Generation Time: {generation_time:.2f} sec")
    print(f"Total Time: {total_time:.2f} sec")

    print("\n--- EVALUATION METRICS ---")
    print(f"Faithfulness Score: {faithfulness}")

    if hallucination_flag:
        print("⚠️ Hallucination Detected")
        print("Suspicious Parts:")
        for h in hallucinated_parts[:3]:
            print(f"- {h}")
    else:
        print("No hallucination detected")

    print("\n--- SOURCES USED ---")
    for i in range(len(docs)):
        print(f"- Chunk {i+1}")

    return response


# -----------------------------
# CLI LOOP
# -----------------------------
if __name__ == "__main__":

    while True:
        q = input("\nAsk a question (or type exit): ")

        if q.lower() == "exit":
            break

        ask_question(q)