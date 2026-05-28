import os
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
import time

# load environment variables from .env file
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Loading and cleaning FAQ data...")
# use pandas to read the CSV file containing the FAQ data
df = pd.read_csv("faq.csv", encoding="utf-8-sig", skipinitialspace=True)

# cleaning data
df["question"] = df["question"].astype(str).str.strip()
df["answer"] = df["answer"].astype(str).str.strip()

# convert questions to a list for embedding
questions = df["question"].tolist()

#load model BGE-M3
print("Loading BGE-M3 model...")
model = SentenceTransformer("BAAI/bge-m3")

#create embeddings for the questions in the FAQ dataset
print("Creating embeddings for Greenwich FAQ questions...")
embeddings = model.encode(
    questions, 
    normalize_embeddings=True, 
    show_progress_bar=True
)
embeddings = np.array(embeddings).astype("float32")

#create FAISS index for efficient similarity search
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  #use inner product for similarity search
index.add(embeddings)

print(f"Successfully loaded and indexed {len(questions)} Greenwich FAQ rows.")

#define a similarity threshold for determining if a user query matches an FAQ question
THRESHOLD = 0.72

#LLM CALL FUNCTION
def call_llm(user_query):
    """Function to handle questions outside the scope of the FAQ data file (Automatic retry mechanism added in case of overload)"""
    prompt = f"""
    You are a helpful assistant for Greenwich Vietnam students. 
    Answer the user's question clearly, politely, and concisely.

    User Question: {user_query}
    """
    
    max_retries = 3  #maximum number of retry attempts in case of server overload
    delay = 2        #waiting time (seconds) between retries

    for attempt in range(max_retries):
        try:
            #call the Gemini LLM to generate a response based on the user query
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text
        except Exception as e:
            # Check if the error is due to server overload (503) or rate limit exceeded (429)
            if "503" in str(e) or "429" in str(e):
                if attempt < max_retries - 1:
                    print(f"\n[WARNING] Google server is busy (Attempt {attempt + 1}/{max_retries}). Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Double the wait time for the next retry (Exponential Backoff)
                    continue
            
            # If it's another error (or all retries are exhausted), return the error message
            return f"Error when connecting to Gemini LLM: {str(e)}"

# ====================================
# SEARCH & ROUTE FUNCTION
# ====================================
def get_bot_response(query):
    # Check if the index is empty (to prevent issues with a corrupted or empty CSV file)
    if index.ntotal == 0:
        return call_llm(query)

    #create embedding for the user query
    query_embedding = model.encode([query.strip()], normalize_embeddings=True)
    query_embedding = np.array(query_embedding).astype("float32")

    #search for the most similar question in the FAQ dataset using FAISS
    scores, indices = index.search(query_embedding, 1)

    best_score = float(scores[0][0])
    best_idx = int(indices[0][0])

    print(f"\n[DEBUG] Most similar question: '{df.iloc[best_idx]['question']}'")
    print(f"[DEBUG] Similarity score found: {best_score:.3f}")

    # 3. Route the query based on the similarity score
    if best_score >= THRESHOLD:
        # Case 1: Similar question found: retrieve the standard answer from the CSV file
        print("[DEBUG] Route: FAQ match -> Retrieve answer from CSV file")
        return df.iloc[best_idx]["answer"]
    else:
        # Case 2: No similar question found: route to Gemini LLM
        print("[DEBUG] Route: No FAQ match -> Forward to Gemini LLM")
        return call_llm(query)

# ====================================
# CHAT LOOP
# ====================================
if __name__ == "__main__":
    print("\n=== Greenwich Hybrid RAG Chatbot ===")
    print("Type 'exit' to quit the chat\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        
        if not query.strip():
            continue

        reply = get_bot_response(query)
    
        print(f"\nBot:\n{reply}")
        print("=" * 50)