import tempfile
from typing import List, Dict, Tuple
import streamlit as st
import fitz
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    st.error("Missing `sentence-transformers`. Install: pip install sentence-transformers")
    raise

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:
    st.error("Missing `scikit-learn`. Install: pip install scikit-learn")
    raise

USE_OLLAMA = True
try:
    from ollama import Client as OllamaClient
except Exception:
    USE_OLLAMA = False


EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 5
OLLAMA_MODEL = "llama3.2:1b"

def extract_text_from_pdf(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    doc = fitz.open(tmp_path)
    pages = [p.get_text() for p in doc]
    doc.close()
    return "\n".join(pages)

def extract_text_from_uploaded(uploaded_file) -> str:
    if uploaded_file.type == "text/plain" or uploaded_file.name.lower().endswith(".txt"):
        try:
            return uploaded_file.read().decode("utf-8")
        except Exception:
            return uploaded_file.getvalue().decode("utf-8")
    return extract_text_from_pdf(uploaded_file)

def extract_name(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:8]:
        if "@" in ln or any(ch.isdigit() for ch in ln):
            continue
        if 1 <= len(ln.split()) <= 5:
            if any(k in ln.lower() for k in ["experience", "education", "skills", "contact", "profile", "summary"]):
                continue
            return ln
    return "Unknown Candidate"

def chunk_text(text: str, chunk_size: int, overlap: int):
    text = " ".join(text.split())
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= length:
            break
        start = end - overlap
    return chunks

@st.cache_resource
def load_embedding_model(name=EMBED_MODEL_NAME):
    return SentenceTransformer(name)

model = load_embedding_model()

def build_rag_index(resumes, chunk_size, chunk_overlap, top_k):
    chunks, meta = [], []
    for fname, text in resumes:
        name = extract_name(text)
        pieces = chunk_text(text, chunk_size, chunk_overlap)
        for i, p in enumerate(pieces):
            chunks.append(p)
            meta.append({"candidate_name": name, "filename": fname, "chunk_id": f"{fname}__{i}"})

    if not chunks:
        embeddings = np.zeros((0, model.get_sentence_embedding_dimension()))
        return {"embeddings": embeddings, "chunks": [], "meta": meta, "nn": None}

    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)

    nn = NearestNeighbors(n_neighbors=min(top_k, len(embeddings)), metric="cosine")
    nn.fit(embeddings)

    return {"embeddings": embeddings, "chunks": chunks, "meta": meta, "nn": nn}

def retrieve(query: str, index: Dict, top_k: int, candidate_filter: str = None):
    if index["nn"] is None:
        return []
    q_emb = model.encode([query], convert_to_numpy=True)
    k = min(top_k, len(index["chunks"]))
    distances, indices = index["nn"].kneighbors(q_emb, n_neighbors=k)
    distances, indices = distances[0], indices[0]

    results = []
    for dist, idx in zip(distances, indices):
        sim = 1 - float(dist)
        md = index["meta"][idx]
        cand = md["candidate_name"]
        if candidate_filter and candidate_filter.lower() not in cand.lower():
            continue
        results.append({
            "score": sim,
            "candidate_name": cand,
            "filename": md["filename"],
            "chunk_text": index["chunks"][idx]
        })


    if candidate_filter and len(results) < top_k:
        sims = cosine_similarity(q_emb, index["embeddings"])[0]
        sorted_idx = np.argsort(-sims)
        results = []
        for idx in sorted_idx:
            md = index["meta"][idx]
            cand = md["candidate_name"]
            if candidate_filter.lower() in cand.lower():
                results.append({
                    "score": float(sims[idx]),
                    "candidate_name": cand,
                    "filename": md["filename"],
                    "chunk_text": index["chunks"][idx]
                })
            if len(results) >= top_k:
                break

    return results

def get_ollama_client():
    if not USE_OLLAMA:
        return None
    try:
        return OllamaClient()
    except Exception:
        return None

@st.cache_resource
def cached_ollama_client():
    return get_ollama_client()

def generate_answer_ollama(question, retrieved, model_name=OLLAMA_MODEL):
    client = cached_ollama_client()
    if client is None:
        return "⚠️ Ollama not available. Please ensure daemon is running."

    if len(retrieved) == 0:
        return "No relevant information found."

    context_parts = []
    for r in retrieved:
        header = f"[Candidate: {r['candidate_name']} | File: {r['filename']} | score: {r['score']:.3f}]"
        context_parts.append(header + "\n" + r["chunk_text"])
    context = "\n\n---\n\n".join(context_parts)
    context = context[:3500]

    system = (
        "You are a helpful assistant that answers questions ABOUT CANDIDATES using ONLY the provided context. "
        "If context doesn't contain the answer, say 'I don't know'. Cite candidate names in the answer."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely."

    resp = client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        options={"num_predict": 512, "temperature": 0.1}
    )

    if isinstance(resp, dict) and "message" in resp and "content" in resp["message"]:
        return resp["message"]["content"]

    return resp.get("message", {}).get("content", "")

def generate_answer_extractive(question, retrieved):
    if not retrieved:
        return "No relevant information found."

    by_candidate = {}
    for r in retrieved:
        by_candidate.setdefault(r["candidate_name"], []).append(r)

    parts = ["Extractive results:\n"]
    for cand, items in by_candidate.items():
        parts.append(f"--- {cand} ---")
        for i, it in enumerate(items, 1):
            snippet = it["chunk_text"][:700] + "..." if len(it["chunk_text"]) > 700 else it["chunk_text"]
            parts.append(f"{i}. (score: {it['score']:.3f}) {snippet}")
    return "\n".join(parts)


st.set_page_config(page_title="AI Resume Ranker + RAG Chatbot", layout="wide")
st.title("📄 AI-Powered Resume Ranker & Retrieval Chatbot")

st.markdown("""
Upload a **Job Description** and one or more **Resumes**.  
The system will:
1. Rank candidates using semantic similarity  
2. Build a searchable RAG index  
3. Allow LLM-powered Q&A about candidates  
""")


with st.sidebar:
    st.header("⚙️ Settings")

    jd_file = st.file_uploader("📌 Job Description", type=["pdf", "txt"])
    resume_files = st.file_uploader("📎 Candidate Resumes", type=["pdf"], accept_multiple_files=True)

    st.markdown("### 🔧 Retrieval Settings")
    top_k = st.slider("Top K chunks", 1, 20, DEFAULT_TOP_K)
    chunk_size = st.slider("Chunk size", 200, 2000, DEFAULT_CHUNK_SIZE, step=50)
    chunk_overlap = st.slider("Chunk overlap", 0, 800, DEFAULT_CHUNK_OVERLAP, step=20)

    st.markdown("### 🤖 LLM Settings")
    if USE_OLLAMA:
        client = cached_ollama_client()
        if client:
            st.success(f"Ollama model: `{OLLAMA_MODEL}`")
        else:
            st.warning("Ollama not detected.")
    else:
        st.info("Ollama package not installed.")

    st.markdown("---")
    run_button = st.button("🚀 Process & Rank Candidates", use_container_width=True)


tab1, tab2 = st.tabs(["📊 Ranking", "💬 Chatbot"])

with tab1:
    if run_button:
        if jd_file is None:
            st.error("Upload a Job Description first.")
            st.stop()

        if not resume_files:
            st.error("Upload at least one resume.")
            st.stop()

        jd_text = extract_text_from_uploaded(jd_file)
        st.session_state["jd_text"] = jd_text

        resumes = []
        for f in resume_files:
            resumes.append((f.name, extract_text_from_uploaded(f)))

        jd_emb = model.encode([jd_text], convert_to_numpy=True)
        scored = []
        for fname, txt in resumes:
            emb = model.encode([txt], convert_to_numpy=True)
            sim = float(cosine_similarity(jd_emb, emb)[0][0])
            name = extract_name(txt)
            scored.append({"filename": fname, "name": name, "score": sim, "text": txt})

        scored = sorted(scored, key=lambda x: x["score"], reverse=True)

        st.session_state["resumes"] = resumes
        st.session_state["scored"] = scored

        st.subheader("🏆 Top Candidates")
        for i, r in enumerate(scored, 1):
            with st.expander(f"{i}. {r['name']} — {r['filename']}  | Score: {r['score']:.3f}"):
                st.write(r["text"][:20000])

        st.session_state["rag_index"] = build_rag_index(
            resumes, chunk_size, chunk_overlap, top_k
        )

        st.success("RAG index built successfully. Go to the **Chatbot** tab to ask questions.")


with tab2:
    st.header("Ask Questions About Candidates 👇")

    question = st.text_input("Ask something like: *Who has Kubernetes experience?*")

    if question:
        if "rag_index" not in st.session_state:
            st.warning("Please process resumes first.")
        else:
            index = st.session_state["rag_index"]

            candidate_filter = None
            for cand in {m["candidate_name"] for m in index["meta"]}:
                if cand.lower() in question.lower():
                    candidate_filter = cand
                    break

            retrieved = retrieve(question, index, top_k, candidate_filter)

            if USE_OLLAMA and cached_ollama_client():
                answer = generate_answer_ollama(question, retrieved)
            else:
                answer = generate_answer_extractive(question, retrieved)

            st.subheader("📌 Answer")
            st.write(answer)

            st.subheader("📚 Retrieved Evidence")
            for r in retrieved:
                snippet = r["chunk_text"][:900] + "..." if len(r["chunk_text"]) > 900 else r["chunk_text"]
                with st.expander(f"{r['candidate_name']} — {r['filename']}  | score: {r['score']:.3f}"):
                    st.text(snippet)
