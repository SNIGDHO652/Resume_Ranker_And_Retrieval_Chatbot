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
    """Extract text from uploaded PDF using PyMuPDF (fitz)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    doc = fitz.open(tmp_path)
    pages = []
    for p in doc:
        pages.append(p.get_text())
    doc.close()
    return "\n".join(pages)

def extract_text_from_uploaded(uploaded_file) -> str:
    if uploaded_file.type == "text/plain" or uploaded_file.name.lower().endswith(".txt"):
        try:
            return uploaded_file.read().decode("utf-8")
        except Exception:
            return uploaded_file.getvalue().decode("utf-8")
    else:
        return extract_text_from_pdf(uploaded_file)

def extract_name(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:8]:
        if "@" in ln or any(ch.isdigit() for ch in ln):
            continue
        if 1 <= len(ln.split()) <= 5:
            low = ln.lower()
            if any(k in low for k in ["experience", "education", "skills", "contact", "profile", "summary"]):
                continue
            return ln
    return "Unknown Candidate"

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
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

def build_rag_index(resumes: List[Tuple[str, str]], chunk_size: int, chunk_overlap: int, top_k: int):
    chunks = []
    meta = []
    for fname, text in resumes:
        name = extract_name(text)
        pieces = chunk_text(text, chunk_size, chunk_overlap)
        for i, p in enumerate(pieces):
            chunks.append(p)
            meta.append({"candidate_name": name, "filename": fname, "chunk_id": f"{fname}__{i}"})
    if len(chunks) == 0:
        embeddings = np.zeros((0, model.get_sentence_embedding_dimension()))
        return {"embeddings": embeddings, "chunks": [], "meta": meta, "nn": None}
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    nn = NearestNeighbors(n_neighbors=min(top_k, len(embeddings)), metric="cosine")
    nn.fit(embeddings)
    return {"embeddings": embeddings, "chunks": chunks, "meta": meta, "nn": nn}

def retrieve(query: str, index: Dict, top_k: int = DEFAULT_TOP_K, candidate_filter: str = None):
    if index["nn"] is None or len(index["chunks"]) == 0:
        return []
    q_emb = model.encode([query], convert_to_numpy=True)
    k = min(top_k, len(index["chunks"]))
    distances, indices = index["nn"].kneighbors(q_emb, n_neighbors=k)
    distances = distances[0]
    indices = indices[0]
    results = []
    for dist, idx in zip(distances, indices):
        sim = 1 - float(dist)
        md = index["meta"][idx]
        cand = md["candidate_name"]
        if candidate_filter and candidate_filter.lower() not in cand.lower():
            continue
        results.append({"score": sim, "candidate_name": cand, "filename": md["filename"], "chunk_text": index["chunks"][idx]})
    if candidate_filter and len(results) < top_k:
        sims = cosine_similarity(q_emb, index["embeddings"])[0]
        sorted_idx = np.argsort(-sims)
        results = []
        for idx in sorted_idx:
            md = index["meta"][idx]
            cand = md["candidate_name"]
            if candidate_filter.lower() in cand.lower():
                results.append({"score": float(sims[idx]), "candidate_name": cand, "filename": md["filename"], "chunk_text": index["chunks"][idx]})
            if len(results) >= top_k:
                break
    return results

def get_ollama_client():
    if not USE_OLLAMA:
        return None
    try:
        client = OllamaClient() 
        return client
    except Exception:
        return None

@st.cache_resource
def cached_ollama_client():
    return get_ollama_client()

def generate_answer_ollama(question: str, retrieved: List[Dict], model_name: str = OLLAMA_MODEL) -> str:
    client = cached_ollama_client()
    if client is None:
        return "Ollama client not available. Ensure the Ollama daemon is running."

    if len(retrieved) == 0:
        return "No relevant information found in resumes."

    context_parts = []
    for r in retrieved:
        header = f"[Candidate: {r['candidate_name']} | File: {r['filename']} | score: {r['score']:.3f}]"
        context_parts.append(header + "\n" + r["chunk_text"])
    context = "\n\n---\n\n".join(context_parts)

    if len(context) > 3500:
        context = context[:3500] + "\n\n[TRUNCATED]"

    system = (
        "You are a helpful assistant that answers questions ABOUT CANDIDATES using ONLY the provided context. "
        "If context doesn't contain the answer, say 'I don't know'. Cite candidate names in the answer."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely."

    try:
        resp = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            options={   
                "num_predict": 512,
                "temperature": 0.1
            }
        )

        if isinstance(resp, dict):
            if "message" in resp and "content" in resp["message"]:
                return resp["message"]["content"]
            elif "choices" in resp and len(resp["choices"]) and "message" in resp["choices"][0]:
                return resp["choices"][0]["message"]["content"]
            else:
                return resp.get("message", {}).get("content", "")

        else:
            return resp.get("message", {}).get("content", "")

    except Exception as e:
        return f"Ollama error: {e}"


def generate_answer_extractive(question: str, retrieved: List[Dict]) -> str:
    if not retrieved:
        return "No relevant information found in resumes."
    by_candidate = {}
    for r in retrieved:
        by_candidate.setdefault(r["candidate_name"], []).append(r)
    parts = []
    parts.append("Extractive results (excerpts):\n")
    for cand, items in by_candidate.items():
        parts.append(f"--- {cand} ---")
        for i, it in enumerate(items, 1):
            snippet = it["chunk_text"]
            if len(snippet) > 700:
                snippet = snippet[:700] + " ... [truncated]"
            parts.append(f"{i}. (score: {it['score']:.3f}) {snippet}\n")
    parts.append("\nYou can ask follow-ups like 'What skills does <Candidate> have?' or 'Summarize <Candidate>'s experience.'")
    return "\n".join(parts)



from sklearn.metrics.pairwise import cosine_similarity  

st.set_page_config(page_title="Resume Ranker + Retrieval Chatbot", layout="wide")
st.title("📄 Resume Ranker + Retrieval Chatbot")

left, right = st.columns([1, 2])

with left:
    st.header("Uploads & Settings")
    jd_file = st.file_uploader("Job Description (PDF or TXT)", type=["pdf", "txt"])
    resume_files = st.file_uploader("Candidate Resumes (PDF) — multiple", type=["pdf"], accept_multiple_files=True)

    st.write("---")
    top_k = st.number_input("Retriever top_k", value=DEFAULT_TOP_K, min_value=1, max_value=20, step=1)
    chunk_size = st.number_input("Chunk size (chars)", value=DEFAULT_CHUNK_SIZE, min_value=100, max_value=2000, step=50)
    chunk_overlap = st.number_input("Chunk overlap (chars)", value=DEFAULT_CHUNK_OVERLAP, min_value=0, max_value=1000, step=10)

    st.write("---")
    st.markdown("**Local LLM (Ollama)**")
    if USE_OLLAMA:
        client = cached_ollama_client()
        if client is None:
            st.warning("`ollama` client not available or daemon not running. Make sure Ollama is installed and the daemon is running.")
        else:
            st.success(f"Ollama client OK. Using model `{OLLAMA_MODEL}` (pull with `ollama pull {OLLAMA_MODEL}`).")
    else:
        st.info("`ollama` package not installed — chat will use extractive fallback.")

    st.write("---")
    question = st.text_input("Ask a question about the resumes (e.g., 'Who has TensorFlow experience?', 'Tell me about Alice')")

    run_button = st.button("🔍 Rank Resumes & Build Index")

with right:
    st.header("Results")
    ranking_area = st.empty()
    chat_area = st.empty()

if run_button:
    if jd_file is None:
        st.error("Upload a Job Description (PDF/TXT) first.")
        st.stop()
    if not resume_files:
        st.error("Upload at least one resume (PDF).")
        st.stop()

    jd_text = extract_text_from_uploaded(jd_file)
    resumes = []
    for f in resume_files:
        txt = extract_text_from_uploaded(f)
        resumes.append((f.name, txt))

    jd_emb = model.encode([jd_text], convert_to_numpy=True)
    scored = []
    for fname, txt in resumes:
        emb = model.encode([txt], convert_to_numpy=True)
        sim = float(cosine_similarity(jd_emb, emb)[0][0])
        name = extract_name(txt)
        scored.append({"filename": fname, "name": name, "score": sim, "text": txt})
    scored = sorted(scored, key=lambda x: x["score"], reverse=True)

    with ranking_area.container():
        st.subheader("Top Ranked Candidates")
        for i, r in enumerate(scored, start=1):
            st.markdown(f"**{i}. {r['name']}** — `{r['filename']}` — Score: **{r['score']:.3f}**")
            with st.expander(f"Show resume: {r['filename']}"):
                st.write(r["text"][:20000])

    index = build_rag_index(resumes, int(chunk_size), int(chunk_overlap), int(top_k))
    st.session_state["rag_index"] = index
    st.session_state["resumes"] = resumes
    st.session_state["scored"] = scored
    st.session_state["jd_text"] = jd_text

    st.success("Index built. You can now ask questions in the chat box.")

if question and "rag_index" in st.session_state:
    index = st.session_state["rag_index"]
    candidate_names = list({m["candidate_name"] for m in index["meta"]}) if index["meta"] else []
    candidate_filter = None
    for cand in candidate_names:
        if cand.strip().lower() in question.strip().lower():
            candidate_filter = cand
            break
    retrieved = retrieve(question, index, top_k=int(top_k), candidate_filter=candidate_filter)

    answer = None
    if USE_OLLAMA:
        client = cached_ollama_client()
        if client:
            answer = generate_answer_ollama(question, retrieved)
        else:
            answer = generate_answer_extractive(question, retrieved)
    else:
        answer = generate_answer_extractive(question, retrieved)

    with chat_area.container():
        st.subheader("💬 Chatbot Answer")
        st.write(answer)
        st.markdown("**Retrieved passages (evidence)**")
        if not retrieved:
            st.write("No retrieved passages found.")
        else:
            for r in retrieved:
                st.markdown(f"- **{r['candidate_name']}** | `{r['filename']}` — score: {r['score']:.3f}")
                snippet = r["chunk_text"]
                if len(snippet) > 1200:
                    snippet = snippet[:1200] + " ... [truncated]"
                st.text(snippet)

elif question and "rag_index" not in st.session_state:
    st.warning("Build the index first by uploading files and clicking 'Rank Resumes & Build Index'.")

with st.expander("⚙️ Debug / Tips"):
    st.write(f"Embedding model: `{EMBED_MODEL_NAME}`")
    if USE_OLLAMA:
        st.write(f"Ollama model to use: `{OLLAMA_MODEL}` (pull with `ollama pull {OLLAMA_MODEL}`)")
    st.write("- For best responses include candidate name in your question.")
    st.write("- Ollama runs locally — no keys needed. If you see errors, ensure Ollama daemon is running and you pulled the model.")
