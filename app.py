import json
import re
import tempfile
from typing import List, Dict, Tuple, Optional
import streamlit as st
import fitz
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    st.error(
        "Missing `sentence-transformers`. Install: pip install sentence-transformers"
    )
    raise

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    st.error("Missing `scikit-learn`. Install: pip install scikit-learn")
    raise


USE_OLLAMA = True
try:
    from ollama import Client as OllamaClient
except Exception:
    USE_OLLAMA = False


EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_TOKENS = 220
DEFAULT_CHUNK_OVERLAP_TOKENS = 40
DEFAULT_TOP_K = 5
OLLAMA_MODEL = "llama3.2:1b"
LLM_ATS_TOP_K = 6
LLM_CHAT_TOP_K = 5


def extract_text_from_pdf(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    doc = fitz.open(tmp_path)
    pages = [p.get_text() for p in doc]
    doc.close()
    return "\n".join(pages)


def extract_text_from_uploaded(uploaded_file) -> str:
    if uploaded_file.type == "text/plain" or uploaded_file.name.lower().endswith(
        ".txt"
    ):
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
        if 1 <= len(ln.split()) <= 6:
            low = ln.lower()
            if any(
                k in low
                for k in [
                    "experience",
                    "education",
                    "skills",
                    "contact",
                    "profile",
                    "summary",
                ]
            ):
                continue
            return ln
    return "Unknown Candidate"


def token_chunk_text(
    text: str,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> List[str]:
    """
    Approximate token-based chunking by splitting on whitespace (words).
    chunk_tokens = number of words per chunk
    overlap_tokens = number of words overlapping between chunks
    """
    tokens = text.split()
    if len(tokens) <= chunk_tokens:
        return [" ".join(tokens)]
    chunks = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + chunk_tokens, n)
        chunk = tokens[start:end]
        chunks.append(" ".join(chunk))
        if end >= n:
            break
        start = max(0, end - overlap_tokens)
    return chunks


@st.cache_resource(show_spinner=False)
def load_embedding_model(name=EMBED_MODEL_NAME):
    return SentenceTransformer(name)


model = load_embedding_model()


def build_rag_index(
    resumes: List[Tuple[str, str]], chunk_tokens: int, overlap_tokens: int, top_k: int
):
    chunks = []
    meta = []
    for fname, text in resumes:
        name = extract_name(text)
        pieces = token_chunk_text(text, chunk_tokens, overlap_tokens)
        for i, p in enumerate(pieces):
            chunks.append(p)
            meta.append(
                {"candidate_name": name, "filename": fname, "chunk_id": f"{fname}__{i}"}
            )
    if not chunks:
        return {
            "embeddings": np.zeros((0, model.get_sentence_embedding_dimension())),
            "chunks": [],
            "meta": meta,
            "nn": None,
        }
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    nn = NearestNeighbors(n_neighbors=min(top_k, len(embeddings)), metric="cosine")
    nn.fit(embeddings)
    return {"embeddings": embeddings, "chunks": chunks, "meta": meta, "nn": nn}


def retrieve_topk(
    query: str, index: Dict, top_k: int = 5, candidate_filter: Optional[str] = None
):
    if index["nn"] is None or len(index["chunks"]) == 0:
        return []
    q_emb = model.encode([query], convert_to_numpy=True)
    k = min(top_k, len(index["chunks"]))
    dists, idxs = index["nn"].kneighbors(q_emb, n_neighbors=k)
    dists = dists[0]
    idxs = idxs[0]
    results = []
    for dist, idx in zip(dists, idxs):
        sim = 1 - float(dist)
        md = index["meta"][idx]
        cand = md["candidate_name"]
        if candidate_filter and candidate_filter.lower() not in cand.lower():
            continue
        results.append(
            {
                "score": sim,
                "candidate_name": cand,
                "filename": md["filename"],
                "chunk_text": index["chunks"][idx],
            }
        )
    if candidate_filter and len(results) < top_k:
        sims = cosine_similarity(q_emb, index["embeddings"])[0]
        sorted_idx = np.argsort(-sims)
        results = []
        for idx in sorted_idx:
            md = index["meta"][idx]
            cand = md["candidate_name"]
            if candidate_filter.lower() in cand.lower():
                results.append(
                    {
                        "score": float(sims[idx]),
                        "candidate_name": cand,
                        "filename": md["filename"],
                        "chunk_text": index["chunks"][idx],
                    }
                )
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


@st.cache_resource(show_spinner=False)
def cached_ollama_client():
    return get_ollama_client()


def extract_ollama_content(resp) -> Optional[str]:
    if resp is None:
        return None
    if isinstance(resp, dict):
        if (
            "message" in resp
            and isinstance(resp["message"], dict)
            and "content" in resp["message"]
        ):
            return resp["message"]["content"]
        if "choices" in resp and resp["choices"]:
            c = resp["choices"][0]
            if (
                isinstance(c, dict)
                and "message" in c
                and isinstance(c["message"], dict)
                and "content" in c["message"]
            ):
                return c["message"]["content"]
    try:
        return str(resp.get("message", {}).get("content", ""))
    except Exception:
        return None


ACTION_VERBS = {
    "led",
    "designed",
    "implemented",
    "developed",
    "built",
    "improved",
    "reduced",
    "increased",
    "optimized",
    "created",
    "managed",
    "launched",
    "spearheaded",
    "organized",
}


def heuristic_ats(resume_text: str, jd_text: str) -> Dict:
    words = re.findall(r"[A-Za-z0-9\+#\.\-]+", resume_text.lower())
    num_words = len(words)
    bullets = len(re.findall(r"^\s*[\-\•\*\u2022]", resume_text, flags=re.M))
    has_email = bool(
        re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", resume_text)
    )
    has_phone = bool(
        re.search(
            r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4,6}", resume_text
        )
    )
    has_table = bool(re.search(r"\|.+\|", resume_text))
    jd_tokens = set(re.findall(r"\b\w+\b", jd_text.lower()))
    resume_tokens = set(words)
    keyword_overlap = len(jd_tokens & resume_tokens) / max(1, len(jd_tokens))
    verb_hits = sum(
        1
        for v in ACTION_VERBS
        if re.search(r"\b" + re.escape(v) + r"\b", resume_text, re.I)
    )
    issues = []
    recs = []
    if not has_email:
        issues.append("Missing email")
        recs.append("Add a professional email.")
    if not has_phone:
        issues.append("Missing phone")
        recs.append("Add a contact number.")
    if has_table:
        issues.append("Table-like content (may break ATS)")
        recs.append("Replace tables with bullets.")
    if num_words < 150:
        issues.append("Very short resume")
        recs.append("Add detailed responsibilities and metrics.")
    if verb_hits < 2:
        recs.append("Use more action verbs (led, implemented, improved).")
    score = 100
    score -= int((1 - keyword_overlap) * 40)
    if not has_email:
        score -= 10
    if not has_phone:
        score -= 8
    if has_table:
        score -= 10
    if num_words < 150:
        score -= 10
    score = max(0, min(100, score))
    section_scores = {
        "skills": round(100.0 * keyword_overlap, 2),
        "experience": round(min(100.0, verb_hits * 10.0), 2),
        "education": 0.0,
        "projects": 0.0,
        "summary": 0.0,
    }
    return {
        "ats_score": score,
        "issues": issues,
        "recommendations": recs,
        "section_scores": section_scores,
    }


def llm_evaluate_ats(
    candidate_name: str,
    resume_text: str,
    jd_text: str,
    rag_index: Dict,
    top_k: int = LLM_ATS_TOP_K,
) -> Optional[Dict]:
    client = cached_ollama_client()
    if client is None:
        return None

    retrieved = retrieve_topk(
        jd_text, rag_index, top_k=top_k, candidate_filter=candidate_name
    )
    if not retrieved:
        context_text = resume_text[:3000]
    else:
        parts = []
        for r in retrieved:
            parts.append(
                f"[{r['candidate_name']} | {r['filename']} | {r['score']:.3f}]\n{r['chunk_text']}"
            )
        context_text = "\n\n---\n\n".join(parts)
        if len(context_text) > 3500:
            context_text = context_text[:3500] + "\n\n[TRUNCATED]"

    system = (
        "You are an assistant that evaluates a resume against a job description for ATS compatibility and section-wise match. "
        "Output ONLY valid JSON with keys: "
        "ats_score (0-100), issues (list), recommendations (list), "
        "section_scores (object with skills, experience, education, projects, summary each 0-100), "
        "short_summary (1-3 sentence). No other text."
    )

    user_prompt = f"Job Description:\n{jd_text[:3000]}\n\nResume context for {candidate_name}:\n{context_text}\n\nTask: produce the JSON."

    try:
        resp = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            options={"num_predict": 512, "temperature": 0.0},
        )
        content = extract_ollama_content(resp)
        if not content:
            return None

        content_clean = content.strip()
        content_clean = re.sub(r"^```json\s*|\s*```$", "", content_clean, flags=re.I)
        first = content_clean.find("{")
        last = content_clean.rfind("}")
        if first != -1 and last != -1:
            content_clean = content_clean[first : last + 1]
        parsed = json.loads(content_clean)

        parsed["ats_score"] = float(parsed.get("ats_score", 0.0))
        ss = parsed.get("section_scores", {})
        for k in ["skills", "experience", "education", "projects", "summary"]:
            ss.setdefault(k, 0.0)
        parsed["section_scores"] = {
            k: float(ss[k])
            for k in ["skills", "experience", "education", "projects", "summary"]
        }
        parsed["issues"] = parsed.get("issues", []) or []
        parsed["recommendations"] = parsed.get("recommendations", []) or []
        parsed["short_summary"] = str(parsed.get("short_summary", ""))
        return parsed
    except Exception:
        return None


def llm_generate_answer(
    question: str, retrieved: List[Dict], model_name: str = OLLAMA_MODEL
) -> str:
    client = cached_ollama_client()
    if client is None:
        return "LLM not available."
    if not retrieved:
        return "No relevant context found."
    context_parts = []
    for r in retrieved:
        context_parts.append(
            f"[{r['candidate_name']} | {r['filename']} | {r['score']:.3f}]\n{r['chunk_text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    if len(context) > 3500:
        context = context[:3500] + "\n\n[TRUNCATED]"
    system = "You are an assistant that answers questions about candidates using only the context. If not present, say 'I don't know'."
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely and cite candidate names."
    try:
        resp = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            options={"num_predict": 512, "temperature": 0.0},
        )
        return extract_ollama_content(resp) or "No answer"
    except Exception as e:
        return f"LLM error: {e}"


def extractive_answer(question: str, retrieved: List[Dict]) -> str:
    if not retrieved:
        return "No retrieved passages."
    lines = []
    for r in retrieved:
        snippet = r["chunk_text"][:400].replace("\n", " ")
        lines.append(f"{r['candidate_name']} ({r['score']:.3f}): {snippet}")
    return "Extractive results:\n" + "\n".join(lines)


st.set_page_config(page_title="Resume Ranker_Insight Chatbot_ATS", layout="wide")
st.title("📄 Resume Ranker + RAG Chatbot + ATS Insights")

page = st.sidebar.radio(
    "Navigate", ["Upload", "Ranking", "Chatbot", "ATS Insights"], index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Core Settings")
chunk_tokens = st.sidebar.slider(
    "Chunk tokens (approx words)", 100, 600, DEFAULT_CHUNK_TOKENS, step=20
)
overlap_tokens = st.sidebar.slider(
    "Chunk overlap (tokens)", 0, 200, DEFAULT_CHUNK_OVERLAP_TOKENS, step=10
)
top_k_chat = st.sidebar.slider("Retriever top_k (chat)", 1, 12, DEFAULT_TOP_K)
use_llm_for_ats = st.sidebar.checkbox("Use Ollama for ATS (if available)", value=True)
show_debug = st.sidebar.checkbox("Show debug info", value=False)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: token-based chunking approximates LLM token windows. Increase chunk size if LLM supports larger context."
)

if page == "Upload":
    st.header("Upload Job Description & Resumes")
    st.info(
        "Upload a Job Description (PDF/TXT) and multiple resumes (PDF). Then go to 'Ranking' to process."
    )
    jd_file = st.file_uploader(
        "Job Description (PDF or TXT)", type=["pdf", "txt"], key="jd_upload"
    )
    resume_files = st.file_uploader(
        "Candidate Resumes (PDF) — multiple",
        type=["pdf"],
        accept_multiple_files=True,
        key="res_upload",
    )
    if st.button("Save uploads to session"):
        if jd_file is None:
            st.error("Please upload a Job Description.")
        elif not resume_files:
            st.error("Please upload at least one resume.")
        else:
            jd_text = extract_text_from_uploaded(jd_file)
            resumes = []
            for f in resume_files:
                txt = extract_text_from_uploaded(f)
                resumes.append((f.name, txt))
            st.session_state["jd_text"] = jd_text
            st.session_state["resumes"] = resumes
            st.success(
                f"Saved {len(resumes)} resumes and JD to session. Now go to 'Ranking'."
            )

elif page == "Ranking":
    st.header("Ranking — Semantic match & Build RAG index")
    if "jd_text" not in st.session_state or "resumes" not in st.session_state:
        st.info(
            "No uploads found in session. Please upload files on the 'Upload' page and click Save."
        )
    else:
        jd_text = st.session_state["jd_text"]
        resumes = st.session_state["resumes"]

        with st.expander("Job Description (preview)"):
            st.write(jd_text[:4000])

        if st.button("Run ranking & build index"):
            with st.spinner("Computing embeddings & ranking..."):
                jd_emb = model.encode([jd_text], convert_to_numpy=True)
                scored = []
                for fname, txt in resumes:
                    emb = model.encode([txt], convert_to_numpy=True)
                    sem_sim = float(cosine_similarity(jd_emb, emb)[0][0])
                    name = extract_name(txt)
                    scored.append(
                        {
                            "filename": fname,
                            "name": name,
                            "semantic_score": sem_sim,
                            "text": txt,
                        }
                    )
                st.session_state["semantic_scored"] = sorted(
                    scored, key=lambda x: x["semantic_score"], reverse=True
                )
                rag_index = build_rag_index(
                    resumes,
                    chunk_tokens,
                    overlap_tokens,
                    max(LLM_ATS_TOP_K, top_k_chat, DEFAULT_TOP_K),
                )
                st.session_state["rag_index"] = rag_index
            st.success(
                "Ranking & RAG index built. Move to 'ATS Insights' to evaluate or 'Chatbot' to ask questions."
            )

        if "semantic_scored" in st.session_state:
            st.subheader("Top semantic matches")
            for i, r in enumerate(st.session_state["semantic_scored"], start=1):
                cols = st.columns([1, 1, 4])
                cols[0].write(f"**{i}**")
                cols[1].metric("Score", f"{r['semantic_score']:.3f}")
                cols[2].markdown(f"**{r['name']}** — `{r['filename']}`")
                with st.expander("Preview / actions"):
                    st.write(r["text"][:3000])
                    if st.button(
                        f"View ATS (quick) — {r['filename']}", key=f"quick_ats_{i}"
                    ):
                        st.session_state["_quick_preview"] = r

            if "_quick_preview" in st.session_state:
                r = st.session_state["_quick_preview"]
                st.markdown("### Quick Preview")
                st.code(r["text"][:2000])

elif page == "Chatbot":
    st.header("RAG Chatbot — Ask about candidates")
    if "rag_index" not in st.session_state:
        st.info("Build RAG index first on the 'Ranking' page.")
    else:
        rag_index = st.session_state["rag_index"]
        jd_text = st.session_state.get("jd_text", "")
        names = (
            sorted(list({m["candidate_name"] for m in rag_index["meta"]}))
            if rag_index["meta"]
            else []
        )
        candidate_choice = st.selectbox(
            "Filter to candidate (optional)", ["All candidates"] + names
        )
        question = st.text_input(
            "Enter your question (e.g., 'Who has Docker experience?')"
        )
        if st.button("Ask"):
            if not question.strip():
                st.warning("Enter a question.")
            else:
                candidate_filter = (
                    None if candidate_choice == "All candidates" else candidate_choice
                )
                retrieved = retrieve_topk(
                    question,
                    rag_index,
                    top_k=top_k_chat,
                    candidate_filter=candidate_filter,
                )
                client = cached_ollama_client() if USE_OLLAMA else None
                if client:
                    answer = llm_generate_answer(question, retrieved)
                else:
                    answer = extractive_answer(question, retrieved)
                st.markdown("### Answer")
                st.write(answer)
                st.markdown("### Retrieved Evidence")
                for r in retrieved:
                    with st.expander(
                        f"{r['candidate_name']} — {r['filename']} | score {r['score']:.3f}"
                    ):
                        st.text(r["chunk_text"])

elif page == "ATS Insights":
    st.header("ATS Insights — LLM-based (preferred) or heuristic fallback")
    if "resumes" not in st.session_state or "jd_text" not in st.session_state:
        st.info(
            "Upload and save resumes on 'Upload' and build index on 'Ranking' first."
        )
    else:
        jd_text = st.session_state["jd_text"]
        resumes = st.session_state["resumes"]
        rag_index = st.session_state.get("rag_index", None)
        if st.button("Run ATS evaluation for all resumes (LLM if enabled)"):
            analyses = []
            client_ok = (
                use_llm_for_ats and USE_OLLAMA and (cached_ollama_client() is not None)
            )
            with st.spinner("Evaluating... This may take a few seconds per resume"):
                for fname, txt in resumes:
                    name = extract_name(txt)
                    llm_result = None
                    if client_ok and rag_index:
                        llm_result = llm_evaluate_ats(
                            name, txt, jd_text, rag_index, top_k=LLM_ATS_TOP_K
                        )
                    if llm_result:
                        analyses.append(
                            {
                                "filename": fname,
                                "name": name,
                                "ats_score": round(llm_result.get("ats_score", 0.0), 2),
                                "issues": llm_result.get("issues", []),
                                "recommendations": llm_result.get(
                                    "recommendations", []
                                ),
                                "section_scores": llm_result.get("section_scores", {}),
                                "short_summary": llm_result.get("short_summary", ""),
                                "method": "llm",
                            }
                        )
                    else:
                        h = heuristic_ats(txt, jd_text)
                        analyses.append(
                            {
                                "filename": fname,
                                "name": name,
                                "ats_score": h["ats_score"],
                                "issues": h["issues"],
                                "recommendations": h["recommendations"],
                                "section_scores": h["section_scores"],
                                "short_summary": "",
                                "method": "heuristic",
                            }
                        )
            st.session_state["analyses"] = sorted(
                analyses, key=lambda x: x["ats_score"], reverse=True
            )
            st.success("ATS evaluation complete. Scroll below for results.")

        if "analyses" in st.session_state:
            st.subheader("ATS Results")
            for i, a in enumerate(st.session_state["analyses"], start=1):
                card_cols = st.columns([1, 1, 2])
                card_cols[0].metric("Rank", f"{i}")
                card_cols[1].metric("Method", a["method"])
                card_cols[2].markdown(f"**{a['name']}** — `{a['filename']}`")
                with st.expander("Details & Recommendations"):
                    st.markdown("**Issues**")
                    if a.get("issues"):
                        for it in a["issues"]:
                            st.write(f"- {it}")
                    else:
                        st.write("No major issues detected.")
                    st.markdown("**Recommendations**")
                    if a.get("recommendations"):
                        for rec in a["recommendations"]:
                            st.write(f"- {rec}")
                    else:
                        st.write("No recommendations.")
                    if a.get("short_summary"):
                        st.markdown("**LLM Short summary**")
                        st.write(a["short_summary"][:800])
                    with st.expander("Preview resume (first 2000 chars)"):
                        txt = next(
                            (t for (fn, t) in resumes if fn == a["filename"]), ""
                        )
                        st.code(txt[:2000])
        else:
            st.info("No ATS results yet. Click 'Run ATS evaluation' to start.")

if show_debug:
    st.sidebar.markdown("## Debug Info")
    st.sidebar.write(
        {
            "ollama_installed": USE_OLLAMA,
            "ollama_client": bool(cached_ollama_client()) if USE_OLLAMA else False,
            "session_keys": list(st.session_state.keys()),
        }
    )

st.markdown("---")
st.caption(
    "Built with Streamlit • Embeddings: sentence-transformers • Optional LLM: Ollama (local)."
)

