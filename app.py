import streamlit as st
import os, json
from google import genai
from google.genai import types
import chromadb

st.set_page_config(page_title="CiteMed-RAG", layout="wide")
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.0-flash"

api_key = st.sidebar.text_input("Gemini API Key", type="password")
if not api_key:
    st.info("Enter your Gemini API key in the sidebar.")
    st.stop()
os.environ["GEMINI_API_KEY"] = api_key
gclient = genai.Client()

@st.cache_data
def load_kb():
    p = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
    with open(p, encoding="utf-8") as f: return json.load(f)["documents"]

@st.cache_resource
def build_vs(_key):
    docs = load_kb()
    chunks = []
    for doc in docs:
        words = doc["text"].split()
        s, ci = 0, 0
        while s < len(words):
            e = s + 400
            cid = doc["id"] + "_c" + str(ci).zfill(2)
            chunks.append({"id": cid, "title": doc["title"], "source": doc["source"], "text": " ".join(words[s:e])})
            ci += 1
            if e >= len(words): break
            s = e - 100
    gc = genai.Client()
    embs = []
    for i in range(0, len(chunks), 20):
        b = chunks[i:i+20]
        r = gc.models.embed_content(model=EMBEDDING_MODEL, contents=[c["text"] for c in b])
        embs.extend([em.values for em in r.embeddings])
    ch = chromadb.Client()
    try: ch.delete_collection("citemed_kb")
    except: pass
    col = ch.create_collection("citemed_kb", metadata={"hnsw:space":"cosine"})
    for i, c in enumerate(chunks):
        col.add(ids=[c["id"]], embeddings=[embs[i]], documents=[c["text"]], metadatas=[{"doc_title":c["title"],"source":c["source"]}])
    return col

with st.spinner("Building knowledge base..."):
    collection = build_vs(api_key)

A1_P = "You are CiteMed-Prompt, bilingual health education assistant. No diagnosis/prescribing/dosing. Answer in user language. Format: 1) Brief Answer 2) Key Points 3) Safety 4) When to Seek Care"
A2_T = "You are CiteMed-RAG, health assistant with knowledge base. No diagnosis/prescribing/dosing. Base answer on CONTEXT. Cite [1],[2]. If no info say so.\nCONTEXT:\n{ctx}\nSOURCES:\n{src}"

def retr(q, k=5):
    em = gclient.models.embed_content(model=EMBEDDING_MODEL, contents=[q])
    r = collection.query(query_embeddings=[em.embeddings[0].values], n_results=k, include=["documents","metadatas","distances"])
    out = []
    for i in range(len(r["ids"][0])):
        out.append({"text":r["documents"][0][i],"title":r["metadatas"][0][i]["doc_title"],"source":r["metadatas"][0][i]["source"],"dist":r["distances"][0][i]})
    return out

def gen(q, sp):
    cfg = types.GenerateContentConfig(system_instruction=sp, temperature=0.3, max_output_tokens=1000, seed=42)
    return gclient.models.generate_content(model=GENERATION_MODEL, contents=q, config=cfg).text or ""

st.title("CiteMed-RAG")
st.markdown("Bilingual patient-education assistant with medical knowledge base")
mode = st.radio("Mode", ["A2: RAG Enhanced","A1: Baseline"], horizontal=True)
question = st.text_area("Health question (English or Arabic):", height=100)

if st.button("Get Answer", type="primary") and question:
    with st.spinner("Thinking..."):
        if mode.startswith("A2"):
            cks = retr(question)
            ctx = chr(10).join("[" + str(i+1) + "] " + c["text"] for i,c in enumerate(cks))
            src = chr(10).join("[" + str(i+1) + "] " + c["title"] + " -- " + c["source"] for i,c in enumerate(cks))
            answer = gen(question, A2_T.replace("{ctx}",ctx).replace("{src}",src))
            st.markdown("### Answer")
            st.markdown(answer)
            with st.expander("Retrieved Sources"):
                for i,c in enumerate(cks):
                    st.markdown("**[" + str(i+1) + "] " + c["title"] + "** (" + c["source"] + ")") 
                    st.caption("Relevance: " + str(round((1-c["dist"])*100,1)) + "%")
                    st.text(c["text"][:200] + "...")
        else:
            st.markdown("### Answer")
            st.markdown(gen(question, A1_P))

st.sidebar.markdown("---")
st.sidebar.markdown("CiteMed-RAG v2.0")
st.sidebar.markdown("Assignment 2 -- Healthcare RAG")
