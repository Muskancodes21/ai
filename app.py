import os
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline

# ------ Config ---------
FAISS_PATH = "multi_format_faiss_index"
MODEL_ID = "google/flan-t5-base"
DEVICE = "cpu"
CHUNKS_TO_USE = 4
MAX_ANSWER_LENGTH = 512
# -----------------------

st.set_page_config(page_title="Semantic Search Engine", page_icon="🔎")

st.title("🔎 Semantic Search Engine (LLM Summary)")

@st.cache_resource(show_spinner="Loading vector DB and LLM…")
def load_resources():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": False}
    )
    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    gen_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if DEVICE == "cuda" else -1,
        max_length=MAX_ANSWER_LENGTH,
    )
    llm = HuggingFacePipeline(pipeline=gen_pipeline)
    return vectorstore, llm

def get_top_chunks(vectorstore, query, k=CHUNKS_TO_USE):
    results = vectorstore.similarity_search(query, k=k)
    return [(doc.page_content, doc.metadata.get("source", "")) for doc in results]

def build_prompt(context, question):
    prompt_template = (
        "Answer the question based on the following context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    return prompt_template.format(context=context, question=question)

def answer_query(query):
    vectorstore, llm = load_resources()
    top_chunks = get_top_chunks(vectorstore, query)
    if not top_chunks:
        return "No relevant documents found.", []
    context = "\n\n".join([text for text, _ in top_chunks])
    prompt = build_prompt(context, query)
    answer = llm(prompt)
    if isinstance(answer, list):
        answer = answer[0].get("generated_text", "")
    return answer.strip(), top_chunks

st.write("Ask a question based on your uploaded knowledge base!")

query = st.text_input("Enter your question:", "")

if query:
    if not os.path.exists(os.path.join(FAISS_PATH, "index.faiss")):
        st.error("Could not find FAISS index. Make sure you've built the index!")
        st.stop()

    with st.spinner("Searching and generating answer..."):
        answer, sources = answer_query(query)

    st.markdown("### 🧠 Semantic Answer")
    st.success(answer)

    st.markdown("### 📄 Top Sources / Context")
    for i, (chunk, src) in enumerate(sources, 1):
        with st.expander(f"Source {i}: {src if src else 'Unknown source'}"):
            st.write(chunk)
