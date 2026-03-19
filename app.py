
import os
import re
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from bs4 import BeautifulSoup
import urllib.request
import pandas as pd

st.set_page_config(
    page_title="FinSight AI",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for finance theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1B4F72;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #5D6D7E;
        margin-bottom: 2rem;
    }
    .company-badge {
        background-color: #1B4F72;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-right: 8px;
    }
    .source-box {
        background-color: #EBF5FB;
        border-left: 4px solid #1B4F72;
        padding: 10px;
        border-radius: 4px;
        margin-top: 10px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">FinSight AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Financial Document Intelligence Platform — Powered by RAG + Llama 3</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Available Filings")
    st.markdown("""
    | Company | Year | Type |
    |---------|------|------|
    | Apple   | 2023 | 10-K |
    | Apple   | 2022 | 10-K |
    | Amazon  | 2023 | 10-K |
    """)
    
    st.markdown("---")
    st.markdown("### Suggested Questions")
    questions = [
        "What are Apple main revenue sources?",
        "What are Amazon main business segments?",
        "What manufacturing risks does Apple mention?",
        "How does Apple describe its Services business?",
        "What inventory risks does Amazon mention?",
        "Compare Apple and Amazon cloud services"
    ]
    for q in questions:
        if st.button(q, use_container_width=True):
            st.session_state.suggested = q

    st.markdown("---")
    st.markdown("### About")
    st.markdown("FinSight AI analyzes SEC 10-K filings from Apple and Amazon using Retrieval Augmented Generation.")
    st.markdown("**Stack:** LangChain · FAISS · Llama 3.3 70B · HuggingFace")

def extract_clean_text(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "ix:header", "ix:nonfraction", "ix:nonnumeric", "xbrl", "head"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[a-zA-Z]+:[a-zA-Z]+\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    lines = [line.strip() for line in text.split(".") if len(line.strip()) > 50]
    return ". ".join(lines).strip()

@st.cache_resource
def build_pipeline():
    filings_info = {
        "apple_10k_2023": {
            "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm",
            "company": "Apple", "year": "2023"
        },
        "apple_10k_2022": {
            "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019322000108/aapl-20220924.htm",
            "company": "Apple", "year": "2022"
        },
        "amazon_10k_2023": {
            "url": "https://www.sec.gov/Archives/edgar/data/1018724/000101872424000008/amzn-20231231.htm",
            "company": "Amazon", "year": "2023"
        }
    }

    documents = []
    for key, info in filings_info.items():
        try:
            req = urllib.request.Request(
                info["url"],
                headers={"User-Agent": "finsight-ai@gmail.com"}
            )
            with urllib.request.urlopen(req) as response:
                html = response.read().decode("utf-8", errors="ignore")
            text = extract_clean_text(html)
            documents.append(Document(
                page_content=text,
                metadata={"company": info["company"], "year": info["year"], "type": "10-K"}
            ))
        except Exception as e:
            st.warning(f"Could not load {key}: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    groq_api_key = st.secrets["GROQ_API_KEY"]
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, groq_api_key=groq_api_key)

    prompt = ChatPromptTemplate.from_template("""
You are FinSight AI, an expert financial document analyst.
Use the context below from SEC 10-K filings to answer the question.
Always mention the company name and year of the filing you are referencing.
If some information is partial, still provide what you can find.
Be precise with numbers, percentages and financial figures when available.

Context from SEC filings:
{context}

Question: {question}

Analysis:""")

    def format_docs(docs):
        return "\n\n---\n\n".join([
            f"[{doc.metadata['company']} | {doc.metadata['year']} | {doc.metadata['type']}]\n{doc.page_content}"
            for doc in docs
        ])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

# Build pipeline
with st.spinner("Loading SEC filings and building RAG pipeline... (takes 2-3 mins on first load)"):
    rag_chain, retriever = build_pipeline()

st.success("FinSight AI is ready. Ask any question about Apple or Amazon filings.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            st.markdown('<div class="source-box">Sources: ' + " | ".join(msg["sources"]) + '</div>', unsafe_allow_html=True)

# Handle suggested question from sidebar
default_input = st.session_state.pop("suggested", "")

# Chat input
question = st.chat_input("Ask anything about Apple or Amazon financials...")

if not question and default_input:
    question = default_input

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing filings..."):
            answer = rag_chain.invoke(question)
            relevant_docs = retriever.invoke(question)
            sources = list(set([
                f"{doc.metadata['company']} {doc.metadata['year']} {doc.metadata['type']}"
                for doc in relevant_docs
            ]))

        st.markdown(answer)
        st.markdown('<div class="source-box">Sources: ' + " | ".join(sources) + '</div>', unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
