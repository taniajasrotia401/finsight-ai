# FinSight AI — Financial Document Intelligence Platform

A RAG-powered AI that analyzes SEC 10-K filings from Apple and Amazon and answers financial questions with source citations.

## Live Demo
[FinSight AI](https://finsight-ai-ltpmptnk2glhnykiewytnr.streamlit.app/)

## What It Does
Ask any financial question and FinSight AI searches through real SEC 10-K filings and returns precise answers with citations — company name, year and filing type included.

## Available Filings
| Company | Year | Type |
|---------|------|------|
| Apple   | 2023 | 10-K |
| Apple   | 2022 | 10-K |
| Amazon  | 2023 | 10-K |

## Example Questions
- "What are Apple's main revenue sources?"
- "What are Amazon's main business segments?"
- "What manufacturing risks does Apple mention?"
- "What inventory risks does Amazon mention?"
- "How does Apple describe its Services business?"

## Tech Stack
| Component | Technology |
|---|---|
| LLM | Llama 3.3 70B via Groq API |
| Framework | LangChain |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| UI | Streamlit |
| Data Source | SEC EDGAR 10-K Filings |

## Run Locally
```bash
git clone https://github.com/taniajasrotia401/finsight-ai.git
cd finsight-ai
pip install -r requirements.txt
streamlit run app.py
```

## Environment Variables
```
GROQ_API_KEY=your_groq_api_key_here
```

## Author
Tania Jasrotia — Data Science enthusiast | Built RAG systems using LangChain, FAISS and Llama 3 | Open to Data Science and AI roles