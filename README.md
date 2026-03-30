# CiteMed-RAG: RAG-Enhanced Bilingual Patient-Education Assistant

**Assignment 2 - LLMs and Foundation AI Models**
**Domain:** Healthcare

---

## Project Description

CiteMed-RAG is a bilingual (English/Arabic) patient-education assistant that explains
common health topics in plain language. It builds on the Assignment 1 baseline
(CiteMed-Prompt) and adds Retrieval-Augmented Generation (RAG) to ground responses
in a curated medical knowledge base.

Features:
- Bilingual support (English and Arabic)
- RAG-enhanced answers with source citations [1], [2], etc.
- Safety-first: refuses diagnosis, prescribing, dosing; escalates emergencies
- Retrieval trace so users can inspect which sources were used
- Toggle between A1 baseline and A2 RAG mode

Improvement over Assignment 1:
- Factuality: RAG retrieves from 15 curated medical documents
- Citations: responses now include verifiable source references
- Judge bias fixed: different judge model from generation model
- Hallucination reduction: grounding instructions and gap handling

## Architecture

```
User Query (EN/AR)
       |
       v
  Language Detection
       |
       v
  Query Embedding       Knowledge Base (15 docs: WHO, MedlinePlus, NHS)
  (gemini-embedding-001)      |
       |                 Chunking (400 words, 100 overlap)
       |                      |
       v                      v
  +--------------------------------------+
  |     ChromaDB Vector Store             |
  |     (Cosine Similarity, top-k=5)      |
  +------------------+-------------------+
                     | Retrieved chunks
                     v
  +--------------------------------------+
  |  RAG Prompt (context + citations +    |
  |  safety rules)                        |
  +------------------+-------------------+
                     |
                     v
  +--------------------------------------+
  |  Gemini 2.0 Flash (temp=0.3, seed=42)|
  +------------------+-------------------+
                     |
                     v
  Grounded Response with Citations

Evaluation:
  25 Test Cases --> A1 + A2 --> Judge (gemini-2.5-flash-lite, not the generation model)
  6 Criteria: Clarity, Safety, Factuality, Groundedness, Citation Accuracy, Retrieval Relevance
```

## Setup Instructions

### Deployed App
Visit the Streamlit Cloud deployment and enter your Gemini API key in the sidebar.

### Run Locally
```bash
git clone https://github.com/hhayaa/CiteMed-RAG.git
cd CiteMed-RAG
pip install -r requirements.txt
streamlit run app.py
```

### Google Colab
Open Assignment2_CiteMed_RAG.ipynb in Colab and run all cells.

### API Key
Get a free key from https://aistudio.google.com/apikey

## Repository Structure

```
CiteMed-RAG/
  app.py                         Streamlit application
  requirements.txt               Dependencies
  knowledge_base.json            15 curated medical documents
  Assignment2_CiteMed_RAG.ipynb  Full evaluation notebook
  README.md
  .gitignore
```

## Tech Stack
- LLM: Google Gemini 2.0 Flash + Gemini Embedding 001
- Vector Store: ChromaDB
- Frontend: Streamlit
- Knowledge Base: 15 documents (WHO, MedlinePlus, NHS, ADA, AHA)
- Evaluation: LLM-as-judge, 6 criteria, 25 test cases

## References
- Huyen, C. (2025). AI Engineering. O Reilly.
- Alammar, J. and Groenendiijk, M. (2025). Hands-On Large Language Models.
- Google AI Documentation: https://ai.google.dev/docs
- ChromaDB: https://docs.trychroma.com
- WHO Fact Sheets: https://www.who.int/news-room/fact-sheets
- MedlinePlus: https://medlineplus.gov

## Acknowledgments
This project used AI assistants (OpenAI ChatGPT + Google Gemini) for prompt wording
and report drafting. All design choices, evaluation, and analysis are original.
