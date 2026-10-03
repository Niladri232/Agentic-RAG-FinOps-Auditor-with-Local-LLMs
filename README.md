# Autonomous Localized FinOps Governance & Audit Framework

An edge-deployed, privacy-preserving multi-agent auditing framework built with **LangGraph**, **FastAPI**, **Streamlit**, and local **Llama-3 (via Ollama & CUDA acceleration)** on an NVIDIA RTX 3080.

---

## 📌 Key Features

- **Agentic Workflow Orchestration:** Uses **LangGraph** state graph machinery to coordinate data parsing, policy retrieval, and evaluation steps.
- **Privacy-Preserving Local Evaluation:** All transaction auditing is executed locally using quantized **Llama-3 (8B)** via Ollama—zero data leaves the local workstation.
- **Dynamic Vector Policy Retrieval:** Employs **ChromaDB** with `all-MiniLM-L6-v2` embeddings for semantic context matching against corporate policies.
- **Decoupled Architecture:** Asynchronous **FastAPI** backend endpoint serving an interactive **Streamlit** visual dashboard.
- **Multi-Rule Evaluation:** Detects temporal infractions (e.g., unauthorized weekend software expenses) and volumetric budget threshold breaches ($5,000+).

---

## 📁 Repository Structure

```text
finops_auditor/
├── test_ledger.csv            # Sample evaluation transaction ledger
├── Dockerfile                 # Container setup configuration
├── docker-compose.yml         # Multi-container orchestration
├── requirements.txt           # Python dependency manifests
└── src/
    ├── agents/
    │   └── orchestrator.py    # LangGraph multi-agent execution pipeline
    ├── api/
    │   ├── server.py          # FastAPI REST ingestion backend service
    │   └── dashboard.py       # Streamlit interactive UI panel
    └── pipelines/
        ├── data_prep.py       # Stream parsing & temporal filtering module
        └── vector_store.py    # ChromaDB vector store policy manager
