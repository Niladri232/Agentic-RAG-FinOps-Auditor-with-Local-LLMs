import os
import warnings
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Dict, Any

# Suppress framework output noise completely
logging.basicConfig(level=logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.simplefilter("ignore")

# Core Architecture Imports
from src.agents.orchestrator import FinOpsOrchestrator
from langchain_core.messages import HumanMessage

app = FastAPI(
    title="Enterprise FinOps Auditor Engine API",
    description="Dynamic REST interface for file-driven transaction auditing.",
    version="2.1.0"
)

# Instantiate the orchestration engine globally
orchestrator = FinOpsOrchestrator()

@app.post("/api/v1/audit")
async def run_compliance_audit(
    query: str = Form(...), 
    file: UploadFile = File(...)
):
    """
    Accepts an audit query strategy and an uploaded transaction CSV file,
    streaming the text content directly into the LangGraph orchestration framework.
    """
    try:
        # 1. Read file bytes and decode to text string stream
        contents = await file.read()
        csv_text = contents.decode("utf-8")
        
        # 2. Build the initial state payload for the entire graph execution pipeline
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "retrieved_rules": "",
            # We add the raw uploaded content string to the state dictionary
            "raw_csv_data": csv_text, 
            "transactions_to_audit": [],
            "audit_violations": []
        }
        
        # 3. Invoke the complete compiled graph natively from START to END
        final_state = orchestrator.executable_graph.invoke(initial_state)
        
        return {
            "summary": final_state["messages"][-1].content,
            "violations": final_state["audit_violations"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph Pipeline Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "gpu_acceleration": "CUDA active"}

if __name__ == "__main__":
    import uvicorn
    # FIX: Pass the 'app' instance directly instead of a module lookup string path
    uvicorn.run(app, host="0.0.0.0", port=8000)
