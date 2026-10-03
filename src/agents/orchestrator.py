import os
import warnings
import logging

# 1. ROOT LEVEL SILENCER: Control background output noise programmatically
logging.basicConfig(level=logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.simplefilter("ignore")

# 2. Structural Imports
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional
import operator
import json
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

# --- Modular System Ingestions ---
from src.pipelines.vector_store import PolicyVectorStoreManager
from src.pipelines.data_prep import FinancialDataPipeline

# Expanded Graph State to carry the uploaded raw CSV text stream along the network path
class AgentWorkflowState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    retrieved_rules: str
    raw_csv_data: Optional[str]  # Holds the raw uploaded CSV text content stream
    transactions_to_audit: List[Dict[str, Any]]
    audit_violations: List[Dict[str, Any]]

class FinOpsOrchestrator:
    def __init__(self):
        print("Initializing Local ChatOllama Engine on RTX 3080...")
        self.llm = ChatOllama(model="llama3", base_url="http://localhost:11434", temperature=0.0)
        self.vector_manager = PolicyVectorStoreManager()
        self.data_pipeline = FinancialDataPipeline()
        self._build_graph()
                

    def triage_input_node(self, state: AgentWorkflowState) -> dict:
        """Evaluates incoming query, fetches corporate policy rules, and maps data targets dynamically."""
        last_message = state["messages"][-1].content
        print(f"\n[Node: Triage] Evaluating user strategy request: '{last_message}'")
        
        # 1. Similarity search against ChromaDB to find relevant rules
        compliance_context = self.vector_manager.get_relevant_rules(str(last_message), k=1)
        
        # 2. Check if an uploaded raw CSV data string exists in the current state context
        # If it exists, parse it dynamically; otherwise, fall back to the native mock file system tracker.
        if "raw_csv_data" in state and state["raw_csv_data"]:
            print("[Node: Triage] Incoming upload file stream detected. Routing to CSV text parsing processor...")
            all_transactions = self.data_pipeline.parse_csv_bytes(state["raw_csv_data"])
        else:
            print("[Node: Triage] No upload stream detected. Falling back to local internal mock ledger file...")
            all_transactions = self.data_pipeline.load_mock_ledger()
            
        # 3. Isolate targeted weekend rows using the pipeline engine
        target_transactions = self.data_pipeline.extract_weekend_transactions(all_transactions)
        print(f"[Node: Triage] Found {len(target_transactions)} weekend transactions out of {len(all_transactions)} total rows.")

        # Update global state variables
        return {
            "retrieved_rules": compliance_context,
            "transactions_to_audit": target_transactions
        }

    def execute_audit_node(self, state: AgentWorkflowState) -> dict:
        """Loops through isolated transactions, validating them via the local LLM against retrieved rules."""
        print(f"\n[Node: Execute Audit] Cross-referencing transactions against retrieved compliance policies...")
        
        rules = state["retrieved_rules"]
        txns = state["transactions_to_audit"]
        violations = []
        
        # Construct verification loop for targeted analysis
        for txn in txns:
            prompt = (
                f"You are a strict corporate financial compliance validator. Your task is to evaluate if a transaction complies with corporate rules.\n\n"
                f"[CORPORATE POLICY RULE CONTEXT]:\n{rules}\n\n"
                f"[CRITICAL EVALUATION CRITERIA]:\n"
                f"Weekend software charges are strictly PROHIBITED unless the Metadata description explicitly contains the words 'emergency devops'. No other phrases or excuses are valid.\n\n"
                f"[FEW-SHOT EXAMPLES]:\n"
                f"Example 1:\n"
                f"Timestamp: 2026-05-17 (Weekend)\n"
                f"Metadata: Routine modifications for design layout\n"
                f"Output: VIOLATION - Weekend charge lacks emergency devops metadata.\n\n"
                f"Example 2:\n"
                f"Timestamp: 2026-05-16 (Weekend)\n"
                f"Metadata: Emergency devops patching for server crash\n"
                f"Output: COMPLIANT - Transaction contains required emergency devops flag.\n\n"
                f"[TARGET TRANSACTION TO EVALUATE]:\n"
                f"Vendor: {txn['vendor']}\n"
                f"Amount: ${txn['amount']}\n"
                f"Timestamp: {txn['timestamp']}\n"
                f"Metadata Description: {txn['metadata']}\n\n"
                f"Provide your verdict following the exact format of the examples above (Start with 'VIOLATION' or 'COMPLIANT'):"
            )
            
            response = self.llm.invoke(prompt)
            verdict = str(response.content).strip()
            
            # Print clean status line to console
            clean_verdict_line = verdict.splitlines()[0] if verdict else "UNKNOWN"
            print(f" -> Txn {txn['transaction_id']} ({txn['vendor']}): {clean_verdict_line}")
            
            if "VIOLATION" in verdict.upper():
                txn_copy = txn.copy()
                txn_copy["reason"] = verdict
                violations.append(txn_copy)

        # Build final summarizing communication response
        summary_text = (
            f"Audit compilation complete. Scanned {len(txns)} targeted rows. "
            f"Detected {len(violations)} corporate governance infractions."
        )
        
        return {
            "messages": [AIMessage(content=summary_text)],
            "audit_violations": violations
        }

    def _build_graph(self):
        """Compiles a dual-node multi-agent blueprint for complete data-driven auditing."""
        workflow = StateGraph(AgentWorkflowState)
        
        # Add the multi-tiered nodes
        workflow.add_node("triage", self.triage_input_node)
        workflow.add_node("execute_audit", self.execute_audit_node)
        
        # Configure Linear Transition States
        workflow.set_entry_point("triage")
        workflow.add_edge("triage", "execute_audit")  # Flow moves sequentially from triage into execution
        workflow.add_edge("execute_audit", END)
        
        self.executable_graph = workflow.compile()

if __name__ == "__main__":
    orchestrator = FinOpsOrchestrator()
    print("LangGraph Multi-Agent Execution Blueprint compiled.")
    
    initial_state = {
        "messages": [HumanMessage(content="Audit weekend software expenses")],
        "retrieved_rules": "",
        "raw_csv_data": None,  # Mock execution default parameters definition
        "transactions_to_audit": [],
        "audit_violations": []
    }
    
    final_output = orchestrator.executable_graph.invoke(initial_state)
    
    print("\n" + "="*50)
    print("--- AUTOMATED COMPLIANCE VERDICT SUMMARY ---")
    print(final_output["messages"][-1].content)
    print("="*50)
    print("Flagged Infraction Items Details Ledger:")
    print(json.dumps(final_output["audit_violations"], indent=2))
