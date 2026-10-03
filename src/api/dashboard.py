import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Enterprise FinOps Auditor Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Autonomous FinOps Governance & Audit Interface")
st.markdown("---")

st.sidebar.header("Configuration Panel")
api_url = st.sidebar.text_input("Auditor Engine API Endpoint", value="http://localhost:8000/api/v1/audit")
st.sidebar.markdown("---")
st.sidebar.info("Target Hardware Core: **NVIDIA RTX 3080 via CUDA**\nLocal Model Model: **Llama-3 (Greedy Mode)**")

st.subheader("Execute Compliance Audit")
user_query = st.text_input(
    "Enter automated auditing instruction or policy query:",
    value="Audit weekend software expenses"
)

# NEW: File Uploader Component
uploaded_file = st.file_uploader("Upload Corporate Ledger Sheet (.csv)", type=["csv"])

if st.button("Trigger Autonomous Audit Pipeline", type="primary"):
    if uploaded_file is None:
        st.warning("Please upload a transactional ledger CSV file to proceed with the execution loop.")
    else:
        with st.spinner("Orchestrating multi-agent state machines... Parsing live file rows and querying CUDA vector spaces..."):
            try:
                # Prepare multipart/form-data payload dictionary
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                data = {"query": user_query}
                
                response = requests.post(api_url, files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("Audit complete! Dynamic analytical data returned successfully.")
                    
                    summary_msg = result.get("summary", "")
                    violations_list = result.get("violations", [])
                    
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric(label="Total Compliance Violations Flagged", value=len(violations_list), delta=len(violations_list), delta_color="inverse")
                    with metric_col2:
                        st.info(f"**Engine Summary Statement:**\n{summary_msg}")
                    
                    st.subheader("Flagged Infraction Items Details Ledger")
                    if violations_list:
                        df = pd.DataFrame(violations_list)
                        display_cols = ["transaction_id", "timestamp", "vendor", "amount", "metadata", "reason"]
                        df = df[display_cols]
                        
                        st.dataframe(
                            df.style.set_properties(**{'background-color': '#fee2e2', 'color': '#991b1b'}, subset=['reason']),
                            use_container_width=True
                        )
                    else:
                        st.balloons()
                        st.success("Zero corporate governance infractions detected across this document stream configuration!")
                        
                else:
                    st.error(f"API Error. Server responded with status code: {response.status_code}")
                    st.caption(response.text)
                    
            except Exception as e:
                st.error(f"Failed to connect to the backend auditor service engine: {str(e)}")
