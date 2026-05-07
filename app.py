import streamlit as st
from src.pipeline import StratumPipeline

st.set_page_config(
    page_title="Stratum · Document Intelligence",
    page_icon="🪨",
    layout="wide",
)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = StratumPipeline()
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🪨 Stratum")

# Simple Chat Layout
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    answer = st.session_state.pipeline.execute_query(prompt)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
