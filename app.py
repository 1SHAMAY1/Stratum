import streamlit as st
from src.pipeline import StratumPipeline

st.set_page_config(
    page_title="Stratum · Document Intelligence",
    page_icon="🪨",
    layout="wide",
)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = StratumPipeline()

st.title("🪨 Stratum")

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
    if st.button("Build Knowledge Tree"):
        st.write("Ingesting...")
