import streamlit as st
import os
import sys

# Hack to make ChromaDB work on Streamlit Cloud (Linux) where sqlite3 version is too old
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# Add the current directory to sys.path so we can import from main.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import build_vector_store, run_crew

# Configuration
DOCS_DIR = os.environ.get("DOCS_DIR", "docs")
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "chroma_db")

# Setup Streamlit page
st.set_page_config(page_title="Smart Book Q&A", page_icon="📚", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    /* Hide Streamlit default header/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main {
        padding-top: 2rem;
    }
    
    /* App Title */
    .app-header {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid #e0e4eb;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .status-ok {
        background-color: #d1fae5;
        color: #065f46;
    }
    .status-warning {
        background-color: #fef3c7;
        color: #92400e;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="app-header">
    <h1>📚 Smart Book Q&A Crew</h1>
    <p style="font-size: 1.2rem; color: #555;">Ask an intelligent agent questions about your specific documents.</p>
</div>
""", unsafe_allow_html=True)

# Ensure docs directory exists
os.makedirs(DOCS_DIR, exist_ok=True)

# Check database status
db_exists = os.path.exists(CHROMA_DB_DIR)
db_status_class = "status-ok" if db_exists else "status-warning"
db_status_text = "Ready (Vector Store Found)" if db_exists else "Action Required (Index Documents First)"

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown(f'<div class="status-badge {db_status_class}">Database Status: {db_status_text}</div>', unsafe_allow_html=True)
    
    st.markdown("### 1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs or Text files", 
        type=["pdf", "txt"], 
        accept_multiple_files=True,
        help="These files will be processed and indexed by the AI."
    )
    
    if uploaded_files:
        if st.button("Save Uploaded Files", use_container_width=True):
            saved_count = 0
            with st.spinner("Saving files..."):
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(DOCS_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_count += 1
            st.success(f"Saved {saved_count} file(s) to '{DOCS_DIR}'!")
            
    st.markdown("---")
    
    st.markdown("### 2. Index Documents")
    st.markdown("Create the vector store so the AI can search through them.")
    if st.button("Build/Rebuild Vector Store", use_container_width=True, type="primary"):
        num_files = len(os.listdir(DOCS_DIR))
        if num_files == 0:
            st.warning(f"No files found in '{DOCS_DIR}'. Please upload files first.")
        else:
            with st.spinner(f"Indexing {num_files} document(s)... This may take a few moments."):
                try:
                    result = build_vector_store(docs_folder=DOCS_DIR)
                    if result:
                        st.success("Vector store built successfully!")
                        st.balloons()
                    else:
                        st.error("Failed to build vector store. Check the logs.")
                except Exception as e:
                    st.error(f"Error during indexing: {str(e)}")
                    
    st.markdown("---")
    # Show current files
    st.markdown("### Cached Files")
    existing_files = os.listdir(DOCS_DIR)
    if existing_files:
        st.caption(f"{len(existing_files)} file(s) ready for indexing:")
        for f in existing_files:
            st.caption(f"- {f}")
    else:
        st.caption("No files currently saved.")


# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the uploaded documents..."):
    # Require DB to exist before answering
    if not os.path.exists(CHROMA_DB_DIR):
        st.warning(f"⚠️ Vector store not found. Please upload documents and click 'Build Vector Store' in the sidebar first.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents to find your answer..."):
                try:
                    # Run the Langchain/Crew process
                    result = run_crew(prompt)
                    
                    # Ensure we extract the raw text if Crew returns an object
                    answer = getattr(result, "raw", str(result))
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"**Error executing search:** {str(e)}\n\n*Check your API keys in the `.env` file.*"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
