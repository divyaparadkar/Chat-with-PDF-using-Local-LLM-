import streamlit as st
import os
import tempfile
import warnings
warnings.filterwarnings('ignore')

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# Page configuration
st.set_page_config(
    page_title="Secure Local PDF Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System via CSS injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Background */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #080c14 !important;
        color: #e2e8f0 !important;
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: #0b111e !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Gradient Header styling */
    .header-gradient {
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
        text-align: center;
        letter-spacing: -0.03em;
    }
    
    .subtitle {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 400;
        font-size: 1.05rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Glassmorphic elements */
    .glass-card {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2) !important;
    }
    
    .glass-card-hover:hover {
        border-color: rgba(255, 255, 255, 0.12) !important;
        background: rgba(255, 255, 255, 0.03) !important;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Chat bubble tweaks */
    [data-testid="stChatMessage"] {
        background-color: rgba(255, 255, 255, 0.025) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 14px !important;
        margin-bottom: 0.8rem !important;
        padding: 1rem !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
    }
    
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }

    /* Style improvements for standard Streamlit badges / buttons */
    div.stButton > button {
        background-color: rgba(99, 102, 241, 0.15) !important;
        color: #a5b4fc !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 10px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }
    
    div.stButton > button:hover {
        background-color: rgba(99, 102, 241, 0.3) !important;
        color: #ffffff !important;
        border-color: rgba(99, 102, 241, 0.6) !important;
        transform: translateY(-1px);
    }

    /* Badges */
    .custom-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
        border: 1px solid;
    }
    .badge-info {
        background-color: rgba(59, 130, 246, 0.1);
        color: #60a5fa;
        border-color: rgba(59, 130, 246, 0.2);
    }
    .badge-success {
        background-color: rgba(16, 185, 129, 0.1);
        color: #34d399;
        border-color: rgba(16, 185, 129, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_file_hash" not in st.session_state:
    st.session_state.processed_file_hash = None

# Sidebar Content
with st.sidebar:
    st.markdown("<div style='text-align: center; padding: 1rem 0;'><h2 style='font-family: Outfit; font-weight:800; color:#e2e8f0; margin:0;'>⚙️ Control Center</h2></div>", unsafe_allow_html=True)
    
    # PDF File Upload
    uploaded_file = st.file_uploader(
        "Upload Document (PDF)",
        type=["pdf"],
        help="Upload a PDF file to process and chat with. All processing stays local."
    )
    
    st.markdown("---")
    
    # Model details
    st.markdown("<h4 style='font-family: Outfit; font-weight:600; color:#e2e8f0; margin-bottom: 0.5rem;'>🔒 Secure Local Stack</h4>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='background: rgba(255,255,255,0.02); padding: 0.8rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); font-size: 0.85rem;'>
            <div style='margin-bottom: 0.4rem;'>🤖 <b>LLM:</b> <code style='color:#a78bfa;'>llama3.2</code></div>
            <div style='margin-bottom: 0.4rem;'>🎨 <b>Embeddings:</b> <code style='color:#60a5fa;'>nomic-embed-text</code></div>
            <div>🗄️ <b>Vector Store:</b> <code style='color:#f472b6;'>ChromaDB</code></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Clear conversation button
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Title Section
st.markdown("<h1 class='header-gradient'>🧠 Secure Local PDF Chat</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Interact with your PDF documents completely offline and private using local LLMs.</p>", unsafe_allow_html=True)

# Helper function to generate file content hash to detect changes
def get_file_identifier(file):
    if file is None:
        return None
    return f"{file.name}_{file.size}"

@st.cache_resource(show_spinner=False)
def initialize_vector_store(file_bytes, file_name):
    # Save file bytes to temporary file to load via PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        # Load PDF
        loader = PyPDFLoader(file_path=temp_path)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        # Build Vector DB
        db_dir = tempfile.mkdtemp()
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model="nomic-embed-text"),
            collection_name="vector_collection",
            persist_directory=db_dir
        )
        return vector_db, len(documents), len(chunks)
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

# Main Area Logic
if uploaded_file is not None:
    file_id = get_file_identifier(uploaded_file)
    
    # If a new file is uploaded, process it
    if st.session_state.processed_file_hash != file_id:
        with st.status("Reading and indexing PDF contents locally...", expanded=True) as status:
            try:
                status.write("Parsing pages...")
                bytes_data = uploaded_file.read()
                
                status.write("Splitting text chunks and embedding...")
                vector_db, num_pages, num_chunks = initialize_vector_store(bytes_data, uploaded_file.name)
                
                status.update(label="PDF processed and embedded successfully!", state="complete", expanded=False)
                st.session_state.processed_file_hash = file_id
                st.session_state.chat_history = []  # reset chat history for the new file
            except Exception as e:
                status.update(label=f"Error indexing PDF: {str(e)}", state="error")
                st.stop()
    else:
        # If it's already indexed, retrieve cached db
        bytes_data = uploaded_file.getvalue()
        vector_db, num_pages, num_chunks = initialize_vector_store(bytes_data, uploaded_file.name)
    
    # Display Stats Badge
    st.markdown(
        f"""
        <div style='margin-bottom: 1.5rem;'>
            <span class='custom-badge badge-success'>✓ Active Document: {uploaded_file.name}</span>
            <span class='custom-badge badge-info'>📄 {num_pages} Pages</span>
            <span class='custom-badge badge-info'>🧩 {num_chunks} Text Chunks</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Initialize the LLM Chain elements
    llm = ChatOllama(model="llama3.2")
    
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )
    
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )
    
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Render Conversation History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Input field for new question
    if user_query := st.chat_input("Ask a question about the document..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Stream response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Searching document context and preparing answer..."):
                # Get the stream generator
                response_stream = chain.stream(user_query)
                
                # Write to UI in real-time
                full_response = st.write_stream(response_stream)
                
        # Append response to session history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

else:
    # Empty State Display
    st.markdown(
        """
        <div class="glass-card glass-card-hover" style="text-align: center; margin-top: 2rem; padding: 3rem 2rem !important;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📂</div>
            <h3 style="font-family: Outfit; font-weight:600; color:#e2e8f0; margin-bottom: 0.5rem;">No Document Uploaded</h3>
            <p style="color: #94a3b8; max-width: 500px; margin: 0 auto 1.5rem auto;">
                Please upload a PDF document in the sidebar to start chatting. Your file will be processed completely locally and never leave your machine.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
