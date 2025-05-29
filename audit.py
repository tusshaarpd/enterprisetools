import os
import shutil
import threading
import logging
from dotenv import load_dotenv
from flask import Flask, send_from_directory
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Set up models
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Directory to serve static files
STATIC_DIR = "static_docs"
os.makedirs(STATIC_DIR, exist_ok=True)

def load_and_index_document(file_path, file_type):
    """Load and index a document for retrieval."""
    try:
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
        elif file_type == "docx":
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        logger.info(f"Loading document: {file_path}")
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        
        # Enhanced metadata processing
        for i, text in enumerate(texts):
            metadata = text.metadata.copy()
            metadata["source_page"] = metadata.get("page", 1)  # Default to page 1
            metadata["file_path"] = file_path
            metadata["chunk_id"] = i
            texts[i].metadata = metadata
        
        # Create vector store
        db = FAISS.from_documents(texts, embedding)
        retriever = db.as_retriever(search_kwargs={"k": 5})
        
        logger.info(f"Successfully indexed {len(texts)} chunks from {file_path}")
        return retriever, file_path
        
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {str(e)}")
        raise

def format_references(references, static_path, file_type):
    """Format reference links based on file type."""
    ref_text = "\n\n### 📚 References:\n"
    base_name = os.path.basename(static_path)
    
    # Debug: Check if file exists
    if not os.path.exists(static_path):
        logger.error(f"Static file not found: {static_path}")
        ref_text += f"❌ Error: File {base_name} not found in static directory\n"
        return ref_text
    
    seen_pages = set()  # Avoid duplicate page references
    
    for i, ref in enumerate(references, 1):
        source_page = ref.metadata.get("source_page", "Unknown")
        chunk_preview = ref.page_content[:100] + "..." if len(ref.page_content) > 100 else ref.page_content
        
        if file_type == "pdf":
            page_key = f"{base_name}-{source_page}"
            if page_key not in seen_pages:
                # Use Flask server URL for file serving
                file_url = f"http://localhost:5000/static_docs/{base_name}#page={source_page}"
                ref_text += f'**{i}.** [📄 Page {source_page}]({file_url})\n'
                ref_text += f'   *Preview: "{chunk_preview}"*\n\n'
                seen_pages.add(page_key)
        else:
            file_url = f"http://localhost:5000/static_docs/{base_name}"
            ref_text += f'**{i}.** [📝 {base_name}]({file_url})\n'
            ref_text += f'   *Preview: "{chunk_preview}"*\n\n'
    
    return ref_text

def query_document(file, query):
    """Main function to query uploaded documents."""
    if not file or not query.strip():
        return "❌ Please upload a file and enter a query."
    
    try:
        # Determine file type
        file_ext = file.name.split(".")[-1].lower()
        if file_ext not in ["pdf", "docx", "doc"]:
            return "❌ Unsupported file type. Please upload a PDF or Word document."
        
        file_type = "pdf" if file_ext == "pdf" else "docx"
        
        # Save file to static directory
        static_path = os.path.join(STATIC_DIR, os.path.basename(file.name))
        shutil.copy(file.name, static_path)
        
        # Debug: Verify file was copied
        if os.path.exists(static_path):
            logger.info(f"✅ File successfully saved to: {static_path}")
            logger.info(f"File size: {os.path.getsize(static_path)} bytes")
        else:
            logger.error(f"❌ File NOT saved to: {static_path}")
            return "❌ Error: Could not save file to static directory."
        
        # Load and index document
        retriever, _ = load_and_index_document(static_path, file_type)
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=retriever,
            return_source_documents=True
        )
        
        # Get answer
        logger.info(f"Processing query: {query}")
        result = qa_chain({"query": query})
        
        # Get relevant documents for references
        references = retriever.get_relevant_documents(query)
        
        # Format response
        answer = f"## 🤖 Answer:\n{result['result']}\n"
        ref_text = format_references(references, static_path, file_type)
        
        return answer + ref_text
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"❌ Error processing your request: {str(e)}"

# Enhanced Gradio interface
with gr.Blocks(title="Document QA Bot", theme=gr.themes.Soft()) as iface:
    gr.Markdown("""
    # 📚 Document QA Bot with Reference Links
    
    Upload a PDF or Word document and ask questions about its content. 
    For PDFs, you'll get clickable links that jump directly to the relevant pages!
    
    **Supported formats:** PDF, DOC, DOCX
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="📁 Upload Document", 
                file_types=[".pdf", ".docx", ".doc"],
                type="filepath"
            )
            query_input = gr.Textbox(
                label="❓ Enter your question",
                placeholder="What would you like to know about this document?",
                lines=2
            )
            submit_btn = gr.Button("🔍 Ask Question", variant="primary")
            
            gr.Markdown("""
            ### 💡 Tips:
            - Be specific in your questions
            - Try asking about key topics, dates, or names
            - For PDFs, click the page links to view the source
            """)
    
        with gr.Column(scale=2):
            output = gr.Markdown(label="📝 Response")
    
    submit_btn.click(
        fn=query_document,
        inputs=[file_input, query_input],
        outputs=output
    )
    
    # Auto-submit on Enter key
    query_input.submit(
        fn=query_document,
        inputs=[file_input, query_input],
        outputs=output
    )

# Flask app to serve static files
app = Flask(__name__)

@app.route("/static_docs/<path:filename>")
def serve_file(filename):
    """Serve uploaded documents."""
    try:
        file_path = os.path.join(STATIC_DIR, filename)
        logger.info(f"Attempting to serve file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return f"File not found: {filename}", 404
            
        # Check if it's a file (not directory)
        if not os.path.isfile(file_path):
            logger.error(f"Path is not a file: {file_path}")
            return f"Invalid file path: {filename}", 404
            
        logger.info(f"Successfully serving file: {file_path}")
        return send_from_directory(os.path.abspath(STATIC_DIR), filename)
        
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        return f"Error serving file: {str(e)}", 500

@app.route("/health")
def health_check():
    """Health check endpoint."""
    files_in_static = []
    try:
        files_in_static = os.listdir(STATIC_DIR) if os.path.exists(STATIC_DIR) else []
    except:
        pass
    return {
        "status": "healthy", 
        "static_dir": STATIC_DIR,
        "files_available": files_in_static
    }

@app.route("/")
def index():
    """Root endpoint - redirect to Gradio."""
    return """
    <h1>Document QA Bot Server</h1>
    <p>Flask server is running!</p>
    <p><a href="http://localhost:7860">Go to Gradio Interface</a></p>
    <p><a href="/health">Check Health Status</a></p>
    """

def run_gradio():
    """Run Gradio interface."""
    try:
        logger.info("Starting Gradio interface...")
        iface.launch(
            server_name="localhost", 
            server_port=7860, 
            share=False,
            show_error=True
        )
    except Exception as e:
        logger.error(f"Error starting Gradio: {str(e)}")

def run_flask():
    """Run Flask server."""
    try:
        logger.info("Starting Flask server...")
        logger.info(f"Static directory: {os.path.abspath(STATIC_DIR)}")
        app.run(port=5000, debug=True, use_reloader=False, host='localhost')
    except Exception as e:
        logger.error(f"Error starting Flask: {str(e)}")

# Main execution
if __name__ == "__main__":
    try:
        # Start Gradio in a separate thread
        gradio_thread = threading.Thread(target=run_gradio, daemon=True)
        gradio_thread.start()
        
        # Start Flask server (blocking)
        run_flask()
        
    except KeyboardInterrupt:
        logger.info("Shutting down servers...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")