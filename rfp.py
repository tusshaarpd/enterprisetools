import streamlit as st
import os
import io
from dotenv import load_dotenv
import PyPDF2
import docx
from crewai import Agent, Task, Crew, Process

# --- Step 1: Load environment variables ---
load_dotenv()

# Check if OpenAI API key exists
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY not found. Check your .env file.")
    st.stop()

# --- Step 2: Text extraction functions ---
def extract_text_from_pdf(file_bytes):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return "".join([page.extract_text() for page in pdf_reader.pages])

def extract_text_from_docx(file_bytes):
    doc = docx.Document(io.BytesIO(file_bytes))
    return "".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file_bytes):
    return file_bytes.decode("utf-8")

# --- Step 3: Define Agents ---
reader = Agent(
    role='RFP Reader',
    goal='Extract content from RFPs and RFIs.',
    backstory='An expert in understanding RFP documents.',
    verbose=True,
    llm="gpt-4o-mini"
)

analyzer = Agent(
    role='RFP Analyzer',
    goal='Analyze the RFP for key asks, deadlines, compliance, and scoring.',
    backstory='An expert in proposal evaluation and sales enablement.',
    verbose=True,
    llm="gpt-4o-mini"
)

summarizer = Agent(
    role='Proposal Summarizer',
    goal='Summarize findings in a business-friendly format.',
    backstory='A professional who summarizes technical content for sales teams.',
    verbose=True,
    llm="gpt-4o-mini"
)

# --- Step 4: Define Tasks ---
read_rfp_task = Task(
    description="Extract and clean content from the uploaded RFP/RFI document: {document_text}",
    expected_output="Raw text extracted from the document.",
    agent=reader
)

analyze_rfp_task = Task(
    description="""Analyze the extracted RFP content to identify:
- Key asks and deliverables
- Submission deadlines
- Compliance or eligibility requirements
- Evaluation and scoring criteria
Use the content from the previous task.""",
    expected_output="Structured findings from the RFP/RFI.",
    agent=analyzer
)

summarize_task = Task(
    description="Summarize the RFP/RFI analysis in bullet points for sales/pre-sales team. Use the analysis from the previous task.",
    expected_output="Sales-focused summary of the RFP/RFI.",
    agent=summarizer
)

# --- Step 5: Create Crew ---
rfp_crew = Crew(
    agents=[reader, analyzer, summarizer],
    tasks=[read_rfp_task, analyze_rfp_task, summarize_task],
    verbose=True,
    process=Process.sequential
)

# --- Step 6: Streamlit UI ---
st.title("📄 RFP/RFI Analyzer")

uploaded_file = st.file_uploader("Upload RFP or RFI document", type=["pdf", "docx", "txt"])

if st.button("Analyze Document"):
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        
        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            extracted_text = extract_text_from_pdf(file_bytes)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            extracted_text = extract_text_from_docx(file_bytes)
        elif uploaded_file.type == "text/plain":
            extracted_text = extract_text_from_txt(file_bytes)
        else:
            st.error("Unsupported file format.")
            st.stop()
        
        # Run the crew with the extracted text
        with st.spinner("Analyzing document..."):
            try:
                result = rfp_crew.kickoff(inputs={"document_text": extracted_text})
                
                # Display the result
                st.subheader("✅ Analysis Complete")
                st.markdown(str(result))
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    else:
        st.warning("Please upload a valid RFP or RFI document.")