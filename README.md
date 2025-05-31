 # enterprisetools

1. Document QA Bot – AI-Powered Document Intelligence Tool
Built using Gradio, LangChain, OpenAI GPT-4o, and FAISS
This tool allows users to upload PDF or Word documents and ask natural-language questions to instantly extract relevant answers. Designed for speed, accuracy, and transparency, it leverages cutting-edge AI and vector search to streamline document review.

Core Features:
1. Supports PDF and Word (DOCX) document formats
2. Extracts answers using semantic search across document content
3. Returns contextual answers with references to specific page numbers
4. References include clear, readable links for traceability
5. Built with OpenAI GPT-4o and LangChain’s RetrievalQA for accurate language understanding
6. Uses FAISS for fast vector-based retrieval
7. Secure API integration and easy-to-use web interface powered by Gradio

How It Helps Audit Teams:
1. Accelerates review of lengthy policy documents, contracts, and compliance records
2. Reduces manual effort in locating specific clauses or evidence
3. Improves accuracy in referencing and citing source documents
4. Enhances productivity during audits, investigations, or regulatory reporting
5. Supports auditors in quickly verifying compliance-related queries with traceable sources
****************************************************************************************************************************************************************************
2. Smart Contract Analyzer (SCA)
Created a web-based application that enables legal, procurement, or business teams to upload contracts and receive:
A structured summary of key clauses (obligations, penalties, timelines)
Highlighted risks or red flags
Clause-level classification (e.g., Indemnity, Termination, Jurisdiction)
Suggested standard language vs. deviation detection

Core Features (Summary)
1. Document Upload
2. Accepts PDF, DOCX, TXT (up to 25MB)
3. Text Extraction & Preprocessing
4.Extracts and cleans text using tools like Tika or PyMuPDF
5. Preserves page-level context for traceability
6. Clause Detection & Classification
7.Identifies and categorizes key clauses using legal NLP models (e.g., Legal-BERT, ContractNLI)
8.Targets clause types: Termination, Indemnification, Confidentiality, Governing Law, Liability, Payment Terms, IP Ownership
9.Clause Summarization
10. Generates natural-language summaries of each clause
This was created with help of Manus AI 

