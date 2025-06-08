# Enterprise Log Analyzer (Streamlit + CrewAI + Claude 3.5)
import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import re
from datetime import datetime

# Load API keys
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    st.error("Missing ANTHROPIC_API_KEY. Please set it in .env file.")
    st.stop()

# Set environment variable for CrewAI to use
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

# --- Streamlit UI ---
st.title("🔍 Enterprise Log Analyzer")
st.write("Upload log entries to categorize, explain, and get remediation suggestions using AI agents.")

# Sidebar for configuration
st.sidebar.header("Configuration")
max_logs = st.sidebar.slider("Max logs to analyze", 1, 50, 10)
show_debug = st.sidebar.checkbox("Show debug output", False)

# Sample logs for testing
sample_logs = """2024-01-15 10:30:15 ERROR [DatabaseConnector] Connection timeout after 30s to mysql://prod-db:3306
2024-01-15 10:30:16 WARNING [MemoryManager] Heap usage at 85% (3.4GB/4GB)
2024-01-15 10:30:17 INFO [UserService] User authentication successful for user_id=12345
2024-01-15 10:30:18 CRITICAL [SecurityAlert] Multiple failed login attempts detected from IP 192.168.1.100
2024-01-15 10:30:19 DEBUG [CacheManager] Cache hit ratio: 87% (435/500 requests)"""

# Input section
input_method = st.radio("Choose input method:", ["Paste logs", "Use sample logs"])

if input_method == "Use sample logs":
    log_input = sample_logs
    st.text_area("Sample log entries:", value=sample_logs, height=150, disabled=True)
else:
    log_input = st.text_area("Paste raw log lines here (one per line):", height=150, 
                            placeholder="Enter log entries, one per line...")

if st.button("🚀 Analyze Logs", type="primary") and log_input:
    log_lines = [line.strip() for line in log_input.strip().split("\n") if line.strip()]
    
    if len(log_lines) > max_logs:
        log_lines = log_lines[:max_logs]
        st.warning(f"Analyzing first {max_logs} log entries only.")
    
    with st.spinner("Analyzing logs with AI agents..."):
        try:
            # --- Define Agents with proper LLM configuration ---
            classifier_agent = Agent(
                role='Senior Log Classification Specialist',
                goal='Accurately categorize log entries by severity level and provide clear reasoning',
                backstory="""You are a senior DevOps engineer with 10+ years of experience in 
                system monitoring and log analysis. You excel at quickly identifying the severity 
                and nature of log entries across various systems and applications.""",
                verbose=show_debug,
                llm="claude-3-5-haiku-20241022"  # Correct model name for CrewAI
            )
            
            explainer_agent = Agent(
                role='System Behavior Analyst',
                goal='Translate technical log entries into clear, actionable insights',
                backstory="""You are a technical communication expert who specializes in 
                making complex system behaviors understandable to both technical and 
                non-technical stakeholders. You have deep knowledge of common system patterns.""",
                verbose=show_debug,
                llm="claude-3-5-haiku-20241022"
            )
            
            remediation_agent = Agent(
                role='Infrastructure Problem Solver',
                goal='Provide practical, tested solutions for system issues',
                backstory="""You are a senior site reliability engineer with extensive 
                experience in incident response and system optimization. You focus on 
                providing actionable, tested solutions that prevent recurring issues.""",
                verbose=show_debug,
                llm="claude-3-5-haiku-20241022"
            )
            
            # --- Define Tasks ---
            classification_task = Task(
                description=f"""
                Analyze these {len(log_lines)} log entries and classify each one:
                
                LOG ENTRIES:
                {chr(10).join(f"{i+1}. {line}" for i, line in enumerate(log_lines))}
                
                For each log entry, provide:
                1. **Severity Level**: INFO, DEBUG, WARNING, ERROR, or CRITICAL
                2. **Component**: What system/service generated this log
                3. **Classification Reason**: Why you chose this severity level
                
                Format your response as:
                ## Log Classification Results
                
                **Entry 1:** [Original log line]
                - **Severity:** [LEVEL]
                - **Component:** [System/Service]
                - **Reason:** [Brief explanation]
                
                [Continue for all entries...]
                """,
                expected_output="Structured classification of all log entries with severity levels and reasoning",
                agent=classifier_agent
            )
            
            explanation_task = Task(
                description="""
                Based on the classified log entries, provide detailed explanations for each entry:
                
                For each log entry, explain:
                1. **What happened**: Plain English explanation of the event
                2. **Likely cause**: Most probable reasons for this log entry
                3. **Business impact**: How this affects system performance or users
                4. **Urgency level**: How quickly this needs attention
                
                Focus especially on WARNING, ERROR, and CRITICAL entries.
                """,
                expected_output="Detailed explanations of what each log entry means and its implications",
                agent=explainer_agent,
                context=[classification_task]
            )
            
            remediation_task = Task(
                description="""
                For any WARNING, ERROR, or CRITICAL log entries, provide specific remediation guidance:
                
                For each problematic entry, provide:
                1. **Immediate Actions**: Steps to take right now
                2. **Root Cause Investigation**: How to find the underlying issue
                3. **Long-term Prevention**: How to prevent this from recurring
                4. **Monitoring Recommendations**: What to watch for in the future
                
                Prioritize solutions by severity level (CRITICAL first, then ERROR, then WARNING).
                """,
                expected_output="Actionable remediation steps for all problematic log entries",
                agent=remediation_agent,
                context=[classification_task, explanation_task]
            )
            
            # --- Build and Execute Crew ---
            log_analysis_crew = Crew(
                agents=[classifier_agent, explainer_agent, remediation_agent],
                tasks=[classification_task, explanation_task, remediation_task],
                verbose=show_debug,
                process=Process.sequential
            )
            
            # --- Run Analysis ---
            with st.status("Running AI analysis...", expanded=True) as status:
                st.write("🔍 Classifying log entries...")
                st.write("💡 Generating explanations...")  
                st.write("🛠️ Creating remediation plans...")
                
                result = log_analysis_crew.kickoff()
                status.update(label="Analysis complete!", state="complete")
            
            # --- Display Results ---
            st.success("✅ Analysis completed successfully!")
            
            # Display the final result
            if hasattr(result, 'raw'):
                st.subheader("📊 Complete Analysis Report")
                st.markdown(result.raw)
            elif isinstance(result, str):
                st.subheader("📊 Complete Analysis Report")
                st.markdown(result)
            else:
                st.subheader("📊 Analysis Results")
                st.write(result)
            
            # Additional insights
            st.subheader("📈 Summary Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Logs Analyzed", len(log_lines))
            
            with col2:
                # Count different severity levels (basic regex matching)
                error_count = sum(1 for line in log_lines if re.search(r'\b(ERROR|CRITICAL)\b', line, re.IGNORECASE))
                st.metric("Critical Issues", error_count)
            
            with col3:
                warning_count = sum(1 for line in log_lines if re.search(r'\bWARNING\b', line, re.IGNORECASE))
                st.metric("Warnings", warning_count)
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            if show_debug:
                st.exception(e)
            
            # Provide fallback suggestions
            st.subheader("🔧 Troubleshooting Suggestions")
            st.write("""
            **Common issues and solutions:**
            - Ensure your ANTHROPIC_API_KEY is valid and has sufficient credits
            - Check that CrewAI is properly installed: `pip install crewai crewai-tools`
            - Verify log format is readable (one entry per line)
            - Try with fewer log entries if you're hitting rate limits
            """)

else:
    if not log_input:
        st.info("👆 Please enter some log entries to analyze, or select 'Use sample logs' to try with example data.")

# --- Footer ---
st.markdown("---")
st.markdown("""
**About this tool:**
This Enterprise Log Analyzer uses multiple AI agents to provide comprehensive log analysis:
- **Classifier Agent**: Categorizes logs by severity
- **Explainer Agent**: Provides human-readable explanations  
- **Remediation Agent**: Suggests practical solutions

Built with [CrewAI](https://crewai.com) and [Claude 3.5 Haiku](https://anthropic.com).
""")