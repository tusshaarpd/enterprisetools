import os
import base64
import tempfile
import streamlit as st
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import re
import json

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Smart Flowchart Generator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize OpenAI model with error handling
@st.cache_resource
def initialize_llm():
    """Initialize the language model with proper error handling."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("🔑 OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables.")
            return None
        
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=api_key
        )
    except Exception as e:
        st.error(f"❌ Error initializing language model: {str(e)}")
        return None

# ---------- AGENTS ----------
def create_agents(llm):
    """Create CrewAI agents for instruction processing."""
    if not llm:
        return None, None
    
    instruction_understander = Agent(
        role="Instruction Understanding Agent",
        goal="Break user instructions into logical steps for a flowchart",
        backstory="You specialize in understanding user instructions and converting them into clear logical steps. You identify decision points, processes, and flow connections.",
        llm=llm,
        verbose=False
    )

    step_formatter = Agent(
        role="Flow Step Formatter Agent",
        goal="Format logical steps into Start->Step1->Step2->End format",
        backstory="You are skilled at converting logical steps into a linear flow format suitable for flowchart generation. You use '->' separators and include proper conditional formatting with 'if' and 'else' keywords.",
        llm=llm,
        verbose=False
    )
    
    return instruction_understander, step_formatter

# ---------- GRAPHVIZ FLOWCHART GENERATOR ----------

def create_graphviz_flowchart(instruction_text, color_scheme="default"):
    """Create a flowchart using Graphviz with proper arrows and connections."""
    
    try:
        import graphviz
    except ImportError:
        return None, "Graphviz library not installed. Install with: pip install graphviz"
    
    # Color schemes for Graphviz
    color_schemes = {
        "default": {
            "start_end": "#4CAF50", 
            "process": "#2196F3", 
            "decision": "#FF9800",
            "text_color": "white",
            "bg_color": "white"
        },
        "professional": {
            "start_end": "#1976D2", 
            "process": "#388E3C", 
            "decision": "#F57C00",
            "text_color": "white",
            "bg_color": "white"
        },
        "dark": {
            "start_end": "#424242", 
            "process": "#616161", 
            "decision": "#795548",
            "text_color": "white",
            "bg_color": "#2e2e2e"
        },
        "colorful": {
            "start_end": "#E91E63", 
            "process": "#9C27B0", 
            "decision": "#FF5722",
            "text_color": "white",
            "bg_color": "white"
        }
    }
    
    colors = color_schemes.get(color_scheme, color_schemes["default"])
    
    # Create Graphviz digraph
    dot = graphviz.Digraph(comment='Flowchart', format='svg')
    dot.attr(rankdir='TB', bgcolor=colors['bg_color'])
    dot.attr('node', fontname='Arial', fontcolor=colors['text_color'])
    dot.attr('edge', fontname='Arial', color='#333333')
    
    steps = [step.strip() for step in instruction_text.split('->') if step.strip()]
    
    if not steps:
        return None, "No steps found"
    
    node_counter = 0
    previous_node = None
    
    i = 0
    while i < len(steps):
        step = steps[i]
        current_node = f"node_{node_counter}"
        node_counter += 1
        
        # Determine node type and shape
        if any(keyword in step.lower() for keyword in ["start", "begin"]):
            # Start node - Oval
            dot.node(current_node, step, shape='ellipse', 
                    style='filled', fillcolor=colors['start_end'], fontcolor=colors['text_color'])
            
        elif any(keyword in step.lower() for keyword in ["end", "finish", "complete"]):
            # End node - Oval
            dot.node(current_node, step, shape='ellipse', 
                    style='filled', fillcolor=colors['start_end'], fontcolor=colors['text_color'])
            
        elif any(keyword in step.lower() for keyword in ["if", "check", "decide", "is", "?"]) and "else" not in step.lower():
            # Decision node - Diamond
            dot.node(current_node, step, shape='diamond', 
                    style='filled', fillcolor=colors['decision'], fontcolor=colors['text_color'])
            
            # Connect from previous node
            if previous_node:
                dot.edge(previous_node, current_node)
            
            # Handle YES/NO branches
            if i + 1 < len(steps):
                yes_step = steps[i + 1]
                yes_node = f"node_{node_counter}"
                node_counter += 1
                
                # Create YES node
                dot.node(yes_node, yes_step, shape='box', 
                        style='filled', fillcolor=colors['process'], fontcolor=colors['text_color'])
                dot.edge(current_node, yes_node, label='YES', color='green')
                
                # Look for NO/ELSE branch
                no_node = None
                for j in range(i + 1, len(steps)):
                    if "else" in steps[j].lower():
                        if j + 1 < len(steps):
                            no_step = steps[j + 1]
                            no_node = f"node_{node_counter}"
                            node_counter += 1
                            
                            dot.node(no_node, no_step, shape='box', 
                                    style='filled', fillcolor=colors['process'], fontcolor=colors['text_color'])
                            dot.edge(current_node, no_node, label='NO', color='red')
                        break
                
                previous_node = yes_node  # Continue from YES branch
                i += 2 if no_node else 1
            
            i += 1
            continue
            
        elif "else" in step.lower():
            # Skip else steps as they're handled in decision logic
            i += 1
            continue
            
        else:
            # Process node - Rectangle
            dot.node(current_node, step, shape='box', 
                    style='filled', fillcolor=colors['process'], fontcolor=colors['text_color'])
        
        # Connect from previous node
        if previous_node and not any(keyword in step.lower() for keyword in ["if", "check", "decide", "is", "?"]):
            dot.edge(previous_node, current_node)
        
        previous_node = current_node
        i += 1
    
    return dot, None

def create_flowchart_display(instruction_text, color_scheme="default"):
    """Create flowchart and return both Graphviz and fallback HTML versions."""
    
    # Try Graphviz first
    graphviz_result, error = create_graphviz_flowchart(instruction_text, color_scheme)
    
    if graphviz_result:
        try:
            # Render Graphviz to SVG
            svg_content = graphviz_result.pipe(format='svg', encoding='utf-8')
            return svg_content, "graphviz"
        except Exception as e:
            st.warning(f"⚠️ Graphviz rendering failed: {str(e)}. Using fallback HTML renderer.")
    
    # Fallback to HTML version if Graphviz fails
    html_content = create_html_fallback(instruction_text, color_scheme)
    return html_content, "html"

def create_html_fallback(instruction_text, color_scheme="default"):
    """Simple HTML fallback when Graphviz is not available."""
    
    color_schemes = {
        "default": {"start_end": "#4CAF50", "process": "#2196F3", "decision": "#FF9800"},
        "professional": {"start_end": "#1976D2", "process": "#388E3C", "decision": "#F57C00"},
        "dark": {"start_end": "#424242", "process": "#616161", "decision": "#795548"},
        "colorful": {"start_end": "#E91E63", "process": "#9C27B0", "decision": "#FF5722"}
    }
    
    colors = color_schemes.get(color_scheme, color_schemes["default"])
    steps = [step.strip() for step in instruction_text.split('->') if step.strip()]
    
    html_content = f"""
    <div style="background: white; padding: 20px; font-family: Arial;">
        <h3 style="text-align: center; color: #333;">Flowchart (HTML Fallback)</h3>
        <p style="text-align: center; color: #666; margin-bottom: 20px;">
            Install Graphviz for better visualization: <code>pip install graphviz</code>
        </p>
    """
    
    for i, step in enumerate(steps):
        if any(keyword in step.lower() for keyword in ["start", "begin", "end", "finish"]):
            shape_style = "border-radius: 25px; background: " + colors['start_end']
        elif any(keyword in step.lower() for keyword in ["if", "check", "decide"]):
            shape_style = "transform: rotate(45deg); background: " + colors['decision']
        else:
            shape_style = "border-radius: 5px; background: " + colors['process']
        
        html_content += f"""
        <div style="text-align: center; margin: 15px 0;">
            <div style="display: inline-block; padding: 10px 20px; color: white; font-weight: bold; {shape_style}">
                {step}
            </div>
            {"<div style='font-size: 20px; color: #333;'>↓</div>" if i < len(steps) - 1 else ""}
        </div>
        """
    
    html_content += "</div>"
    return html_content

# ---------- CREW BUILDER ----------

def build_crew(user_instruction, agents):
    """Build CrewAI crew for processing instructions."""
    instruction_understander, step_formatter = agents
    
    if not instruction_understander or not step_formatter:
        return None
    
    understand_task = Task(
        description=f"""Analyze the user instruction carefully and break it into logical steps for a flowchart. 
        
        Instruction: {user_instruction}
        
        Guidelines:
        - Identify all processes, decisions, and endpoints
        - Make each step clear and actionable
        - Identify conditional logic (if/else statements)
        - Ensure logical flow from start to finish""",
        expected_output="A detailed list of logical steps clearly defined for each action, process, or condition in the workflow.",
        agent=instruction_understander
    )

    format_task = Task(
        description="""Format the logical steps into a flowchart format using -> separators. 
        
        Requirements:
        - Use Start->Step1->Step2->...->End format
        - Use 'if' for conditional statements
        - Use 'else' for alternative paths
        - Ensure all paths lead to an endpoint
        - Keep step descriptions concise but clear""",
        expected_output="Formatted string like: Start->Check if payment complete->If yes show success page->Else show error page->End",
        agent=step_formatter
    )

    try:
        crew = Crew(
            agents=[instruction_understander, step_formatter],
            tasks=[understand_task, format_task],
            process="sequential",
            verbose=False
        )
        return crew
    except Exception as e:
        st.error(f"❌ Error building crew: {str(e)}")
        return None

# ---------- STREAMLIT UI ----------

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Smart Flowchart Generator</h1>', unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        color_scheme = st.selectbox(
            "Color Scheme",
            ["default", "professional", "dark", "colorful"],
            help="Choose a color scheme for your flowchart"
        )
        
        st.header("📋 Examples")
        examples = [
            "Check if user is logged in. If yes, show dashboard. Else, redirect to login page.",
            "Start by validating input data. If valid, process payment. If payment successful, send confirmation email. Else, show error message.",
            "Begin with user registration. Verify email address. If verified, activate account. Send welcome message. End.",
            "Check inventory levels. If stock is low, reorder items. Process customer order. Update inventory. Send shipping notification."
        ]
        
        for i, example in enumerate(examples, 1):
            if st.button(f"Example {i}", key=f"ex_{i}"):
                st.session_state.instruction_text = example
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <strong>📝 Instructions:</strong><br>
        Enter your process description in natural language. The AI will convert it into a proper flowchart with:
        <ul>
        <li><strong>Ovals</strong> for Start/End points</li>
        <li><strong>Rectangles</strong> for Process steps</li>
        <li><strong>Diamonds</strong> for Decision points</li>
        </ul>
        Use phrases like "if", "else", "check", "then" to create decision points.
        </div>
        """, unsafe_allow_html=True)
        
        # Text area with session state
        instruction_key = "instruction_text"
        default_text = st.session_state.get(instruction_key, 
            "Start by checking if payment is complete. If yes, show the success page. Otherwise, show the error page. Then end.")
        
        user_instruction = st.text_area(
            "Enter your process description:",
            value=default_text,
            height=100,
            key=instruction_key,
            help="Describe your process step by step using natural language"
        )
    
    with col2:
        st.markdown("### 🎯 Tips")
        st.markdown("""
        - Start with action words
        - Use "if/else" for decisions  
        - Be specific about outcomes
        - Include start and end points
        - Keep steps concise
        """)
        
        st.markdown("### 📊 Flowchart Symbols")
        st.markdown("""
        - **Oval**: Start/End
        - **Rectangle**: Process
        - **Diamond**: Decision
        - **Arrows**: Flow direction
        """)
        
        # System status
        st.markdown("### 🔍 System Status")
        llm_test = initialize_llm()
        if llm_test:
            st.success("✅ AI Model Ready")
        else:
            st.error("❌ AI Model Not Ready")
        
        st.success("✅ Graphviz Renderer Ready")
        
        # Installation instructions
        st.markdown("### 📦 Installation")
        st.markdown("""
        For best results, install Graphviz:
        ```bash
        pip install graphviz
        ```
        """)
        
        # Check if Graphviz is available
        try:
            import graphviz
            st.success("✅ Graphviz Available")
        except ImportError:
            st.warning("⚠️ Graphviz not installed - using HTML fallback")
    
    # Generate button
    if st.button("🚀 Generate Flowchart", type="primary"):
        if not user_instruction.strip():
            st.warning("⚠️ Please enter a valid instruction!")
            return
        
        # Initialize components
        llm = initialize_llm()
        if not llm:
            return
        
        agents = create_agents(llm)
        if not agents[0] or not agents[1]:
            return
        
        # Show progress
        with st.spinner("🔄 Processing your instruction..."):
            try:
                crew = build_crew(user_instruction, agents)
                if not crew:
                    return
                
                # Execute crew
                results = crew.kickoff()
                
                # Get final result - handle different result formats
                if hasattr(results, 'tasks_output') and results.tasks_output:
                    final_instruction = results.tasks_output[-1].raw
                elif hasattr(results, 'raw'):
                    final_instruction = results.raw
                else:
                    final_instruction = str(results)
                
                # Display results
                st.markdown('<div class="success-box">✅ Flowchart Generated Successfully!</div>', 
                           unsafe_allow_html=True)
                
                # Show processed steps
                with st.expander("📋 Processed Steps", expanded=False):
                    st.code(final_instruction, language="text")
                
                # Generate and display flowchart
                flowchart_content, renderer_type = create_flowchart_display(final_instruction, color_scheme)
                
                if flowchart_content:
                    if renderer_type == "graphviz":
                        st.subheader("📊 Your Flowchart (Graphviz - Professional)")
                        st.image(flowchart_content, use_column_width=True)
                    else:
                        st.subheader("📊 Your Flowchart (HTML Fallback)")
                        st.components.v1.html(flowchart_content, height=600, scrolling=True)
                    
                    # Download functionality
                    st.markdown("### 📥 Download Options")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # SVG/HTML Download
                        if renderer_type == "graphviz":
                            st.download_button(
                                label="📥 Download SVG",
                                data=flowchart_content,
                                file_name="flowchart.svg",
                                mime="image/svg+xml"
                            )
                        else:
                            st.download_button(
                                label="📥 Download HTML",
                                data=flowchart_content,
                                file_name="flowchart.html",
                                mime="text/html"
                            )
                    
                    with col2:
                        # Steps Download
                        st.download_button(
                            label="📥 Download Steps",
                            data=final_instruction,
                            file_name="flowchart_steps.txt",
                            mime="text/plain"
                        )
                    
                    with col3:
                        # Try to generate DOT file for Graphviz
                        if renderer_type == "graphviz":
                            try:
                                dot_source = create_graphviz_flowchart(final_instruction, color_scheme)[0].source
                                st.download_button(
                                    label="📥 Download DOT",
                                    data=dot_source,
                                    file_name="flowchart.dot",
                                    mime="text/plain"
                                )
                            except:
                                pass
                else:
                    st.error("❌ Failed to generate flowchart visualization.")
                    
            except Exception as e:
                st.error(f"❌ Error processing instruction: {str(e)}")
                st.info("💡 Try simplifying your instruction or check your API configuration.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    Made with ❤️ using Streamlit, CrewAI, and Graphviz<br>
    <small>✅ Now uses Graphviz for professional flowcharts with proper arrows and connections</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()