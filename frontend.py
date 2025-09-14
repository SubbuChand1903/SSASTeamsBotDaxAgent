"""
DAX Agent Frontend - Modern Streamlit UI with Azure OpenAI Support
Complete latest version with comprehensive dropdowns and agent logging
"""

import streamlit as st
import requests
import json
import uuid
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Optional, Any

# Page config
st.set_page_config(
    page_title="🚀 DAX Analytics AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .config-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .agent-log {
        background: #1e1e1e;
        color: #00ff00;
        padding: 1rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #155724;
    }
    
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #721c24;
    }
    
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .azure-field {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 3px solid #2196f3;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'agent_logs' not in st.session_state:
    st.session_state.agent_logs = []
if 'backend_url' not in st.session_state:
    st.session_state.backend_url = "http://localhost:8000"
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# Constants
BACKEND_URL = st.session_state.backend_url

LLM_MODELS = {
    "openai": ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    "azure_openai": ["gpt-4", "gpt-4.1", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
    "google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
}

SAMPLE_QUERIES = [
    "What are our total sales this year?",
    "Show me top 10 customers by revenue",
    "Sales trend by month for the last 12 months",
    "Product category performance comparison",
    "Revenue breakdown by region",
    "Year over year growth percentage",
    "Customer acquisition trends",
    "Most profitable products this quarter",
    "Sales performance vs targets",
    "Seasonal sales patterns analysis"
]

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API request to backend"""
    url = f"{BACKEND_URL}/api/{endpoint}"
    try:
        if method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error {response.status_code}: {response.text}"}
    except requests.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}

def display_agent_logs(logs: List[Dict]):
    """Display agent execution logs in a modern format"""
    if not logs:
        st.info("🤖 Agent logs will appear here during execution...")
        return
    
    st.subheader("🔍 Agent Execution Timeline")
    
    for i, log in enumerate(logs):
        timestamp = log.get("timestamp", "")
        step = log.get("step", "Unknown Step")
        status = log.get("status", "unknown")
        details = log.get("details", {})
        
        # Create status icon
        status_icon = {
            "started": "🟡",
            "completed": "✅",
            "failed": "❌",
            "unknown": "⚪"
        }.get(status, "⚪")
        
        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime("%H:%M:%S.%f")[:-3]
        except:
            time_str = timestamp
        
        # Create expandable log entry
        with st.expander(f"{status_icon} {step.replace('_', ' ').title()} - {time_str}"):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.write(f"**Status:** {status}")
                st.write(f"**Step:** {step}")
                st.write(f"**Time:** {time_str}")
            
            with col2:
                if details:
                    st.write("**Details:**")
                    st.json(details)

def format_data_for_visualization(result_data: Dict) -> pd.DataFrame:
    """Convert API result data back to DataFrame"""
    if not result_data:
        return pd.DataFrame()
    
    columns = result_data.get("columns", [])
    data = result_data.get("data", [])
    
    return pd.DataFrame(data, columns=columns)

def create_visualization(df: pd.DataFrame, viz_type: str, query: str) -> Optional[go.Figure]:
    """Create visualization based on data and suggested type"""
    if df.empty:
        return None
    
    try:
        if viz_type == "line":
            # Find date/time and numeric columns
            date_cols = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time', 'month', 'year'])]
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if date_cols and numeric_cols:
                fig = px.line(df, x=date_cols[0], y=numeric_cols[0], 
                             title=f"📈 {query}", 
                             template="plotly_white")
                fig.update_layout(height=400)
                return fig
        
        elif viz_type == "bar":
            categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if categorical_cols and numeric_cols:
                # Limit to top 20 for readability
                df_plot = df.head(20)
                fig = px.bar(df_plot, x=categorical_cols[0], y=numeric_cols[0],
                            title=f"📊 {query}",
                            template="plotly_white")
                fig.update_layout(height=400, xaxis_tickangle=-45)
                return fig
        
        elif viz_type == "pie":
            categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if categorical_cols and numeric_cols and len(df) <= 15:
                fig = px.pie(df, values=numeric_cols[0], names=categorical_cols[0],
                            title=f"🥧 {query}",
                            template="plotly_white")
                fig.update_layout(height=400)
                return fig
        
        # Fallback to simple bar chart
        if len(df.columns) >= 2:
            fig = px.bar(df.head(20), x=df.columns[0], y=df.columns[1],
                        title=f"📊 {query}",
                        template="plotly_white")
            fig.update_layout(height=400)
            return fig
            
    except Exception as e:
        st.warning(f"⚠️ Visualization error: {str(e)}")
    
    return None

# =====================================================================
# MAIN APP
# =====================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 DAX Analytics AI</h1>
        <p>Transform natural language into powerful DAX queries with AI-powered analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Backend URL Configuration
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.subheader("🔗 Backend Connection")
        backend_url = st.text_input("Backend URL", value=st.session_state.backend_url)
        if st.button("🔄 Update Backend URL"):
            st.session_state.backend_url = backend_url
            st.success("Backend URL updated!")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Test backend connection
        if st.button("🏥 Test Backend Health"):
            health = make_api_request("health")
            if "error" in health:
                st.error(f"❌ Backend not reachable: {health['error']}")
            else:
                st.success(f"✅ Backend healthy: {health.get('status', 'unknown')}")
        
        st.divider()
        
        # LLM Configuration
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.subheader("🤖 AI Model Configuration")
        
        llm_provider = st.selectbox(
            "LLM Provider",
            options=["openai", "azure_openai", "anthropic", "google"],
            format_func=lambda x: {
                "openai": "🔷 OpenAI",
                "azure_openai": "🔹 Azure OpenAI",
                "anthropic": "🟣 Anthropic",
                "google": "🟢 Google"
            }[x]
        )
        
        model_name = st.selectbox(
            "Model",
            options=LLM_MODELS[llm_provider],
            help="Select the AI model for DAX generation"
        )
        
        api_key = st.text_input(
            f"{llm_provider.replace('_', ' ').title()} API Key",
            type="password",
            help="Your API key for the selected provider"
        )
        
        # Azure OpenAI specific fields
        deployment_name = None
        api_version = None
        
        if llm_provider == "azure_openai":
            st.markdown('<div class="azure-field">', unsafe_allow_html=True)
            st.write("**Azure OpenAI Configuration**")
            deployment_name = st.text_input(
                "Deployment Name",
                value=model_name,
                help="Azure OpenAI deployment name (usually same as model)"
            )
            api_version = st.text_input(
                "API Version",
                value="2024-12-01-preview",
                help="Azure OpenAI API version"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        custom_endpoint = st.text_input(
            "Custom Endpoint" + (" (Required for Azure)" if llm_provider == "azure_openai" else " (Optional)"),
            placeholder="https://your-resource.openai.azure.com/" if llm_provider == "azure_openai" else "https://api.custom-provider.com/v1",
            help="Azure endpoint URL or custom endpoint"
        )
        
        # Validation for Azure OpenAI
        if llm_provider == "azure_openai":
            if not custom_endpoint:
                st.warning("⚠️ Azure endpoint is required")
            if not deployment_name:
                st.warning("⚠️ Deployment name is required")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # SSAS Connection Configuration
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.subheader("🗄️ SSAS Connection")
        
        cube_type = st.selectbox(
            "Cube Type",
            options=["tabular", "multidimensional"],
            format_func=lambda x: {
                "tabular": "📊 Tabular Model",
                "multidimensional": "🧊 Multidimensional Cube"
            }[x]
        )
        
        auth_type = st.selectbox(
            "Authentication",
            options=["windows", "username_password", "connection_string"],
            format_func=lambda x: {
                "windows": "🪟 Windows Authentication",
                "username_password": "👤 Username/Password",
                "connection_string": "🔗 Custom Connection String"
            }[x]
        )
        
        # Connection parameters based on auth type
        server = database = username = password = connection_string = ""
        
        if auth_type in ["windows", "username_password"]:
            server = st.text_input("Server", value="SUBBUCHAND19\\SUBBU", help="SSAS server instance")
            database = st.text_input("Database/Catalog", value="RetailAnalyticsCube", help="Cube database name")
        
        if auth_type == "username_password":
            username = st.text_input("Username", help="Database username")
            password = st.text_input("Password", type="password", help="Database password")
        
        if auth_type == "connection_string":
            connection_string = st.text_area(
                "Connection String",
                value="Provider=MSOLAP;Data Source=SUBBUCHAND19\\SUBBU;Initial Catalog=RetailAnalyticsCube;Integrated Security=SSPI;",
                height=100,
                help="Complete SSAS connection string"
            )
        
        # Test SSAS connection
        if st.button("🔌 Test SSAS Connection"):
            conn_config = {
                "auth_type": auth_type,
                "cube_type": cube_type,
                "server": server,
                "database": database,
                "username": username,
                "password": password,
                "connection_string": connection_string
            }
            
            with st.spinner("Testing connection..."):
                result = make_api_request("test-connection", "POST", conn_config)
                if result.get("success"):
                    st.success("✅ SSAS connection successful!")
                else:
                    st.error(f"❌ Connection failed: {result.get('message', 'Unknown error')}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Sample Queries
        st.subheader("💡 Sample Queries")
        for query in SAMPLE_QUERIES[:6]:  # Show first 6
            if st.button(query, key=f"sample_{query}", use_container_width=True):
                st.session_state.selected_query = query
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Ask Your Business Question")
        query_text = st.text_area(
            "Natural Language Query",
            value=st.session_state.get("selected_query", ""),
            placeholder="Example: Show me the top 10 customers by total sales amount for this year",
            height=100,
            help="Describe what you want to analyze in plain English"
        )
    
    with col2:
        st.subheader("🚀 Execute")
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        
        # Check if all required fields are filled
        required_fields_filled = bool(api_key and query_text.strip())
        if llm_provider == "azure_openai":
            required_fields_filled = required_fields_filled and bool(custom_endpoint and deployment_name)
        
        execute_btn = st.button(
            "🔍 Analyze Data",
            type="primary",
            use_container_width=True,
            disabled=not required_fields_filled
        )
        
        # Show validation messages
        if not api_key:
            st.warning("⚠️ Please enter API key")
        if not query_text.strip():
            st.warning("⚠️ Please enter a query")
        if llm_provider == "azure_openai" and not custom_endpoint:
            st.warning("⚠️ Azure endpoint required")
    
    # Execute query
    if execute_btn:
        if not api_key.strip():
            st.error("🔑 Please provide an API key")
            return
        
        if not query_text.strip():
            st.error("❓ Please enter a query")
            return
        
        if llm_provider == "azure_openai" and not custom_endpoint:
            st.error("🔗 Please provide Azure endpoint")
            return
        
        # Prepare request
        llm_config = {
            "provider": llm_provider,
            "model_name": model_name,
            "api_key": api_key,
            "endpoint": custom_endpoint if custom_endpoint.strip() else None
        }
        
        # Add Azure-specific fields
        if llm_provider == "azure_openai":
            llm_config["deployment_name"] = deployment_name
            llm_config["api_version"] = api_version
        
        request_data = {
            "session_id": st.session_state.session_id,
            "natural_query": query_text.strip(),
            "connection_config": {
                "auth_type": auth_type,
                "cube_type": cube_type,
                "server": server,
                "database": database,
                "username": username,
                "password": password,
                "connection_string": connection_string
            },
            "llm_config": llm_config
        }
        
        # Execute with progress tracking
        progress_placeholder = st.empty()
        
        with st.spinner("🤖 AI Agent is analyzing your query..."):
            # Show initial progress
            progress_placeholder.info("🔄 Starting DAX Agent workflow...")
            
            # Make API call
            result = make_api_request("execute-query", "POST", request_data)
            
            if "error" in result:
                st.error(f"❌ Request failed: {result['error']}")
                return
            
            # Clear progress
            progress_placeholder.empty()
            
            # Store results
            st.session_state.agent_logs = result.get("agent_logs", [])
            st.session_state.last_result = result
            
            # Add to query history
            st.session_state.query_history.append({
                "timestamp": datetime.now(),
                "query": query_text,
                "success": result.get("success", False),
                "execution_time": result.get("execution_time", 0)
            })
    
    # Display results if available
    if st.session_state.last_result:
        result = st.session_state.last_result
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "🔍 Agent Logs", "💻 Generated DAX", "📈 Visualization"])
        
        with tab1:
            st.subheader("📊 Query Results")
            
            # Show execution metrics
            col1, col2, col3, col4 = st.columns(4)
            
            execution_time = result.get("execution_time", 0)
            success = result.get("success", False)
            result_data = result.get("execution_result")
            
            with col1:
                st.markdown(f'<div class="metric-card"><h3>⏱️</h3><p>Execution Time</p><h4>{execution_time:.2f}s</h4></div>', unsafe_allow_html=True)
            with col2:
                rows = len(result_data.get("data", [])) if result_data else 0
                st.markdown(f'<div class="metric-card"><h3>📋</h3><p>Rows Returned</p><h4>{rows}</h4></div>', unsafe_allow_html=True)
            with col3:
                cols = len(result_data.get("columns", [])) if result_data else 0
                st.markdown(f'<div class="metric-card"><h3>📊</h3><p>Columns</p><h4>{cols}</h4></div>', unsafe_allow_html=True)
            with col4:
                status_text = "Success" if success else "Failed"
                status_color = "#4caf50" if success else "#f44336"
                st.markdown(f'<div class="metric-card" style="background: {status_color};"><h3>{"✅" if success else "❌"}</h3><p>Status</p><h4>{status_text}</h4></div>', unsafe_allow_html=True)
            
            # Display data if available
            if result_data:
                df = format_data_for_visualization(result_data)
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"dax_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No data returned from query")
            else:
                st.warning("No result data available")
        
        with tab2:
            st.subheader("🔍 Agent Execution Timeline")
            display_agent_logs(st.session_state.agent_logs)
        
        with tab3:
            st.subheader("💻 Generated DAX Query")
            
            generated_dax = result.get("generated_dax", "")
            if generated_dax:
                st.code(generated_dax, language="sql")
                
                # Copy to clipboard button
                if st.button("📋 Copy DAX to Clipboard"):
                    st.success("✅ DAX query displayed above - copy manually")
                
                # DAX explanation
                with st.expander("🔍 Query Explanation"):
                    st.markdown("""
                    **This DAX query was generated based on your natural language request.**
                    
                    The agent analyzed your question and converted it into the appropriate DAX syntax
                    for your RetailAnalyticsCube structure, using the correct table and column references.
                    """)
            else:
                st.warning("No DAX query was generated")
                if result.get("error_message"):
                    st.error(f"Error: {result['error_message']}")
        
        with tab4:
            st.subheader("📈 Data Visualization")
            
            if result_data:
                df = format_data_for_visualization(result_data)
                if not df.empty:
                    viz_type = result.get("visualization_suggestion", "table")
                    
                    # Create visualization
                    fig = create_visualization(df, viz_type, query_text)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fallback table view
                        st.subheader("📋 Data Table")
                        st.dataframe(df, use_container_width=True)
                    
                    # Visualization controls
                    col1, col2 = st.columns(2)
                    with col1:
                        chart_type = st.selectbox("Chart Type", ["Auto", "Bar Chart", "Line Chart", "Pie Chart"])
                    with col2:
                        color_scheme = st.selectbox("Color Scheme", ["Default", "Viridis", "Plasma", "Blues"])
                else:
                    st.info("No data available for visualization")
            else:
                st.warning("No data available for visualization")
    
    # Query History Section
    if st.session_state.query_history:
        st.subheader("📚 Query History")
        
        history_df = pd.DataFrame(st.session_state.query_history)
        if not history_df.empty:
            # Show recent queries
            for idx, row in history_df.tail(3).iterrows():
                status_icon = "✅" if row['success'] else "❌"
                time_str = row['timestamp'].strftime("%H:%M:%S")
                
                with st.expander(f"{status_icon} {time_str} - {row['query'][:80]}..."):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Time:** {time_str}")
                    with col2:
                        st.write(f"**Duration:** {row['execution_time']:.2f}s")
                    with col3:
                        st.write(f"**Status:** {'Success' if row['success'] else 'Failed'}")
                    
                    st.write(f"**Query:** {row['query']}")
                    
                    if st.button(f"🔄 Re-run this query", key=f"rerun_{idx}"):
                        st.session_state.selected_query = row['query']
                        st.rerun()
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🤖 Agent Status:** Ready")
    with col2:
        st.markdown(f"**🔗 Session:** {st.session_state.session_id[:8]}...")
    with col3:
        st.markdown(f"**📊 Queries:** {len(st.session_state.query_history)}")

if __name__ == "__main__":
    main()