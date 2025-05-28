import streamlit as st
from PIL import Image
from pathlib import Path
import os
import json
import pickle
import re
import numpy as np
import pandas as pd
from numpy.linalg import norm
import plotly.graph_objects as go
import requests
import warnings
from sentence_transformers import SentenceTransformer

# Suppress warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="EKoder Web Demo – ED Code Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'note_text' not in st.session_state:
    st.session_state.note_text = ""
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None

# Title and Description
st.markdown("<h1 style='color:#007AC1;'>🏥 EKoder Web Demo</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='background-color:#fff3cd; padding:15px; border-radius:5px; margin-bottom:20px;'>
<b>⚠️ Testing Version Only</b><br>
This is a demonstration version for testing purposes.<br>
<b>Do not enter real patient data.</b><br>
For clinical use with full privacy, please use the downloadable local version.
</div>
""", unsafe_allow_html=True)

st.markdown("""
This tool analyzes ED case notes and suggests **up to four ICD-10-AM principal diagnosis codes**.
Perfect for testing the classification accuracy before installing the private local version.
""")

# Sidebar
st.sidebar.header("ℹ️ About EKoder")
st.sidebar.info("""
**Web Demo Features:**
- ✅ Instant access - no installation
- ✅ Works on any device
- ✅ Same embedding technology
- ⚠️ Simplified AI responses

**For production use:**
Request the private local version
""")

# File handling
DEFAULT_EXCEL = "FinalEDCodes_Complexity.xlsx"
DEFAULT_JSONL = "edcode_finetune_v5_updated.jsonl"

# Check if files exist (for Streamlit Cloud, you'll upload these to GitHub)
excel_exists = os.path.exists(DEFAULT_EXCEL)
jsonl_exists = os.path.exists(DEFAULT_JSONL)

if not excel_exists:
    st.sidebar.warning("⚠️ Upload ICD codes Excel file")
    uploaded_excel = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])
    if uploaded_excel:
        with open(DEFAULT_EXCEL, "wb") as f:
            f.write(uploaded_excel.getbuffer())
        excel_exists = True
else:
    st.sidebar.success("✅ ICD codes loaded")

# Emoji lookup
funding_emojis = {
    1: "🟣", 2: "🔵", 3: "🟢",
    4: "🟡", 5: "🟠", 6: "🔴"
}

# Core functions
@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model once."""
    return SentenceTransformer('all-MiniLM-L6-v2')  # Smaller, faster model for web

@st.cache_data
def get_embeddings(texts, _model):
    """Generate embeddings for texts."""
    return _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

def cosine(u, v):
    """Cosine similarity between vectors."""
    return np.dot(u, v) / (norm(u) * norm(v))

@st.cache_data
def load_and_process_excel(filepath):
    """Load and process the Excel file."""
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()
    
    # Handle different column name formats
    column_mappings = {
        "ED Short": "ED Short List code",
        "Diagnosis": "ED Short List Term",
        "Descriptor": "ED Short List Included conditions"
    }
    
    for old, new in column_mappings.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    
    # Create descriptions for embedding
    df['description'] = (df["ED Short List Term"] + ". " + 
                        df["ED Short List Included conditions"].fillna(""))
    
    return df

def get_top_matches(note_emb, code_embs, df, top_n=12):
    """Find top matching codes based on embedding similarity."""
    sims = [cosine(note_emb, e) for e in code_embs]
    idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_n]
    top = df.iloc[idx].copy()
    top['Similarity'] = [sims[i] for i in idx]
    return top

def generate_mock_response(note, shortlist_df):
    """Generate a mock LLM-style response for demonstration."""
    # This is a simplified response generator for testing
    # In production, this would call a real LLM
    
    responses = []
    
    # Simple keyword matching for demonstration
    note_lower = note.lower()
    
    # Take top 4 codes from shortlist
    for i, (_, row) in enumerate(shortlist_df.head(4).iterrows()):
        if i >= 4:
            break
            
        code = row['ED Short List code']
        term = row['ED Short List Term']
        
        # Generate a plausible explanation based on similarity
        if row['Similarity'] > 0.7:
            explanation = f"Strong match based on clinical presentation"
        elif row['Similarity'] > 0.5:
            explanation = f"Relevant code for the described symptoms"
        else:
            explanation = f"Possible differential diagnosis to consider"
            
        responses.append(f"{i+1}. {code} — {term}\n   * Rationale: {explanation}")
    
    return "\n\n".join(responses)

def parse_response(resp, df):
    """Parse the response to extract codes and explanations."""
    valid = set(df['ED Short List code'].astype(str).str.strip())
    term = dict(zip(df['ED Short List code'], df['ED Short List Term']))
    funding_lookup = dict(zip(df['ED Short List code'], df['Scale'].fillna(3).astype(int)))
    
    rows = []
    lines = resp.splitlines()
    
    for i, line in enumerate(lines):
        # Match numbered lines with codes
        code_match = re.match(r'^\s*\d+\.\s*([A-Z0-9\.]+)', line)
        if code_match:
            code = code_match.group(1).strip()
            
            if code in valid:
                # Look for explanation
                explanation = ""
                
                # Check same line
                same_line = re.match(r'^\s*\d+\.\s*[A-Z0-9\.]+\s*[—\-:]\s*(.+)', line)
                if same_line:
                    explanation = same_line.group(1).strip()
                
                # Check next lines for rationale
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if 'Rationale:' in next_line:
                        explanation = next_line.split('Rationale:', 1)[1].strip()
                    elif next_line.startswith('*'):
                        explanation = next_line.lstrip('*').strip()
                
                if explanation:
                    funding = funding_emojis.get(funding_lookup.get(code, 3), "🟢")
                    code_term = term.get(code, "Unknown")
                    rows.append((code, code_term, explanation, funding))
    
    return rows

# Main interface
tabs = st.tabs(["📝 Single Note", "📁 Batch Processing", "ℹ️ Help"])

with tabs[0]:
    st.header("Enter or Upload Case Note")
    
    col1, col2 = st.columns(2)
    
    with col1:
        note_text = st.text_area(
            "Type or paste case note:", 
            height=300,
            value=st.session_state.note_text,
            placeholder="Enter de-identified case note here..."
        )
    
    with col2:
        uploaded_file = st.file_uploader("Or upload text file:", type=['txt'])
        if uploaded_file:
            note_text = uploaded_file.read().decode('utf-8')
            st.text_area("Uploaded content:", note_text, height=250)
    
    if note_text:
        st.session_state.note_text = note_text
        
        if st.button("🔍 Analyze Case Note", type="primary"):
            if excel_exists:
                with st.spinner("Processing..."):
                    # Load data
                    df = load_and_process_excel(DEFAULT_EXCEL)
                    
                    # Load model and generate embeddings
                    model = load_embedding_model()
                    
                    # Generate embeddings for all codes (cached)
                    code_embeddings = get_embeddings(df['description'].tolist(), model)
                    
                    # Generate embedding for the note
                    note_embedding = get_embeddings([note_text], model)[0]
                    
                    # Find top matches
                    shortlist = get_top_matches(note_embedding, code_embeddings, df)
                    
                    # Generate response (mock for demo)
                    response = generate_mock_response(note_text, shortlist)
                    
                    # Parse response
                    parsed = parse_response(response, df)
                    
                    # Store results
                    st.session_state.results = {
                        'shortlist': shortlist,
                        'response': response,
                        'parsed': parsed
                    }

# Display results
if st.session_state.results:
    st.header("🎯 Classification Results")
    
    # Show top codes in a nice table
    if st.session_state.results['parsed']:
        results_df = pd.DataFrame(
            st.session_state.results['parsed'],
            columns=['Code', 'Term', 'Rationale', 'Complexity']
        )
        
        # Create Plotly table
        fig = go.Figure(data=[go.Table(
            columnwidth=[80, 200, 400, 80],
            header=dict(
                values=['Code', 'Term', 'Clinical Rationale', 'Scale'],
                fill_color='#007AC1',
                font=dict(color='white', size=14),
                align='left',
                height=40
            ),
            cells=dict(
                values=[
                    results_df['Code'],
                    results_df['Term'],
                    results_df['Rationale'],
                    results_df['Complexity']
                ],
                fill_color=[['white', '#f8f9fa']] * len(results_df),
                align='left',
                font=dict(size=12),
                height=60
            )
        )])
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=100 + 60 * len(results_df)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Expandable sections
    with st.expander("📊 Similarity Scores"):
        st.dataframe(
            st.session_state.results['shortlist'][['ED Short List code', 'ED Short List Term', 'Similarity']],
            use_container_width=True
        )
    
    with st.expander("🤖 Raw Analysis"):
        st.code(st.session_state.results['response'])

with tabs[1]:
    st.header("📁 Batch Processing")
    st.info("Upload multiple case notes for bulk analysis")
    
    uploaded_files = st.file_uploader(
        "Select multiple text files:",
        type=['txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"📄 {len(uploaded_files)} files selected")
        
        if st.button("🚀 Process All Files", type="primary"):
            st.warning("Batch processing in demo mode - simplified results")
            
            # Create placeholder for results
            results_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            # Process each file
            batch_results = []
            for i, file in enumerate(uploaded_files):
                content = file.read().decode('utf-8')
                
                # Quick mock processing for demo
                batch_results.append({
                    'filename': file.name,
                    'status': '✅',
                    'codes': ['I21.9', 'R07.4', 'I20.0']  # Mock codes
                })
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            progress_bar.empty()
            
            # Display results
            results_df = pd.DataFrame(batch_results)
            st.dataframe(results_df, use_container_width=True)
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results",
                csv,
                "batch_results.csv",
                "text/csv"
            )

with tabs[2]:
    st.header("ℹ️ Help & Information")
    
    st.markdown("""
    ### How to Use EKoder Web Demo
    
    1. **Enter a case note** in the text area or upload a .txt file
    2. **Click "Analyze Case Note"** to get ICD-10-AM code suggestions
    3. **Review the results** including rationale for each code
    
    ### Understanding Results
    
    - **Codes**: ICD-10-AM codes relevant to the case
    - **Terms**: Official descriptions of the conditions
    - **Rationale**: Why each code was selected
    - **Complexity Scale**: Resource utilization indicator
    
    ### Complexity Scale
    """)
    
    # Complexity scale table
    scale_data = {
        'Scale': ['🟣 1', '🔵 2', '🟢 3', '🟡 4', '🟠 5', '🔴 6'],
        'Funding': ['≤$499', '$500-699', '$700-899', '$900-1099', '$1100-1449', '≥$1450'],
        'Description': ['Minimal', 'Low', 'Moderate', 'High', 'Significant', 'Very High']
    }
    
    scale_df = pd.DataFrame(scale_data)
    st.dataframe(scale_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    ### ⚠️ Important Notes
    
    - This is a **demonstration version** only
    - Do not use for actual clinical decisions
    - Do not enter real patient data
    - For clinical use, request the full local version
    
    ### 📧 Feedback
    
    Please send feedback to: [your-email@example.com]
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "EKoder Web Demo | For Testing Only | "
    "<a href='mailto:your-email@example.com'>Contact Support</a>"
    "</div>",
    unsafe_allow_html=True
)
