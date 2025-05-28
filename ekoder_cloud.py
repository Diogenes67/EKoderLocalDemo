import streamlit as st
import pandas as pd

st.set_page_config(page_title="EKoder Demo", page_icon="🏥", layout="wide")

st.title("🏥 EKoder Demo - Testing Version")
st.warning("⚠️ Simplified demo for testing. Do not use real patient data.")

# Simple interface
note = st.text_area("Enter case note:", height=200)

if st.button("Analyze", type="primary"):
    if note:
        st.success("✅ Analysis complete!")
        
        # Mock results
        data = {
            'Code': ['I21.0', 'R07.4', 'I20.0'],
            'Description': [
                'Acute myocardial infarction',
                'Chest pain, unspecified', 
                'Unstable angina'
            ],
            'Confidence': ['High', 'Medium', 'Medium']
        }
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    else:
        st.error("Please enter a case note")

st.info("For full features, download the local version.")
