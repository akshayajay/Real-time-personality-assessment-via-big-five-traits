import streamlit as st
import nltk
import os
from utils import (
    preprocess_text, process_resumes, nlp, nltk_downloads
)

# Ensure NLTK data is downloaded
nltk_downloads()

# Set page configuration
st.set_page_config(page_title="Resume Shortlisting App", layout="wide")

def main():
    st.title("📄 Resume Shortlisting Application")

    st.markdown("""
    This application allows HR professionals to input detailed job information and upload resumes. It processes the resumes, ranks the candidates based on their suitability for the role, and provides detailed analytics.
    """)

    # Collect detailed job information
    st.header("Job Details")

    job_title = st.text_input("Job Title")

    required_skills_input = st.text_area("Required Skills (comma-separated)", height=100)
    required_skills = [skill.strip().lower() for skill in required_skills_input.split(',') if skill.strip()]

    preferred_skills_input = st.text_area("Preferred Skills (comma-separated)", height=100)
    preferred_skills = [skill.strip().lower() for skill in preferred_skills_input.split(',') if skill.strip()]

    experience_level = st.selectbox("Experience Level Required", options=['Entry Level', 'Mid Level', 'Senior Level', 'Manager', 'Director', 'Executive'])

    education_level = st.selectbox("Education Level Required", options=['High School', 'Associate Degree', "Bachelor's Degree", "Master's Degree", 'Doctorate'])

    job_responsibilities = st.text_area("Job Responsibilities", height=150)

    # Input field for number of candidates to display
    num_candidates = st.number_input("Number of Top Candidates to Display", min_value=1, value=5, step=1)

    # Upload resumes
    st.header("Upload Resumes")
    resumes = st.file_uploader("Upload PDF or DOCX resumes", type=['pdf', 'docx'], accept_multiple_files=True)

    if st.button("Shortlist Candidates"):
        if job_title and required_skills and resumes:
            with st.spinner('Processing...'):
                # Process resumes and compute fitness scores
                candidates = process_resumes(job_title, required_skills, preferred_skills, experience_level, education_level, job_responsibilities, resumes)
                # Save data to session state
                st.session_state['candidates'] = candidates
                st.session_state['num_candidates'] = num_candidates
                # Redirect to results page
                st.experimental_set_query_params(page="Results")
                st.success("Processing complete. Navigate to the 'Results' page to view shortlisted candidates.")
        else:
            st.error("Please provide the job title, required skills, and upload at least one resume.")

if __name__ == '__main__':
    main()
