import streamlit as st
import os
from utils import (
    preprocess_text, process_resumes, nltk_downloads
)

# Ensure NLTK data is downloaded
nltk_downloads()

# Set page configuration
st.set_page_config(page_title="Resume Shortlisting App", layout="wide")


def main():
    st.title("📄 Resume Shortlisting Application")

    st.markdown("""
    This application allows HR professionals to input detailed job information and upload resumes.
    It processes the resumes, ranks the candidates based on their suitability for the role,
    and provides detailed analytics including Big Five personality trait indicators.
    """)

    # Collect detailed job information
    st.header("Job Details")

    job_title = st.text_input("Job Title")

    required_skills_input = st.text_area("Required Skills (comma-separated)", height=100)
    required_skills = [skill.strip().lower() for skill in required_skills_input.split(',') if skill.strip()]

    preferred_skills_input = st.text_area("Preferred Skills (comma-separated)", height=100)
    preferred_skills = [skill.strip().lower() for skill in preferred_skills_input.split(',') if skill.strip()]

    experience_level = st.selectbox(
        "Experience Level Required",
        options=['Entry Level', 'Mid Level', 'Senior Level', 'Manager', 'Director', 'Executive']
    )

    education_level = st.selectbox(
        "Education Level Required",
        options=['High School', 'Associate Degree', "Bachelor's Degree", "Master's Degree", 'Doctorate']
    )

    job_responsibilities = st.text_area("Job Responsibilities", height=150)

    num_candidates = st.number_input(
        "Number of Top Candidates to Display", min_value=1, value=5, step=1
    )

    # Upload resumes
    st.header("Upload Resumes")
    resumes = st.file_uploader(
        "Upload PDF or DOCX resumes",
        type=['pdf', 'docx'],
        accept_multiple_files=True
    )

    if st.button("Shortlist Candidates"):
        if job_title and required_skills and resumes:
            with st.spinner('Processing resumes…'):
                candidates = process_resumes(
                    job_title, required_skills, preferred_skills,
                    experience_level, education_level, job_responsibilities, resumes
                )
                st.session_state['candidates'] = candidates
                st.session_state['num_candidates'] = num_candidates
                # Use current Streamlit API (experimental_set_query_params is deprecated)
                st.query_params["page"] = "Results"
                st.success(
                    "Processing complete. Navigate to the 'Results' page to view shortlisted candidates."
                )
        else:
            st.error("Please provide the job title, required skills, and upload at least one resume.")

    # Show results inline if available
    if 'candidates' in st.session_state and st.session_state['candidates']:
        candidates = st.session_state['candidates']
        n = int(st.session_state.get('num_candidates', 5))

        st.header("🏆 Shortlisted Candidates")
        import pandas as pd

        display_cols = [
            'Candidate Name', 'Total Score', 'Required Skill Match (%)',
            'Preferred Skill Match (%)', 'Experience Match Score',
            'Education Match Score', 'Conscientiousness', 'Agreeableness',
            'Openness', 'Extraversion', 'Neuroticism'
        ]
        df = pd.DataFrame(candidates[:n])[display_cols]
        st.dataframe(df, use_container_width=True)

        st.subheader("Matched Skills per Candidate")
        for c in candidates[:n]:
            with st.expander(c['Candidate Name']):
                st.write(f"**Required skills matched:** {c['Matched Required Skills'] or 'None'}")
                st.write(f"**Preferred skills matched:** {c['Matched Preferred Skills'] or 'None'}")


if __name__ == '__main__':
    main()
