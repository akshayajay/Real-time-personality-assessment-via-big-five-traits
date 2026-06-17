# Real-Time Personality Assessment via Big Five Traits

[![Tests](https://github.com/akshayajay/Real-time-personality-assessment-via-big-five-traits/actions/workflows/tests.yml/badge.svg)](https://github.com/akshayajay/Real-time-personality-assessment-via-big-five-traits/actions/workflows/tests.yml)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://real-time-personality-assessment-via-big-five-traits.streamlit.app/)

An NLP-powered resume shortlisting application that ranks candidates by job fit and infers Big Five personality traits from resume language.

---

## What it does

Upload multiple resumes (PDF or DOCX) alongside a job description and get:

- **Ranked candidates** by total fitness score
- **Skill matching** via spaCy PhraseMatcher
- **Semantic similarity** via dual SentenceTransformer ensemble (MiniLM + MPNet)
- **Experience and education matching** against role requirements
- **Big Five personality trait indicators** (OCEAN) inferred from resume vocabulary

---

## Scoring breakdown

| Component | Weight |
|---|---|
| Semantic similarity (SentenceTransformers + TF-IDF) | 40% |
| Required skill match | 40% |
| Preferred skill match | 10% |
| Experience level match | 5% |
| Education level match | 5% |

---

## Quickstart

1. Clone the repo and install dependencies:

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"
```

2. Run the app:

```
streamlit run app.py
```

3. Run tests:

```
pip install pytest scipy
pytest tests/ -v
```

---

## Project structure

```
app.py              - Streamlit UI
utils.py            - NLP pipeline
tests/test_utils.py - 35 unit tests
requirements.txt
README.md
```

---

## Tech stack

Python, Streamlit, spaCy, SentenceTransformers, scikit-learn, NLTK, pdfminer.six, python-docx

---

## Author

Akshaya J - github.com/akshayajay
