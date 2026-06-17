# Real-Time Personality Assessment via Big Five Traits

[![Tests](https://github.com/akshayajay/Real-time-personality-assessment-via-big-five-traits/actions/workflows/tests.yml/badge.svg)](https://github.com/akshayajay/Real-time-personality-assessment-via-big-five-traits/actions/workflows/tests.yml)

An NLP-powered resume shortlisting application that ranks candidates by job fit and infers Big Five personality traits from resume language.

---

## What it does

Upload multiple resumes (PDF or DOCX) alongside a job description and get:

- **Ranked candidates** by total fitness score
- **Skill matching** — required and preferred skills extracted via spaCy PhraseMatcher
- **Semantic similarity** — dual SentenceTransformer model ensemble (MiniLM + MPNet)
- **Experience and education matching** against the specified role requirements
- **Big Five personality trait indicators** (OCEAN) inferred from resume vocabulary using keyword scoring

---

## Scoring breakdown

| Component | Weight |
|---|---|
| Semantic similarity (SentenceTransformers + TF-IDF) | 40% |
| Required skill match | 40% |
| Preferred skill match | 10% |
| Experience level match | 5% |
| Education level match | 5% |

Big Five traits (Conscientiousness, Agreeableness, Openness, Extraversion, Neuroticism) are reported separately as supplementary indicators — not included in the ranking score.

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/akshayajay/Real-time-personality-assessment-via-big-five-traits.git
cd Real-time-personality-assessment-via-big-five-traits
```

### 2. Install

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Download NLTK data

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"
```

### 4. Run

```bash
streamlit run app.py
```

### 5. Test

```bash
pip install pytest scipy
pytest tests/ -v
```

---

## Project structure

```
├── app.py              # Streamlit UI
├── utils.py            # NLP pipeline (preprocessing, scoring, extraction)
├── tests/
│   └── test_utils.py   # Unit tests (35 tests, no heavy models required)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech stack

`Python` · `Streamlit` · `spaCy` · `SentenceTransformers` · `scikit-learn` · `NLTK` · `pdfminer.six` · `python-docx`

---

## Author

**Akshaya J** · [github.com/akshayajay](https://github.com/akshayajay)
