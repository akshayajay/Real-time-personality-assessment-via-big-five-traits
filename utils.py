import nltk
import re
import os
import datetime
from typing import List, Set, Tuple, Dict, Optional

# ---------------------------------------------------------------------------
# Lazy-loaded heavy dependencies (spaCy + SentenceTransformers)
# Loaded on first use so tests can import this module without them installed.
# ---------------------------------------------------------------------------

_nlp = None
_model_a = None
_model_b = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load('en_core_web_sm')
    return _nlp


def _get_model_a():
    global _model_a
    if _model_a is None:
        from sentence_transformers import SentenceTransformer
        _model_a = SentenceTransformer('all-MiniLM-L6-v2')
    return _model_a


def _get_model_b():
    global _model_b
    if _model_b is None:
        from sentence_transformers import SentenceTransformer
        _model_b = SentenceTransformer('all-mpnet-base-v2')
    return _model_b


# Keep a module-level alias for backward compatibility with app.py imports
def nlp(text):
    return _get_nlp()(text)


def nltk_downloads():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Lowercase, clean, tokenise, and lemmatise input text."""
    text = text.lower()
    text = re.sub(r'[^\w\s\-\+\.#]', '', text)
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def compute_semantic_similarity(text1: str, text2: str, model) -> float:
    """Cosine similarity between two texts using a SentenceTransformer model."""
    from sentence_transformers import util
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    return util.cos_sim(embeddings1, embeddings2).item()


def compute_tfidf_similarity(text1: str, text2: str) -> float:
    """TF-IDF cosine similarity between two texts."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return float((vectors[0] @ vectors[1].T).toarray()[0][0])


# ---------------------------------------------------------------------------
# Skill matching
# ---------------------------------------------------------------------------

def match_skills(
    resume_skills: Set[str],
    job_required_skills: List[str],
    job_preferred_skills: List[str],
) -> Tuple[float, float, Set[str], Set[str]]:
    """
    Compute required and preferred skill match ratios.

    Returns:
        (required_ratio, preferred_ratio, required_matches, preferred_matches)
    """
    required_matches = resume_skills.intersection(set(job_required_skills))
    preferred_matches = resume_skills.intersection(set(job_preferred_skills))
    required_ratio = len(required_matches) / len(job_required_skills) if job_required_skills else 0.0
    preferred_ratio = len(preferred_matches) / len(job_preferred_skills) if job_preferred_skills else 0.0
    return required_ratio, preferred_ratio, required_matches, preferred_matches


# ---------------------------------------------------------------------------
# Experience and education matching
# ---------------------------------------------------------------------------

EXPERIENCE_LEVEL_YEARS = {
    'Entry Level': 1,
    'Mid Level': 3,
    'Senior Level': 5,
    'Manager': 7,
    'Director': 9,
    'Executive': 11,
}

EDUCATION_LEVEL_RANK = {
    'High School': 1,
    'Associate Degree': 2,
    "Bachelor's Degree": 3,
    "Master's Degree": 4,
    'Doctorate': 5,
}


def experience_level_match(resume_experience: Optional[int], job_experience_level: str) -> float:
    """
    Return a [0, 1] score representing how well the candidate's experience
    matches the required level.
    """
    required = EXPERIENCE_LEVEL_YEARS.get(job_experience_level, 0)
    years = resume_experience or 0
    if required == 0:
        return 0.0
    return min(years / required, 1.0)


def education_level_match(resume_education_level: str, job_education_level: str) -> float:
    """Return 1.0 if resume meets or exceeds required education level, else 0.0."""
    required = EDUCATION_LEVEL_RANK.get(job_education_level, 0)
    actual = EDUCATION_LEVEL_RANK.get(resume_education_level, 0)
    return 1.0 if actual >= required else 0.0


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file) -> str:
    try:
        from pdfminer.high_level import extract_text as _extract
        return _extract(file)
    except Exception as e:
        print(f"Error reading PDF {getattr(file, 'name', file)}: {e}")
        return ''


def extract_text_from_docx(file) -> str:
    try:
        import docx as _docx
        doc = _docx.Document(file)
        return '\n'.join(para.text for para in doc.paragraphs)
    except Exception as e:
        print(f"Error reading DOCX {getattr(file, 'name', file)}: {e}")
        return ''


# ---------------------------------------------------------------------------
# spaCy-based extractors
# ---------------------------------------------------------------------------

def extract_experience_years(doc) -> int:
    """Heuristic: find the earliest relevant year mentioned and compute years since."""
    experience_years = 0
    current_year = datetime.datetime.now().year
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            match = re.search(r'\b(\d{4})\b', ent.text)
            if match:
                year = int(match.group(1))
                if 1900 < year <= current_year:
                    years = current_year - year
                    if 0 < years < 50:
                        experience_years = max(experience_years, years)
    return experience_years


def extract_education_level(doc) -> str:
    """Return the highest education level mentioned in the spaCy doc."""
    education_map = {
        'high school': 'High School',
        'associate': 'Associate Degree',
        "bachelor's": "Bachelor's Degree",
        'bachelor': "Bachelor's Degree",
        'master': "Master's Degree",
        'doctorate': 'Doctorate',
        'phd': 'Doctorate',
        'd.phil': 'Doctorate',
    }
    found = 'High School'
    for sent in doc.sents:
        text_lower = sent.text.lower()
        for key, level in education_map.items():
            if key in text_lower:
                # Keep the highest rank found
                if EDUCATION_LEVEL_RANK.get(level, 0) > EDUCATION_LEVEL_RANK.get(found, 0):
                    found = level
    return found


# ---------------------------------------------------------------------------
# Big Five personality trait scorer
# ---------------------------------------------------------------------------

class ResumeScorer:
    """
    Keyword-based Big Five personality trait scorer.

    Create a new instance per resume — do NOT reuse across resumes,
    as trait scores are accumulated incrementally.
    """

    MAX_SCORE = 10

    TRAIT_KEYWORDS: Dict[str, List[str]] = {
        "Conscientiousness": [
            "organized", "dependable", "punctual", "disciplined", "responsible",
            "thorough", "diligent", "detail-oriented", "careful", "meticulous",
            "reliable", "hardworking", "methodical", "focused", "efficient",
            "structured", "persistent", "self-controlled", "goal-oriented",
            "accountable", "precise", "systematic", "deliberate", "exact",
            "time-management", "prepared", "attentive", "accurate", "committed",
            "proactive", "dedicated", "analytical", "strategic", "industrious",
        ],
        "Agreeableness": [
            "friendly", "cooperative", "empathetic", "kind", "compassionate",
            "helpful", "understanding", "supportive", "respectful", "patient",
            "tactful", "considerate", "generous", "warm", "accommodating",
            "good-natured", "polite", "collaborative", "trustworthy", "forgiving",
            "courteous", "altruistic", "sensitive", "gentle", "nurturing",
            "diplomatic", "encouraging", "caring", "approachable", "tolerant",
            "sympathetic", "team-oriented", "humble", "peaceable", "benevolent",
        ],
        "Neuroticism": [
            "anxious", "insecure", "emotional", "moody", "nervous",
            "self-conscious", "pessimistic", "tense", "stressed", "irritable",
            "impulsive", "worried", "vulnerable", "frustrated", "reactive",
            "overwhelmed", "fearful", "restless", "indecisive", "apprehensive",
            "depressed", "unstable", "erratic", "self-critical", "obsessive",
            "uneasy", "temperamental", "doubtful", "hostile", "unpredictable",
        ],
        "Openness": [
            "creative", "curious", "imaginative", "adventurous", "intellectual",
            "innovative", "insightful", "artistic", "experimental", "open-minded",
            "broad-minded", "introspective", "forward-thinking", "flexible",
            "philosophical", "visionary", "inventive", "adaptable", "progressive",
            "divergent", "reflective", "learning-oriented", "risk-taking",
            "inquisitive", "conceptual", "dynamic",
        ],
        "Extraversion": [
            "outgoing", "talkative", "energetic", "sociable", "assertive",
            "confident", "enthusiastic", "expressive", "charismatic", "gregarious",
            "lively", "approachable", "people-oriented", "positive", "engaging",
            "cheerful", "communicative", "dominant", "friendly", "dynamic",
            "spirited", "charming", "outspoken", "self-assured", "vibrant",
            "motivational", "persuasive", "bold", "passionate", "influential",
        ],
    }

    def __init__(self):
        self.traits: Dict[str, float] = {t: 0.0 for t in self.TRAIT_KEYWORDS}

    def score_trait(self, trait: str, increment: float = 1.0) -> None:
        if trait in self.traits:
            self.traits[trait] = min(self.traits[trait] + increment, self.MAX_SCORE)

    def evaluate_resume(self, resume_text: str) -> Dict[str, float]:
        """
        Score Big Five traits from resume text.

        Args:
            resume_text: Pre-processed (lowercased) resume text.

        Returns:
            Dict mapping trait name → score (0–10).
        """
        text = resume_text.lower()
        for trait, keywords in self.TRAIT_KEYWORDS.items():
            for keyword in keywords:
                count = text.count(keyword)
                if count > 0:
                    self.score_trait(trait, increment=float(count))
        return dict(self.traits)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_resumes(
    job_title: str,
    required_skills: List[str],
    preferred_skills: List[str],
    experience_level: str,
    education_level: str,
    job_responsibilities: str,
    resumes,
) -> List[Dict]:
    """
    Process a list of resume files and return candidates ranked by total score.
    """
    from spacy.matcher import PhraseMatcher

    required_skills = [s.lower() for s in required_skills]
    preferred_skills = [s.lower() for s in preferred_skills]

    job_description = f"{job_title} {job_responsibilities}"
    job_desc_processed = preprocess_text(job_description)

    _nlp_model = _get_nlp()

    required_patterns = [_nlp_model.make_doc(s) for s in required_skills]
    preferred_patterns = [_nlp_model.make_doc(s) for s in preferred_skills]
    matcher = PhraseMatcher(_nlp_model.vocab, attr='LOWER')
    matcher.add("RequiredSkill", required_patterns)
    matcher.add("PreferredSkill", preferred_patterns)

    model_a = _get_model_a()
    model_b = _get_model_b()

    candidates = []

    for resume_file in resumes:
        # --- Extract text ---
        name = getattr(resume_file, 'name', str(resume_file))
        if name.endswith('.pdf'):
            resume_text = extract_text_from_pdf(resume_file)
        elif name.endswith('.docx'):
            resume_text = extract_text_from_docx(resume_file)
        else:
            resume_text = resume_file.read().decode('utf-8', errors='ignore')

        resume_processed = preprocess_text(resume_text)
        doc = _nlp_model(resume_text)

        # --- Skills ---
        resume_skills: Set[str] = set()
        for _, start, end in matcher(doc):
            resume_skills.add(doc[start:end].text.lower())

        req_match, pref_match, req_matched, pref_matched = match_skills(
            resume_skills, required_skills, preferred_skills
        )

        # --- Experience & education ---
        exp_years = extract_experience_years(doc)
        exp_score = experience_level_match(exp_years, experience_level)
        edu_level = extract_education_level(doc)
        edu_score = education_level_match(edu_level, education_level)

        # --- Semantic similarity ---
        sim_a = compute_semantic_similarity(job_desc_processed, resume_processed, model_a)
        sim_b = compute_semantic_similarity(job_desc_processed, resume_processed, model_b)
        tfidf_sim = compute_tfidf_similarity(job_desc_processed, resume_processed)

        combined_sim = (sim_a * 0.4 + sim_b * 0.4 + tfidf_sim * 0.2)

        # --- Big Five — NEW SCORER PER RESUME (bug fix) ---
        scorer = ResumeScorer()
        trait_scores = scorer.evaluate_resume(resume_processed)

        # --- Total score ---
        total = (
            combined_sim * 0.40 +
            req_match    * 0.40 +
            pref_match   * 0.10 +
            exp_score    * 0.05 +
            edu_score    * 0.05
        )

        candidates.append({
            'Candidate Name':              name,
            'Total Score':                 round(total * 100, 2),
            'Combined Similarity Score':   round(combined_sim * 100, 2),
            'Similarity Score A':          round(sim_a * 100, 2),
            'Similarity Score B':          round(sim_b * 100, 2),
            'TF-IDF Similarity Score':     round(tfidf_sim * 100, 2),
            'Required Skill Match (%)':    round(req_match * 100, 2),
            'Preferred Skill Match (%)':   round(pref_match * 100, 2),
            'Experience Match Score':      round(exp_score * 100, 2),
            'Education Match Score':       round(edu_score * 100, 2),
            'Matched Required Skills':     ', '.join(sorted(req_matched)),
            'Matched Preferred Skills':    ', '.join(sorted(pref_matched)),
            'Resume Text':                 resume_processed,
            **{t: v for t, v in trait_scores.items()},
        })

    return sorted(candidates, key=lambda x: x['Total Score'], reverse=True)
