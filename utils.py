import nltk
import re
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import datetime
from pdfminer.high_level import extract_text as extract_pdf_text
import docx
from spacy.matcher import PhraseMatcher

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load Sentence Transformer models
model_a = SentenceTransformer('all-MiniLM-L6-v2')
model_b = SentenceTransformer('all-mpnet-base-v2')

def nltk_downloads():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove unwanted characters but keep hyphens, plus signs, dots, and hashes
    text = re.sub(r'[^\w\s\-\+\.#]', '', text)
    # Tokenize and lemmatize the text
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into string
    return ' '.join(tokens)

def compute_semantic_similarity(text1, text2, model):
    # Use Sentence Transformer for semantic similarity
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    return cosine_scores.item()

def compute_tfidf_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    cosine_similarity = (vectors[0] @ vectors[1].T).toarray()[0][0]
    return cosine_similarity

def match_skills(resume_skills, job_required_skills, job_preferred_skills):
    # All inputs should be sets of lowercase strings
    required_matches = resume_skills.intersection(set(job_required_skills))
    preferred_matches = resume_skills.intersection(set(job_preferred_skills))
    required_skill_match = len(required_matches) / len(job_required_skills) if job_required_skills else 0
    preferred_skill_match = len(preferred_matches) / len(job_preferred_skills) if job_preferred_skills else 0
    return required_skill_match, preferred_skill_match, required_matches, preferred_matches

def experience_level_match(resume_experience, job_experience_level):
    # Simple mapping for demonstration purposes
    levels = {
        'Entry Level': 1,
        'Mid Level': 3,
        'Senior Level': 5,
        'Manager': 7,
        'Director': 9,
        'Executive': 11
    }
    required_level = levels.get(job_experience_level, 0)
    experience_years = resume_experience if resume_experience else 0
    # Normalize and compute match score
    experience_score = min(experience_years / required_level, 1) if required_level else 0
    return experience_score

def education_level_match(resume_education_level, job_education_level):
    levels = {
        'High School': 1,
        'Associate Degree': 2,
        "Bachelor's Degree": 3,
        "Master's Degree": 4,
        'Doctorate': 5
    }
    required_level = levels.get(job_education_level, 0)
    resume_level = levels.get(resume_education_level, 0)
    return 1 if resume_level >= required_level else 0

def extract_text_from_pdf(file):
    try:
        return extract_pdf_text(file)
    except Exception as e:
        print(f"Error reading PDF file {file.name}: {e}")
        return ''

def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX file {file.name}: {e}")
        return ''

def extract_experience_years(doc):
    # Placeholder function to extract total years of experience
    experience_years = 0
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            match = re.search(r'\b(\d{4})\b', ent.text)
            if match:
                year = int(match.group(1))
                current_year = datetime.datetime.now().year
                if 1900 < year <= current_year:
                    years = current_year - year
                    if 0 < years < 50:
                        experience_years = max(experience_years, years)
    return experience_years

def extract_education_level(doc):
    education_levels = {
        'high school': 'High School',
        'associate': 'Associate Degree',
        "bachelor's": "Bachelor's Degree",
        'bachelor': "Bachelor's Degree",
        'master': "Master's Degree",
        'doctorate': 'Doctorate',
        'phd': 'Doctorate',
        'd.phil': 'Doctorate'
    }
    resume_education_level = 'High School'
    for sent in doc.sents:
        for key in education_levels:
            if key in sent.text.lower():
                resume_education_level = education_levels[key]
                break
    return resume_education_level

# Scoring Framework for Personality Traits with Specific Keywords
class ResumeScorer:
    def __init__(self):
        # Define traits and associated keywords in memory
        self.traits = {
            "Conscientiousness": 0,
            "Agreeableness": 0,
            "Neuroticism": 0,
            "Openness": 0,
            "Extraversion": 0
        }
        self.max_score = 10  # Maximum score for each trait

        # In-memory dictionary to store keywords for each trait
        self.trait_keywords = {
            "Conscientiousness": ["organized", "dependable", "punctual", "disciplined", "responsible", "thorough", "diligent", "detail-oriented", "careful", "meticulous", 
        "reliable", "hardworking", "methodical", "focused", "efficient", 
        "structured", "persistent", "self-controlled", "goal-oriented", 
        "accountable", "precise", "steady", "planner", "cautious", "orderly", 
        "neat", "consistent", "conscientious", "productive", "systematic", 
        "deliberate", "exact", "time-management", "perfectionist", 
        "prepared", "attentive", "accurate", "task-oriented", "committed", 
        "vigilant", "proactive", "punctual", "steadfast", "results-driven", 
        "dedicated", "mindful", "analytical", "strategic", "procedural", 
        "prudent", "assiduous", "punctuality", "restrained", "conscientiousness", 
        "industrious", "observant", "goal-setting", "reliable", 
        "performance-focused", "dutiful", "compliance", "self-disciplined", 
        "principled", "pragmatic", "task-driven", "patient", "moderate", 
        "goal-focused", "composed", "procedural", "rule-abiding", 
        "mature", "restrained", "self-motivated", "tenacious", "responsible", 
        "dependable", "organized", "loyal", "objective", "serious", 
        "restrained", "goal-oriented", "mindful", "measured", "respectful", 
        "planful", "anticipatory", "initiative-driven", "calculated", 
        "exacting", "balanced", "perseverant", "vigilant", "accurate", 
        "cautious", "foresighted", "persevering"],
            "Agreeableness": ["friendly", "cooperative", "empathetic", "kind", "compassionate", 
        "helpful", "understanding", "supportive", "respectful", "patient", 
        "tactful", "considerate", "generous", "warm", "accommodating", 
        "good-natured", "polite", "collaborative", "trustworthy", "forgiving", 
        "courteous", "altruistic", "sensitive", "gentle", "agreeable", 
        "nurturing", "diplomatic", "encouraging", "caring", "open-hearted", 
        "loving", "harmonious", "gracious", "approachable", "loyal", 
        "affectionate", "tolerant", "calming", "trusting", "kindhearted", 
        "likable", "obliging", "team-oriented", "sympathetic", "social", 
        "good-willed", "magnanimous", "charitable", "flexible", "hospitable", 
        "inclusive", "fair", "pleasant", "thoughtful", "companionable", 
        "lenient", "big-hearted", "nonjudgmental", "mediating", "appreciative", 
        "modest", "helpful", "personable", "warm-hearted", "respectful", 
        "team-player", "unselfish", "humble", "listener", "patient", 
        "affirming", "communicative", "engaging", "generous", "peaceable", 
        "benevolent", "heartwarming", "inclusive", "sympathetic", 
        "affectionate", "trusting", "encouraging", "compassionate", 
        "amicable", "hospitable", "conciliatory", "amicable", "lenient", 
        "nonjudgmental", "amiable", "forgiving", "tolerant", "calming", 
        "good-natured", "altruistic", "thoughtful"],
            "Neuroticism": ["anxious", "insecure", "emotional", "moody", "nervous", 
        "self-conscious", "pessimistic", "sensitive", "tense", "stressed", 
        "irritable", "impulsive", "worried", "easily upset", "vulnerable", 
        "prone to guilt", "frustrated", "reactive", "overwhelmed", "fearful", 
        "high-strung", "restless", "indecisive", "apprehensive", "paranoid", 
        "depressed", "melancholic", "unstable", "erratic", "brooding", 
        "guilt-ridden", "discouraged", "self-critical", "oversensitive", 
        "perfectionistic", "hyper-vigilant", "suspicious", "obsessive", 
        "uneasy", "temperamental", "doubtful", "fidgety", "irascible", 
        "jumpy", "indecisive", "unsure", "hyper-aware", "hostile", 
        "unpredictable", "withdrawn", "regretful", "controlling", 
        "agitated", "touchy", "strained", "envious", "jealous", "impatient", 
        "self-absorbed", "dissatisfied", "jittery", "short-tempered", 
        "pressured", "cautious", "diffident", "mopey", "edgy", "fretful", 
        "paranoid", "complaining", "melodramatic", "dependent", "critical", 
        "compulsive", "overreacting", "defensive", "sulking", "mistrustful", 
        "clingy", "excitable", "guilt-laden", "volatile", "overthinking", 
        "irritable", "insecure", "hostile", "regretful", "touchy", 
        "paranoid", "restless", "fidgety", "moody", "tense"],
            "Openness": ["creative", "curious", "imaginative", "adventurous", "intellectual", 
        "innovative", "insightful", "artistic", "experimental", "unconventional", 
        "open-minded", "broad-minded", "introspective", "forward-thinking", 
        "flexible", "analytical", "philosophical", "thoughtful", "visionary", 
        "abstract", "inventive", "adaptable", "progressive", "nontraditional", 
        "unorthodox", "independent", "inspired", "speculative", "aesthetic", 
        "divergent", "reflective", "experimental", "idea-driven", "visionary", 
        "receptive", "abstract", "theoretical", "deep-thinking", "questioning", 
        "future-focused", "change-ready", "spontaneous", "aesthetic-minded", 
        "learning-oriented", "broad-minded", "risk-taking", "self-reflective", 
        "exploring", "art-loving", "intellectual", "philosophical", 
        "idealistic", "vision-oriented", "culture-absorbing", "dynamic", 
        "strategic", "conceptual", "novelty-seeking", "perceptive", 
        "introspective", "imaginative", "inventive", "inquisitive", 
        "critical", "broad-based", "modernist", "speculative", "divergent", 
        "explorative", "idealistic", "analytical", "deep-thinking", 
        "conceptual", "intellectual", "innovative"],
            "Extraversion": ["outgoing", "talkative", "energetic", "sociable", "assertive", 
        "confident", "enthusiastic", "expressive", "charismatic", 
        "gregarious", "lively", "spontaneous", "approachable", 
        "people-oriented", "positive", "adventurous", "action-oriented", 
        "engaging", "cheerful", "communicative", "participatory", 
        "dominant", "friendly", "dynamic", "spirited", "expressive", 
        "social", "charming", "outspoken", "self-assured", "vibrant", 
        "motivational", "talkative", "leadership-driven", "connected", 
        "stimulating", "demonstrative", "adventurous", "goal-oriented", 
        "persuasive", "involved", "persuasive", "center of attention", 
        "crowd-friendly", "animated", "life of the party", "captivating", 
        "bubbly", "exciting", "forceful", "persuasive", "radiant", 
        "leader-like", "gregarious", "charming", "outspoken", "bold", 
        "self-assured", "confident", "passionate", "charismatic", 
        "dynamic", "vivacious", "influential", "people-centric", 
        "candid", "enthusiastic", "self-affirming", "impactful", 
        "vivacious", "quick-witted", "affable", "talkative"]
        }

    def get_keywords_for_trait(self, trait_name):
        # Fetch the keywords from the in-memory dictionary
        return self.trait_keywords.get(trait_name, [])

    def score_trait(self, trait, score_increment=1):
        if trait in self.traits:
            self.traits[trait] += score_increment
            # Normalize the score to be between 0 and 10
            self.traits[trait] = min(self.traits[trait], self.max_score)

    def evaluate_resume(self, resume_text):
        # Convert resume text to lowercase for case-insensitive matching
        resume_text = resume_text.lower()

        # Get the trait keywords for scoring
        keyword_map = {
            "Conscientiousness": self.get_keywords_for_trait("Conscientiousness"),
            "Agreeableness": self.get_keywords_for_trait("Agreeableness"),
            "Neuroticism": self.get_keywords_for_trait("Neuroticism"),
            "Openness": self.get_keywords_for_trait("Openness"),
            "Extraversion": self.get_keywords_for_trait("Extraversion")
        }

        # Check for keywords in the resume text
        for trait, keywords in keyword_map.items():
            for keyword in keywords:
                # Count how many times each keyword appears in the resume
                keyword_count = resume_text.count(keyword)
                if keyword_count > 0:
                    self.score_trait(trait, score_increment=keyword_count)

        return self.traits


def process_resumes(job_title, required_skills, preferred_skills, experience_level, education_level, job_responsibilities, resumes):
    # Normalize required and preferred skills
    required_skills = [skill.lower() for skill in required_skills]
    preferred_skills = [skill.lower() for skill in preferred_skills]

    # Combine job title and responsibilities
    job_description = job_title + ' ' + job_responsibilities
    # Preprocess job description
    job_desc_processed = preprocess_text(job_description)

    # Create patterns for skill extraction
    required_skills_patterns = [nlp.make_doc(skill) for skill in required_skills]
    preferred_skills_patterns = [nlp.make_doc(skill) for skill in preferred_skills]
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')  # Use 'LOWER' attribute for case-insensitive matching
    matcher.add("RequiredSkill", required_skills_patterns)
    matcher.add("PreferredSkill", preferred_skills_patterns)

    # Initialize the ResumeScorer
    scorer = ResumeScorer()

    candidates = []
    for resume_file in resumes:
        # Extract text from resume
        if resume_file.name.endswith('.pdf'):
            resume_text = extract_text_from_pdf(resume_file)
        elif resume_file.name.endswith('.docx'):
            resume_text = extract_text_from_docx(resume_file)
        else:
            resume_text = resume_file.read().decode('utf-8', errors='ignore')

        # Preprocess resume text
        resume_processed = preprocess_text(resume_text)

        # Create spaCy document
        doc = nlp(resume_text)

        # Extract skills
        resume_skills = set()
        matches = matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            resume_skills.add(span.text.lower())

        # Skills matching
        required_skill_match, preferred_skill_match, required_matches, preferred_matches = match_skills(
            resume_skills, required_skills, preferred_skills
        )

        # Extract experience
        resume_experience = extract_experience_years(doc)
        experience_match_score = experience_level_match(resume_experience, experience_level)

        # Extract education level
        resume_education_level = extract_education_level(doc)
        education_match_score = education_level_match(resume_education_level, education_level)

        # Compute semantic similarity using multiple models
        similarity_score_a = compute_semantic_similarity(job_desc_processed, resume_processed, model_a)
        similarity_score_b = compute_semantic_similarity(job_desc_processed, resume_processed, model_b)
        tfidf_similarity_score = compute_tfidf_similarity(job_desc_processed, resume_processed)

        # Calculate trait-based scores using ResumeScorer
        trait_scores = scorer.evaluate_resume(resume_processed)

        # Aggregate similarity scores using weights
        weight_a = 0.4
        weight_b = 0.4
        weight_tfidf = 0.2

        combined_similarity_score = (
            (similarity_score_a * weight_a) +
            (similarity_score_b * weight_b) +
            (tfidf_similarity_score * weight_tfidf)
        ) / (weight_a + weight_b + weight_tfidf)

        # Calculate total score (weights can be adjusted)
        total_score = (
            (combined_similarity_score * 0.4) +
            (required_skill_match * 0.4) +
            (preferred_skill_match * 0.1) +
            (experience_match_score * 0.05) +
            (education_match_score * 0.05)
        )

        candidates.append({
            'Candidate Name': resume_file.name,
            'Total Score': round(total_score * 100, 2),
            'Combined Similarity Score': round(combined_similarity_score * 100, 2),
            'Similarity Score A': round(similarity_score_a * 100, 2),
            'Similarity Score B': round(similarity_score_b * 100, 2),
            'TF-IDF Similarity Score': round(tfidf_similarity_score * 100, 2),
            'Required Skill Match (%)': round(required_skill_match * 100, 2),
            'Preferred Skill Match (%)': round(preferred_skill_match * 100, 2),
            'Experience Match Score': round(experience_match_score * 100, 2),
            'Education Match Score': round(education_match_score * 100, 2),
            'Matched Required Skills': ', '.join(required_matches),
            'Matched Preferred Skills': ', '.join(preferred_matches),
            'Resume Text': resume_processed,  # For further processing
            'Conscientiousness': trait_scores['Conscientiousness'],
            'Agreeableness': trait_scores['Agreeableness'],
            'Neuroticism': trait_scores['Neuroticism'],
            'Openness': trait_scores['Openness'],
            'Extraversion': trait_scores['Extraversion'],
        })

    # Sort candidates by total score
    candidates_sorted = sorted(candidates, key=lambda x: x['Total Score'], reverse=True)
    return candidates_sorted
