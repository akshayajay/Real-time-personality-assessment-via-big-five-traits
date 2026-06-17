"""
tests/test_utils.py
-------------------
Unit tests for pure utility functions in utils.py.

No spaCy, SentenceTransformers, or NLTK downloads required —
heavy imports are lazy-loaded and not triggered by these tests.
"""

import sys
import os
import types
import pytest

# Stub heavy optional packages before importing utils
for _mod in ("spacy", "sentence_transformers", "pdfminer", "pdfminer.high_level",
             "docx", "sklearn", "sklearn.feature_extraction",
             "sklearn.feature_extraction.text"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# Stub TfidfVectorizer inside sklearn
import numpy as np
from scipy.sparse import csr_matrix

class _FakeTfidf:
    def fit_transform(self, texts):
        # Return simple one-hot sparse matrices for testing
        vocab = list({w for t in texts for w in t.split()})
        rows = []
        for text in texts:
            row = [1.0 if w in text.split() else 0.0 for w in vocab]
            rows.append(row)
        arr = np.array(rows)
        return csr_matrix(arr)

_sklearn_fe = sys.modules["sklearn.feature_extraction"]
_sklearn_fe_text = sys.modules["sklearn.feature_extraction.text"]
_sklearn_fe_text.TfidfVectorizer = _FakeTfidf

# Stub nltk minimally
import nltk as _nltk_real
# nltk is likely already installed; just make sure downloads are no-ops
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import (
    preprocess_text,
    match_skills,
    experience_level_match,
    education_level_match,
    ResumeScorer,
    EXPERIENCE_LEVEL_YEARS,
    EDUCATION_LEVEL_RANK,
    compute_tfidf_similarity,
)


# ---------------------------------------------------------------------------
# preprocess_text()
# ---------------------------------------------------------------------------

class TestPreprocessText:
    def test_lowercases_text(self):
        result = preprocess_text("Hello World")
        assert result == result.lower()

    def test_removes_special_characters(self):
        result = preprocess_text("Hello, World! @#$%")
        assert "@" not in result
        assert "!" not in result
        assert "," not in result

    def test_keeps_hyphens(self):
        result = preprocess_text("well-organized")
        assert "-" in result

    def test_keeps_hash(self):
        result = preprocess_text("C# developer")
        assert "#" in result

    def test_empty_string(self):
        result = preprocess_text("")
        assert result == ""

    def test_returns_string(self):
        assert isinstance(preprocess_text("some text"), str)


# ---------------------------------------------------------------------------
# match_skills()
# ---------------------------------------------------------------------------

class TestMatchSkills:
    def test_perfect_required_match(self):
        req_ratio, _, _, _ = match_skills(
            {"python", "sql"}, ["python", "sql"], []
        )
        assert req_ratio == 1.0

    def test_partial_required_match(self):
        req_ratio, _, _, _ = match_skills(
            {"python"}, ["python", "sql"], []
        )
        assert req_ratio == pytest.approx(0.5)

    def test_no_required_match(self):
        req_ratio, _, _, _ = match_skills(
            {"java"}, ["python", "sql"], []
        )
        assert req_ratio == 0.0

    def test_empty_required_skills_returns_zero(self):
        req_ratio, _, _, _ = match_skills({"python"}, [], [])
        assert req_ratio == 0.0

    def test_preferred_skill_match(self):
        _, pref_ratio, _, _ = match_skills(
            {"tableau"}, [], ["tableau", "powerbi"]
        )
        assert pref_ratio == pytest.approx(0.5)

    def test_returns_matched_sets(self):
        _, _, req_matched, pref_matched = match_skills(
            {"python", "excel"}, ["python", "sql"], ["excel"]
        )
        assert "python" in req_matched
        assert "excel" in pref_matched
        assert "sql" not in req_matched

    def test_case_sensitivity(self):
        # match_skills expects all lowercase inputs
        req_ratio, _, _, _ = match_skills(
            {"python"}, ["python"], []
        )
        assert req_ratio == 1.0


# ---------------------------------------------------------------------------
# experience_level_match()
# ---------------------------------------------------------------------------

class TestExperienceLevelMatch:
    def test_exact_match_returns_one(self):
        score = experience_level_match(3, "Mid Level")
        assert score == pytest.approx(1.0)

    def test_exceeds_required_capped_at_one(self):
        score = experience_level_match(10, "Entry Level")
        assert score == pytest.approx(1.0)

    def test_zero_experience_returns_zero(self):
        score = experience_level_match(0, "Senior Level")
        assert score == pytest.approx(0.0)

    def test_none_experience_returns_zero(self):
        score = experience_level_match(None, "Mid Level")
        assert score == pytest.approx(0.0)

    def test_partial_experience(self):
        score = experience_level_match(1, "Mid Level")   # requires 3 years
        assert score == pytest.approx(1 / 3)

    def test_unknown_level_returns_zero(self):
        score = experience_level_match(5, "Unknown Level")
        assert score == pytest.approx(0.0)

    def test_all_levels_covered(self):
        for level in EXPERIENCE_LEVEL_YEARS:
            score = experience_level_match(EXPERIENCE_LEVEL_YEARS[level], level)
            assert score == pytest.approx(1.0), f"Failed for level: {level}"


# ---------------------------------------------------------------------------
# education_level_match()
# ---------------------------------------------------------------------------

class TestEducationLevelMatch:
    def test_exact_match_returns_one(self):
        assert education_level_match("Bachelor's Degree", "Bachelor's Degree") == 1.0

    def test_exceeds_required_returns_one(self):
        assert education_level_match("Doctorate", "Bachelor's Degree") == 1.0

    def test_below_required_returns_zero(self):
        assert education_level_match("High School", "Master's Degree") == 0.0

    def test_unknown_resume_level_returns_zero(self):
        assert education_level_match("Trade Certificate", "Bachelor's Degree") == 0.0

    def test_unknown_job_level_returns_one(self):
        # rank 0 means requirement is met by any level >= 0
        assert education_level_match("High School", "Unknown") == 1.0

    def test_all_levels_self_match(self):
        for level in EDUCATION_LEVEL_RANK:
            assert education_level_match(level, level) == 1.0


# ---------------------------------------------------------------------------
# ResumeScorer
# ---------------------------------------------------------------------------

class TestResumeScorer:
    def test_all_five_traits_present(self):
        scorer = ResumeScorer()
        traits = scorer.evaluate_resume("no keywords here")
        assert set(traits.keys()) == {
            "Conscientiousness", "Agreeableness", "Neuroticism", "Openness", "Extraversion"
        }

    def test_conscientiousness_keyword_scores(self):
        scorer = ResumeScorer()
        traits = scorer.evaluate_resume("organized and reliable and efficient")
        assert traits["Conscientiousness"] > 0

    def test_no_keywords_all_zero(self):
        scorer = ResumeScorer()
        traits = scorer.evaluate_resume("xyzzy plugh blorb")
        assert all(v == 0.0 for v in traits.values())

    def test_score_capped_at_ten(self):
        scorer = ResumeScorer()
        # Feed many conscientiousness keywords
        text = " ".join(["organized"] * 50)
        traits = scorer.evaluate_resume(text)
        assert traits["Conscientiousness"] <= 10.0

    def test_fresh_scorer_per_resume(self):
        """Each ResumeScorer instance starts at zero — scores don't leak between resumes."""
        scorer1 = ResumeScorer()
        scorer1.evaluate_resume("creative curious imaginative")

        scorer2 = ResumeScorer()
        traits2 = scorer2.evaluate_resume("no keywords here xyz")
        assert traits2["Openness"] == 0.0

    def test_score_trait_directly(self):
        scorer = ResumeScorer()
        scorer.score_trait("Openness", 5.0)
        assert scorer.traits["Openness"] == 5.0

    def test_unknown_trait_ignored(self):
        scorer = ResumeScorer()
        scorer.score_trait("UnknownTrait", 3.0)  # should not raise
        assert "UnknownTrait" not in scorer.traits

    def test_neuroticism_keyword(self):
        scorer = ResumeScorer()
        traits = scorer.evaluate_resume("anxious stressed overwhelmed")
        assert traits["Neuroticism"] > 0

    def test_extraversion_keyword(self):
        scorer = ResumeScorer()
        traits = scorer.evaluate_resume("outgoing sociable enthusiastic confident")
        assert traits["Extraversion"] > 0
