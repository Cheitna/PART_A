# ======================================
# app.py
# ======================================
import streamlit as st
import pickle
from nltk.metrics.distance import edit_distance
import spacy
import subprocess
import sys

# -----------------------------
# Load spaCy (POS, lemma, dependency) safely for cloud
# -----------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Model not found, download it
    st.info("Downloading spaCy 'en_core_web_sm' model. This may take a few seconds...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")
# -----------------------------
# Load spaCy (POS, lemma, dependency)
# -----------------------------
nlp = spacy.load("en_core_web_sm")
# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Spelling & Grammar Correction System",
    page_icon="✍️",
    layout="centered"
)
# -----------------------------
# Constants
# -----------------------------
BE_VERBS = {"am", "is", "are", "was", "were"}
HAS_VERBS = {"has", "have", "had"}
DO_VERBS = {"do", "does", "did"}
AUX_VERBS = BE_VERBS | HAS_VERBS | DO_VERBS

FUNCTION_WORDS = {
    "no","this","that","which","who","whom","whose",
    "it","they","we","he","she",
    "a","an","the",
    "and","or","but",
    "of","in","on","for","to","with","by","at",
    "not","any","already","very","soon","absolutely"
}

IRREGULAR_PAST = {
    "be": "been",
    "have": "had",
    "do": "done",
    "go": "gone",
    "come": "come",
    "rise": "risen",
    "use": "used",
    "see": "seen",
    "move": "moved",
    "start": "started"
}

PAST_PARTICIPLES = set(IRREGULAR_PAST.values())
# -----------------------------
# Load corpus models
# -----------------------------
@st.cache_resource
def load_models():
    with open("vocabulary.txt", "r", encoding="utf-8") as f:
        vocab = set(w.lower() for w in f.read().splitlines())
    with open("word_freq.pkl", "rb") as f:
        word_freq = pickle.load(f)
    with open("bigram_counts.pkl", "rb") as f:
        bigram_counts = pickle.load(f)
    with open("unigram_counts.pkl", "rb") as f:
        unigram_counts = pickle.load(f)

    vocab |= FUNCTION_WORDS | AUX_VERBS | PAST_PARTICIPLES
    total_unigrams = sum(unigram_counts.values())

    return vocab, word_freq, bigram_counts, unigram_counts, total_unigrams

VOCAB, WORD_FREQ, BIGRAM_COUNTS, UNIGRAM_COUNTS, TOTAL_UNIGRAMS = load_models()
VOCAB_SIZE = len(VOCAB)

# -----------------------------
# Helper functions
# -----------------------------
def bigram_prob(w1, w2):
    return (BIGRAM_COUNTS.get((w1, w2), 0) + 1) / (UNIGRAM_COUNTS.get(w1, 0) + VOCAB_SIZE)

def to_past_participle(lemma):
    if lemma in IRREGULAR_PAST:
        return IRREGULAR_PAST[lemma]
    if lemma.endswith("e"):
        return lemma + "d"
    return lemma + "ed"

# -----------------------------
# Preprocess text (keep punctuation)
# -----------------------------
def preprocess_text(text):
    doc = nlp(text)
    surface_tokens = [t.text for t in doc]  # keep punctuation
    lemma_tokens = [t.lemma_ for t in doc]
    alpha_indices = [i for i, t in enumerate(doc) if t.is_alpha]
    return surface_tokens, lemma_tokens, alpha_indices, doc

# -----------------------------
# Candidate generation
# -----------------------------
def generate_candidates(word, lemma=None):
    candidates = set()
    if lemma and lemma in VOCAB:
        candidates.add(lemma)

    for v in VOCAB:
        if edit_distance(word.lower(), v) <= 2:
            candidates.add(v)
        if v.endswith("s") and edit_distance(word.lower(), v[:-1]) <= 1:
            candidates.add(v)
        if word.endswith("s") and word[:-1] == v:
            candidates.add(v)
    return list(candidates)

def rank_candidates(candidates, prev_word, original_word):
    ranked = []
    for c in candidates:
        score = WORD_FREQ.get(c, 0) / TOTAL_UNIGRAMS
        if prev_word:
            score += bigram_prob(prev_word, c)

        dist = edit_distance(original_word.lower(), c)
        score += max(0, 1 - dist / 5)

        ranked.append({
            "word": c,
            "edit_distance": dist,
            "score": score
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:5]

# -----------------------------
# Grammar correction
# -----------------------------
# -----------------------------
# Grammar correction (STRICT & ROBUST)
# -----------------------------
def apply_grammar(surface_tokens, doc):
    """
    Apply grammar rules:
    1. HAS/HAVE/HAD + VB -> VBN
    2. BE + VB -> VBG
    3. Subject-verb agreement (plural subjects)
    
    Returns:
        corrected: list of tokens (with punctuation)
        grammar_indices: indices of corrected grammar tokens
        grammar_map: {index: (original, corrected)}
    """
    corrected = surface_tokens[:]
    grammar_indices = []
    grammar_map = {}

    for token in doc:
        # Only consider verbs
        if token.pos_ != "VERB":
            continue

        idx = token.i
        lemma = token.lemma_.lower()
        text = token.text
        prev_word = corrected[idx-1].lower() if idx > 0 else None

        # --- HAS/HAVE/HAD + VB -> VBN ---
        if prev_word in HAS_VERBS and lemma not in AUX_VERBS:
            new_form = to_past_participle(lemma)
            if corrected[idx] != new_form:
                grammar_indices.append(idx)
                grammar_map[idx] = (corrected[idx], new_form)
                corrected[idx] = new_form

        # --- BE + VB -> VBG ---
        elif prev_word in BE_VERBS and lemma not in AUX_VERBS:
            new_form = lemma[:-1] + "ing" if lemma.endswith("e") else lemma + "ing"
            if corrected[idx] != new_form:
                grammar_indices.append(idx)
                grammar_map[idx] = (corrected[idx], new_form)
                corrected[idx] = new_form

        # --- Subject-verb agreement ---
        # Find the subject (nsubj/nsubjpass)
        subj_tokens = [child for child in token.children if child.dep_ in {"nsubj", "nsubjpass"}]
        subj = subj_tokens[0] if subj_tokens else None

        # fallback: look for nearest preceding noun
        if not subj:
            for t in reversed(doc[:idx]):
                if t.pos_ in {"NOUN", "PROPN", "PRON"}:
                    subj = t
                    break

        plural_subj = subj.tag_ in {"NNS", "NNPS"} if subj else False

        # Correct singular verbs like 'has', 'does' to plural forms if subject is plural
        if text.lower() == "has" and plural_subj:
            new_form = "have"
        elif text.lower() == "does" and plural_subj:
            new_form = "do"
        else:
            new_form = None

        if new_form and corrected[idx] != new_form:
            grammar_indices.append(idx)
            grammar_map[idx] = (corrected[idx], new_form)
            corrected[idx] = new_form

    return corrected, grammar_indices, grammar_map

# -----------------------------
# Capitalize after punctuation
# -----------------------------
def apply_capitalization(tokens):
    capitalize_next = True
    result = []
    for tok in tokens:
        if tok.isalpha() and capitalize_next:
            result.append(tok.capitalize())
            capitalize_next = False
        else:
            result.append(tok)
        if tok in {".", "!", "?"}:
            capitalize_next = True
    return result

# -----------------------------
# Detect errors
# -----------------------------
def detect_errors(text):
    """
    Detect spelling and grammar errors.
    Keeps punctuation and applies capitalization after full stops.
    
    Returns:
        corrected_tokens: list of surface tokens (punctuation retained)
        grammar_indices: indices of corrected grammar tokens
        grammar_map: mapping of original->corrected grammar tokens
        spelling_errors: list of dicts with word, type, and suggestions
    """
    doc = nlp(text)

    # Keep punctuation
    surface_tokens = [t.text for t in doc]

    # Apply grammar rules
    corrected_tokens, grammar_indices, grammar_map = apply_grammar(surface_tokens, doc)

    # Spelling detection
    spelling_errors = []
    for idx, token in enumerate(doc):
        word = token.text
        if not token.is_alpha:
            continue
        lemma = token.lemma_.lower()
        prev_word = corrected_tokens[idx-1] if idx > 0 else None

        if word.lower() in FUNCTION_WORDS or word.lower() in AUX_VERBS or lemma in PAST_PARTICIPLES:
            continue

        check_word = lemma if lemma in VOCAB else word.lower()
        if check_word not in VOCAB:
            candidates = generate_candidates(word, lemma=lemma)
            spelling_errors.append({
                "word": word,
                "index": idx,
                "type": "non-word",
                "suggestions": rank_candidates(candidates, prev_word, word)
            })

    # Capitalize first word and words after punctuation
    capitalize_next = True
    for i, tok in enumerate(corrected_tokens):
        if tok.isalpha() and capitalize_next:
            corrected_tokens[i] = tok.capitalize()
            capitalize_next = False
        else:
            corrected_tokens[i] = tok
        if tok in {".", "!", "?"}:
            capitalize_next = True

    return corrected_tokens, grammar_indices, grammar_map, spelling_errors

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("✍️ Spelling & Grammar Correction System")
st.markdown("""
**How to use:**  
1. Enter your text  
2. Click **Check Text**  
3. 🔴 Red → spelling error  
4. 🟢 Green → grammar correction  
5. Click words to see suggestions
""")

user_input = st.text_area("Enter text:", height=120)

if st.button("🔍 Check Text", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter text.")
    else:
        corrected_tokens, grammar_indices, grammar_map, errors = detect_errors(user_input)
        spelling_indices = {e["index"] for e in errors}

        # Highlight
        highlighted = []
        for i, tok in enumerate(corrected_tokens):
            if i in spelling_indices:
                highlighted.append(f"[**:red[{tok}]**](#)")
            elif i in grammar_indices:
                highlighted.append(f"[**:green[{tok}]**](#)")
            else:
                highlighted.append(tok)

        # Handle punctuation spacing
        highlighted_text = ""
        for i, w in enumerate(highlighted):
            if w in {".", ",", "!", "?", ";", ":"}:
                highlighted_text = highlighted_text.rstrip() + w + " "
            else:
                highlighted_text += w + " "
        st.subheader("🖍 Highlighted Text")
        st.markdown(highlighted_text.strip())

        # Suggestions with score and edit distance
        st.subheader("📌 Suggestions")
        for e in errors:
            with st.expander(f"`{e['word']}`"):
                for s in e["suggestions"]:
                    st.markdown(f"- **{s['word']}** | edit distance: `{s['edit_distance']}` | score: `{round(s['score'],4)}`")

# -----------------------------
# Vocabulary search
# -----------------------------
st.subheader("🔎 Search / Explore Words")
search_word = st.text_input("Search a word in the corpus:")

if search_word:
    lw = search_word.lower()
    if lw in VOCAB:
        st.success(f"✅ '{search_word}' exists (frequency: {WORD_FREQ.get(lw,0)})")
    else:
        st.error(f"❌ '{search_word}' not found in corpus.")

st.caption("📘 MSc Artificial Intelligence | Rule-Based NLP System")
