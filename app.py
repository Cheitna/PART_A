# ======================================
# app.py — NLTK rule-based system (FIXED)
# ======================================

import streamlit as st
import pickle
from nltk.metrics import edit_distance
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer

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
lemmatizer = WordNetLemmatizer()

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
# Prepare sorted vocabulary for UI
# -----------------------------
SORTED_VOCAB = sorted(
    VOCAB,
    key=lambda w: WORD_FREQ.get(w, 0),
    reverse=True
)

# -----------------------------
# Helper functions
# -----------------------------
def bigram_prob(w1, w2):
    return (BIGRAM_COUNTS.get((w1, w2), 0) + 1) / (
        UNIGRAM_COUNTS.get(w1, 0) + VOCAB_SIZE
    )

def to_past_participle(lemma):
    """Safe past participle conversion"""
    if lemma in PAST_PARTICIPLES:
        return lemma
    if lemma in IRREGULAR_PAST:
        return IRREGULAR_PAST[lemma]
    if lemma.endswith("e"):
        return lemma + "d"
    return lemma + "ed"

def preprocess_text(text):
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    return tokens, tags

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
# Grammar correction (FIXED)
# -----------------------------
def apply_grammar(tokens, pos_tags):
    corrected = tokens[:]
    grammar_indices = []
    grammar_map = {}

    for i, (token, tag) in enumerate(pos_tags):
        lower_tok = token.lower()
        prev_word = corrected[i - 1].lower() if i > 0 else None

        # ---- PERFECT TENSE: has/have/had + VB → VBN ----
        if (
            prev_word in HAS_VERBS
            and tag == "VB"                      # ✅ ONLY base verb
            and lower_tok not in AUX_VERBS
            and lower_tok not in PAST_PARTICIPLES
        ):
            new_form = to_past_participle(lower_tok)
            grammar_indices.append(i)
            grammar_map[i] = (corrected[i], new_form)
            corrected[i] = new_form

        # ---- PROGRESSIVE: be + VB → VBG ----
        elif (
            prev_word in BE_VERBS
            and tag == "VB"
            and lower_tok not in AUX_VERBS
        ):
            new_form = (
                lower_tok[:-1] + "ing"
                if lower_tok.endswith("e")
                else lower_tok + "ing"
            )
            grammar_indices.append(i)
            grammar_map[i] = (corrected[i], new_form)
            corrected[i] = new_form

        # ---- SUBJECT–VERB AGREEMENT ----
        if lower_tok in {"has", "does"}:
            plural_subject = False
            for j in range(i - 1, -1, -1):
                if pos_tags[j][1] in {"NNS", "NNPS"}:
                    plural_subject = True
                    break

            if plural_subject:
                new_form = "have" if lower_tok == "has" else "do"
                grammar_indices.append(i)
                grammar_map[i] = (corrected[i], new_form)
                corrected[i] = new_form

    return corrected, grammar_indices, grammar_map

# -----------------------------
# Capitalization
# -----------------------------
def apply_capitalization(tokens):
    result = []
    cap_next = True
    for tok in tokens:
        if tok.isalpha() and cap_next:
            result.append(tok.capitalize())
            cap_next = False
        else:
            result.append(tok)

        if tok in {".", "!", "?"}:
            cap_next = True
    return result

# -----------------------------
# Detect errors
# -----------------------------
def detect_errors(text):
    tokens, pos_tags = preprocess_text(text)
    corrected, grammar_idx, grammar_map = apply_grammar(tokens, pos_tags)

    spelling_errors = []
    for i, (token, tag) in enumerate(pos_tags):
        if not token.isalpha():
            continue

        lemma = (
    lemmatizer.lemmatize(token.lower(), "v")
    if tag.startswith("VB")
    else lemmatizer.lemmatize(token.lower())
)

        prev_word = corrected[i - 1].lower() if i > 0 else None

        if (
            token.lower() in FUNCTION_WORDS
            or token.lower() in AUX_VERBS
            or lemma in PAST_PARTICIPLES
        ):
            continue

        check_word = lemma if lemma in VOCAB else token.lower()
        if check_word not in VOCAB:
            candidates = generate_candidates(token, lemma)
            spelling_errors.append({
                "word": token,
                "index": i,
                "type": "non-word",
                "suggestions": rank_candidates(candidates, prev_word, token)
            })

    corrected = apply_capitalization(corrected)
    return corrected, grammar_idx, grammar_map, spelling_errors

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


user_input = st.text_area("Enter text (Max 500 words):", height=120)

if st.button("🔍 Check Text", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter text.")
    else:
        corrected, grammar_idx, grammar_map, errors = detect_errors(user_input)
        spelling_idx = {e["index"] for e in errors}

        highlighted = []
        for i, tok in enumerate(corrected):
            if i in spelling_idx:
                highlighted.append(f"**:red[{tok}]**")
            elif i in grammar_idx:
                highlighted.append(f"**:green[{tok}]**")
            else:
                highlighted.append(tok)

        text = ""
        for w in highlighted:
            if w in {".", ",", "!", "?", ";", ":"}:
                text = text.rstrip() + w + " "
            else:
                text += w + " "

        st.subheader("🖍 Highlighted Text")
        st.markdown(text.strip())

        st.subheader("📌 Suggestions")
        for e in errors:
            with st.expander(f"`{e['word']}`"):
                for s in e["suggestions"]:
                    st.markdown(
                        f"- **{s['word']}** | distance `{s['edit_distance']}` | score `{round(s['score'],4)}`"
                    )
# -----------------------------
# Vocabulary search + scrollable list
# -----------------------------
st.subheader("🔎 Search / Explore Words")

search = st.text_input(
    "Search a word in the corpus:",
    placeholder="Type to filter words..."
)

# Filter vocab
if search:
    filtered_vocab = [
        w for w in SORTED_VOCAB
        if w.startswith(search.lower())
    ]
else:
    filtered_vocab = SORTED_VOCAB[:300]  # top frequent words

# Search feedback
if search:
    if search.lower() in VOCAB:
        st.success(f"✅ '{search}' exists (frequency: {WORD_FREQ.get(search.lower(),0)})")
    else:
        st.error(f"❌ '{search}' not found in corpus.")

# Scrollable list
st.markdown("**📜 Dictionary Words**")

with st.container(height=220):
    for w in filtered_vocab:
        freq = WORD_FREQ.get(w, 0)
        st.markdown(f"- **{w}** <span style='color:gray'>(freq: {freq})</span>",
                    unsafe_allow_html=True)
        
st.caption("📘 MSc Artificial Intelligence | Rule-Based NLP System")
