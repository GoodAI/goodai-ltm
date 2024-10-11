import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.tokens import Doc, Span

# Load the English model
nlp = spacy.load("en_core_web_sm")

# Check if a keyword is valid
def is_valid_keyword(keyword_span: Span) -> bool:
    if len(keyword_span) == 1:
        return keyword_span[0].pos_ in ['NOUN', 'PROPN', 'ADJ']
    else:
        return (keyword_span[-1].pos_ in ['NOUN', 'PROPN'] and
                not any(token.dep_ == 'det' for token in keyword_span))

# Remove stop words, punctuation, and determiners from a keyword
def sanitize_keyword(keyword_span: Span) -> str:
    return ' '.join([token.lemma_ for token in keyword_span if not token.is_stop and not token.is_punct and not token.dep_ == 'det'])

# Extract keywords from a text
def extract_keywords(text: str, num_keywords: int = 5, predefined_keywords: list[str] = None) -> list[str]:
    doc = nlp(text.lower())
    predefined_keywords = predefined_keywords or []

    # Extract candidate keywords from the text
    entities_text = [ent.text for ent in doc.ents]
    candidate_keyword_spans = set(doc.ents) | set(doc.noun_chunks)

    # Filter candidates
    candidate_keyword_spans = [keyword for keyword in candidate_keyword_spans
        if is_valid_keyword(keyword) and len(keyword.text) > 1
    ]

    candidate_keywords = [sanitized_kw for kw in candidate_keyword_spans if (sanitized_kw := sanitize_keyword(kw)) != '']

    # Process text for TF-IDF
    processed_text = ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform([processed_text])
    
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    word_tfidf_scores = dict(zip(feature_names, tfidf_scores))
    
    # Calculate importance for all keywords (including predefined ones)
    all_keywords = set(candidate_keywords + predefined_keywords)
    keyword_importance = {}
    for keyword in all_keywords:
        tfidf_score = word_tfidf_scores.get(keyword.lower(), 0.1)
        ner_importance = 2 if keyword in entities_text else 1
        predefined_importance = 1.5 if keyword in predefined_keywords else 1
        location_importance = 1 + (1 - (text.lower().index(keyword.lower()) / len(text))) if keyword in text.lower() else 1
        
        importance = tfidf_score * ner_importance * location_importance * predefined_importance
        keyword_importance[keyword] = importance

    # Sort keywords by importance
    sorted_keywords = sorted(keyword_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Prioritize predefined keywords if they're relevant
    final_keywords = []
    for keyword, _ in sorted_keywords:
        if len(final_keywords) >= num_keywords:
            break
        if keyword in predefined_keywords:
            final_keywords.append(keyword)
        elif keyword in candidate_keywords:
            final_keywords.append(keyword)
    
    return final_keywords