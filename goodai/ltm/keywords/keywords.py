import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import candidate

from goodai.ltm.keywords.classifier import classify_sentence

# Load the English model
nlp = spacy.load("en_core_web_sm")

# Check if a keyword is valid
def is_valid_keyword(keyword:str) -> bool:
    keyword_doc = nlp(keyword)
    if len(keyword_doc) == 1:
        return keyword_doc[0].pos_ in ['NOUN', 'PROPN', 'ADJ']
    else:
        return (keyword_doc[-1].pos_ in ['NOUN', 'PROPN'] and 
                not any(token.dep_ == 'det' for token in keyword_doc))

# Remove stop words, punctuation, and determiners from a keyword
def sanitize_keyword(keyword:str) -> str:
    doc = nlp(keyword)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.dep_ == 'det'])

# Extract keywords from a text
def extract_keywords(text:str, num_keywords:int=5, predefined_keywords:list[str]=None) -> list[str]:
    doc = nlp(text.lower())
    predefined_keywords = predefined_keywords or []

    # Extract candidate keywords from the text
    entities = [ent.text for ent in doc.ents]
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    candidate_keywords = list(set(entities + noun_chunks))

    # Filter candidates
    candidate_keywords = [
        keyword for keyword in candidate_keywords 
        if is_valid_keyword(keyword) and len(keyword) > 1
    ]

    candidate_keywords = [sanitized_kw for kw in candidate_keywords if (sanitized_kw := sanitize_keyword(kw)) != '']

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
        ner_importance = 2 if keyword in entities else 1
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

# Get keywords using the classifier to enhance which keywords are most important and classify the sentence type
def keywords_with_classification(text:str, meta_classes:list[str], predefined_keywords:list[str]) -> tuple[list[str], str]:
    # Get keywords that are applicable to the text
    applicable_keywords = []
    if len(predefined_keywords) > 0:
        # Use the classifier to get the keywords that are most relevant to the text
        bart_keywords, confidence = classify_sentence(text, predefined_keywords, multi_label=True)
        for keyword, score in zip(bart_keywords, confidence):
            if score > 0.5:
                applicable_keywords.append(keyword)

    # Get the keywords that are most important to the text
    keywords = extract_keywords(text, predefined_keywords=applicable_keywords)
    
    # Get the sentence type of the text
    meta_types, confidences = classify_sentence(text, meta_classes)
    sentence_type = meta_types[0]
    confidence = confidences[0]

    return keywords, sentence_type

# Get keywords only without using the classifier
def keywords_only(text:str, predefined_keywords:list[str]) -> list[str]:

    return extract_keywords(text, predefined_keywords=predefined_keywords)


def keywords_with_filtering(text:str, predefined_keywords:list[str]) -> list[str]:
    applicable_keywords = []
    if len(predefined_keywords) > 0:
        # Use the classifier to get the keywords that are most relevant to the text
        bart_keywords, confidence = classify_sentence(text, predefined_keywords, multi_label=True)
        for keyword, score in zip(bart_keywords, confidence):
            if score > 0.5:
                applicable_keywords.append(keyword)

    # Get the keywords that are most important to the text
    keywords = extract_keywords(text, predefined_keywords=applicable_keywords)

    return keywords