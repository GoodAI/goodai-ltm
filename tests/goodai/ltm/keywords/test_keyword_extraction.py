from goodai.ltm.keywords.keywords import extract_keywords


def test_keyword_extraction_ml():
    text = "Machine learning is a subfield of artificial intelligence. It is a type of artificial intelligence that allows machines to learn from data."
    keywords = extract_keywords(text)
    
    assert "machine learning" in keywords
    assert "artificial intelligence" in keywords


def test_keyword_extraction_conversation_archie():
    text = "Archie says: I love to make pasta, it is my favourite food and the best shape is tortellini."
    keywords = extract_keywords(text)
    
    assert "archie" in keywords
    assert "pasta" in keywords
    assert "favourite food" in keywords


def test_keyword_extraction_conversation_archie_predefined():
    text = "Archie says: I love to make pasta, it is my favourite food and the best shape is tortellini."
    predefined_keywords = ["food"]
    keywords = extract_keywords(text, predefined_keywords=predefined_keywords)

    assert "archie" in keywords
    assert "pasta" in keywords
    assert "food" in keywords


def test_keyword_extraction_conversation_maddie():
    text = "Maddie says: I am learning to play the guitar. I love playing it and my favourite song is 'Hotel California'."
    keywords = extract_keywords(text)

    assert "maddie" in keywords
    assert "hotel california" in keywords


def test_keyword_extraction_ml_predefined():
    text = "Machine learning is a subfield of artificial intelligence. It is a type of artificial intelligence that allows machines to learn from data."
    predefined_keywords = ["data"]
    keywords = extract_keywords(text, predefined_keywords=predefined_keywords)

    assert "data" in keywords

