import spacy

# Load the spacy model
nlp = spacy.load("en_core_web_sm")

def summarize_sentence(sentence):
    # Process the sentence using spacy
    doc = nlp(sentence)
    
    # Extract nouns and proper nouns as key words
    keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = [x for x in keywords if not (x in seen or seen.add(x))]
    
    return unique_keywords

# Test the function
sentence = "I find this report very challenging due to the lack of proper research on the user interface."
summary = summarize_sentence(sentence)
print(summary)
