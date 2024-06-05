import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

# Download necessary nltk data files
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Initialize the emotion classifier
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def get_hypernyms(noun):
    """
    Get the hypernyms of a noun using WordNet.
    
    Parameters:
    noun (str): The noun for which to find hypernyms.
    
    Returns:
    list of str: A list of hypernyms for the noun.
    """
    synsets = wn.synsets(noun, pos=wn.NOUN)
    if not synsets:
        return ["unknown"]
    
    hypernyms = synsets[0].hypernyms()
    return [hypernym.lemma_names()[0] for hypernym in hypernyms] if hypernyms else ["unknown"]

def classify_nouns(nouns):
    """
    Classify a list of nouns into groups based on their hypernyms.
    
    Parameters:
    nouns (list of str): A list of nouns to classify.
    
    Returns:
    tuple: A tuple containing:
        - classifications (list of str): A list of hypernyms for each noun.
        - noun_groups (list of str): A list of unique hypernyms representing categories of the nouns.
    """
    classifications = []
    noun_groups = set()
    
    for noun in nouns:
        hypernyms = get_hypernyms(noun)
        classifications.extend(hypernyms)
        noun_groups.update(hypernyms)
    
    return classifications, list(noun_groups)

def deconstruct_sentence(sentence):
    # Tokenize the sentence
    words = word_tokenize(sentence)
    
    # Perform part-of-speech tagging
    tagged_words = pos_tag(words)
    
    # Extract nouns and adjectives
    nouns = []
    adjectives = []

    condensed_sentence = ""

    for word, pos in tagged_words:
        if pos.startswith("NN"):
            nouns.append(word)
            condensed_sentence += f"{word} "
        elif pos.startswith("JJ"):
            adjectives.append(word)
            condensed_sentence += f"{word} "
    
    # Perform sentiment analysis
    sentiment_scores = sia.polarity_scores(sentence)
    sentiment = 'neutral'
    if sentiment_scores['compound'] >= 0.05:
        sentiment = 'positive'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = 'negative'

    # Perform emotion detection
    emotions = emotion_classifier(sentence)
    emotions_dict = {emotion['label']: emotion['score'] for emotion in emotions[0]}
    primary_emotion = max(emotions_dict, key=emotions_dict.get)

    # Classify nouns into groups
    #classifications, noun_groups = classify_nouns(nouns)
    
    return nouns, adjectives, sentiment, primary_emotion, condensed_sentence

# Test the function
print("\n\n\n\n\n")
while True:
    sentence = input("Paragraph for analysis: ")

    N, A, sentiment, primary_emotion, summary_sentence = deconstruct_sentence(sentence)
    print("Tag:", N)
    print("Sentiment:", sentiment)
    print("Primary Emotion:", primary_emotion)
    print("\n")
