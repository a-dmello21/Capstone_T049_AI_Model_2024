import nltk
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Download necessary nltk data files
nltk.download('punkt')

def detect_emotion(sentence):
    # Analyze sentiment
    blob = TextBlob(sentence)
    sentiment = blob.sentiment.polarity
    
    # Determine emotion
    emotion = "Neutral"
    if sentiment < 0:
        emotion = "Frustration"
    elif sentiment > 0:
        emotion = "Positive"

    # Tokenize sentence
    words = word_tokenize(sentence)

    # Identify key cause (simplified)
    negative_words = ['difficult', 'bad', 'poor', 'hard', 'awful', 'worse']
    cause_words = [word for word in words if word.lower() in negative_words]

    # If no specific cause word found, take the three most relevant words (simplified approach)
    if not cause_words:
        cause_words = words[:3]
    
    key_cause = ' '.join(cause_words)

    return emotion, key_cause

# Example usage
sentence = "I found the User interface very difficult to utilise and personally I think that we could do a lot better."
emotion, key_cause = detect_emotion(sentence)
print(f"Emotion: {emotion}")
print(f"Key Cause: {key_cause}")

# User input for custom sentence
custom_sentence = input("Enter a sentence: ")
emotion, key_cause = detect_emotion(custom_sentence)
print(f"Emotion: {emotion}")
print(f"Key Cause: {key_cause}")
