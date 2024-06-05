from transformers import pipeline

# Load the emotion detection model
emotion_pipeline = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')

# Function to detect emotion of a sentence
def detect_emotion(sentence):
    # Get the emotion prediction
    result = emotion_pipeline(sentence)
    # Extract the emotion label and score
    emotion = result[0]['label']
    score = result[0]['score']
    print(result)
    return emotion, score

# Example sentence
sentence = "I found the User interface very difficult to utilise and personally I think that we could do alot better"
# Detect the emotion
emotion, score = detect_emotion(sentence)

# Print the result
print(f"Sentence: {sentence}")
print(f"Detected emotion: {emotion} with score {score:.4f}")