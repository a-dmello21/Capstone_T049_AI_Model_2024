import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from string import punctuation

# Download necessary nltk data files
nltk.download('punkt')
nltk.download('stopwords')

def get_keywords(sentence):
    # Tokenize the sentence
    words = word_tokenize(sentence.lower())
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english') + list(punctuation))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Frequency distribution of the words
    freq_dist = FreqDist(filtered_words)
    
    # Extracting keywords (you can adjust the number of keywords)
    keywords = [word for word, freq in freq_dist.most_common()]
    
    return keywords

# Example usage
sentence = "I found the User interface very difficult to utilise and personally I think that we could do alot better"
keywords = get_keywords(sentence)
print("Keywords:", keywords)
