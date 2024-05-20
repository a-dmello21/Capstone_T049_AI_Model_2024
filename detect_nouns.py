import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Download necessary nltk data files
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def get_nouns(sentence):
    # Tokenize the sentence
    words = word_tokenize(sentence)
    
    # Perform part-of-speech tagging
    tagged_words = pos_tag(words)
    
    # Function to determine if a noun is similar to "lack"
    def is_similar_to_lack(word):
        synsets = wn.synsets(word, pos=wn.NOUN)
        for synset in synsets:
            if 'absence' in synset.lemma_names() or 'deficiency' in synset.lemma_names():
                return True
        return False
    
    # Extract nouns excluding those similar to "lack"
    nouns = [word for word, pos in tagged_words if pos in ['NN', 'NNS'] and not is_similar_to_lack(word)]
    
    return nouns

# Test the function
sentence = "Every time we have to stop with delays it probably adds, I would say at least 1 to 2 days to the job"
print(get_nouns(sentence))