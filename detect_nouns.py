import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


# Sample sentence
sentence = "Missing or outdated documents normally get requested by the operations team in HTW."

# Tokenize the sentence and tag each word with its part of speech
tokens = word_tokenize(sentence)
tagged_words = pos_tag(tokens)

# Initialize an empty dictionary to store nouns and their associated adjectives
noun_list = []

# Iterate through the tagged words
for i in range(len(tagged_words) - 1):
    word, pos = tagged_words[i]
    next_word, next_pos = tagged_words[i + 1]

    print(pos)
    
    # Check if the current word is a nou
    if pos.startswith('NN'):
        noun_list.append(word + " " + next_word)
# Print the dictionary
print(noun_list)
