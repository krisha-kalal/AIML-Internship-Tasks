# stemming
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

text = "available running races connected"

stemmed_text = " ".join([stemmer.stem(word) for word in text.split()])
print("Stemmed Text: ", stemmed_text)

# lemmatization
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

text = "flying running jumped"

lemmatized_text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
print("Lemmatized Text: ", lemmatized_text)