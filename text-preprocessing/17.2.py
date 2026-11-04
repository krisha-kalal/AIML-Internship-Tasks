text = "Natural Language Processing is an exciting field of study."

# tokenize the text into words
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
print("Tokens: ", tokens)

# remove stopwords
from nltk.corpus import stopwords
stopwords_removed_tokens = [i for i in tokens if i.lower() not in stopwords.words('english')]
print("Tokens without Stopwords: ", stopwords_removed_tokens)

# remove urls
import re

text = "Check out this website: https://www.python.org/ for more information."

# regex pattern to match URLs
url_pattern = re.compile(r'https?://\S+')

# remove URLs from the text
text_without_urls = url_pattern.sub('', text)
print("Text without URLs: ", text_without_urls)