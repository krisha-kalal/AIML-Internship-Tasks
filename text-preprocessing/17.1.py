import string

text = "  Natural Language processing, (NLP)!       "

# covert text to lowercase
lowercased_text = text.lower()
print("Lowercased Text: ", lowercased_text)

# remove punctuation
punctuation_removed_text = "".join([i for i in lowercased_text if i not in string.punctuation])
print("Text without Punctuation: ", punctuation_removed_text)

# remove whitespace
whitespace_removed_text = punctuation_removed_text.strip()
print("Text without Whitespace: ", whitespace_removed_text)