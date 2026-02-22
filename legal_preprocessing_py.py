# -*- coding: utf-8 -*-
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

words_to_keep = {
    'not', 'no', 'nor', 'against', 'don', 'doesn', 'didn', 'isn', 'aren',
    'wasn', 'weren', 'hasn', 'haven', 'hadn', 'won', 'wouldn', 'shan',
    'shouldn', 'couldn', 'mustn', 'needn', 'mightn',
    'can', 'will', 'should', 'could', 'would', 'might', 'must', 'need',
    'if', 'but', 'until', 'before', 'after', 'during', 'while', 'now',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'some', 'only',
    'same', 'too', 'very', 'just'
}

legal_stop_words = stop_words - words_to_keep

def clean_legal_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)

    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in legal_stop_words
    ]

    return ' '.join(cleaned_tokens)