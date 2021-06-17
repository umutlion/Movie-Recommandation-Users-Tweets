import tweepy
import pandas as pd
import re
import numpy as np
import string
import pickle
import json

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras.models import Sequential

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
userPattern = '@[^\s]+'

nltk.download('stopwords', quiet=True)
stopword = set(stopwords.words('english'))

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


def process_tweets(tweet):
    tweet = tweet.lower()
    tweet = tweet[1:]
    tweet = re.sub(urlPattern, '', tweet)
    tweet = re.sub(userPattern, '', tweet)
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(tweet)
    final_tokens = [w for w in tokens if w not in stopword]
    wordLemm = WordNetLemmatizer()
    finalwords = []
    for w in final_tokens:
        if len(w) > 1:
            word = wordLemm.lemmatize(w)
            finalwords.append(word)
    return ' '.join(finalwords)


max_words = 5000
max_len = 200

data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin',
                   names=['polarity', 'id', 'date', 'query', 'user', 'text'])

data = data.sample(frac=1)
data = data[:200000]

data['polarity'] = data['polarity'].replace(4, 1)

data.drop(['id', 'date', 'query', 'user'], axis=1, inplace=True)

data['text'] = data['text'].astype('str')

data['processed_tweets'] = data['text'].apply(lambda x: process_tweets(x))

X = data['processed_tweets'].values
y = data['polarity'].values

vector = TfidfVectorizer(sublinear_tf=True)
X = vector.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


def model_Evaluate(model):
    y_pred = model.predict(X_test)

    cf_matrix = confusion_matrix(y_test, y_pred)

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_percentages)]


lg = LogisticRegression()
history = lg.fit(X_train, y_train)
model_Evaluate(lg)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data.processed_tweets)
sequences = tokenizer.texts_to_sequences(data.processed_tweets)
tweets = pad_sequences(sequences, maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tweets, data.polarity.values, test_size=0.2, random_state=101)

max_words = 80000
max_len = 200

model = Sequential()
model.add(layers.Embedding(max_words, 128))
model.add(layers.LSTM(64, dropout=0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(8, activation='sigmoid'))

model.compile(optimizer='adam', loss='categorical_accuracy', metrics=['accuracy'])
model = load_model('sent-model.h5')

file = open('vectoriser.pickle', 'wb')
pickle.dump(vector, file)
file.close()

file = open('logisticRegression.pickle', 'wb')
pickle.dump(lg, file)
file.close()


def load_models():
    file = open('vectoriser.pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    file = open('logisticRegression.pickle', 'rb')
    lg = pickle.load(file)
    file.close()
    return vectoriser, lg


def predict(vectoriser, model, text):
    processes_text = [process_tweets(sen) for sen in text]
    textdata = vectoriser.transform(processes_text)
    sentiment = model.predict(textdata)

    data = []
    for text, pred in zip(text, sentiment):
        data.append((text, pred))
    df = pd.DataFrame(data, columns=['text', 'sentiment'])
    df = df.replace([0, 1], ["Negative", "Positive"])
    return df


# Twitter

consumer_key = "yB8GV2Qqofa4iKfhR7JVEAWHq"
consumer_secret = "ltwuIyMdNjQe6bqLuIJVaCdqXNp5FSL1VPul1NiRMSrdM6zkhu"
access_token = "353460411-G8ggU0Pe7CplI22y1UbVPQVu2fyZ0sVfP6aVKC29"
access_token_secret = "qGrMLWPvS8GMd1MvKueLPRM9v7zCf80PwWKRHwhClAITs"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

tweetCount = 40
pages = 5

language = "en"
public_tweets = api.home_timeline(language=language, count=tweetCount, pages=pages)

username = []
text = []
dummy = []

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002500-\U00002BEF"
                           u"\U00002702-\U000027B0"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"
                           u"\u3030"
                           "]+", flags=re.UNICODE)

for tweet in public_tweets:
    username.append(tweet.user.screen_name)
    dummy.append(tweet.text)
    for i in dummy:
        i = emoji_pattern.sub(r'', i)
    text.append(i)

data = pd.DataFrame(
    {'username': username, 'Tweet': text}
)
data['Tweet'] = data['Tweet'].replace(to_replace=r'[RT]+', value='', regex=True)
data['Tweet'] = data['Tweet'].replace(to_replace=r'@[A-Z0-9a-z_:]+', value='', regex=True)
data['Tweet'] = data['Tweet'].replace(to_replace='https?://[A-Za-z0-9./]+', value='', regex=True)

tweetData = []
pred_data = []
tweetData = data['Tweet'].tolist()

from google_trans_new import google_translator

translator = google_translator()

translated_words = []
for i in range(3):
    a = translator.translate(data['Tweet'][i])
    translated_words.append(a)

if __name__ == "__main__":
    vectoriser, lg = load_models()

    df = predict(vectoriser, lg, translated_words)


for i in translated_words:
    sequence = tokenizer.texts_to_sequences([i])
    test = pad_sequences(sequence, maxlen=max_len)
    pred = model.predict(test)
    pred_data.append(pred)

total = 0
count = 0
for i in pred_data:
    total = total + i
    count += count
average_pred = total / 3
print(average_pred)
