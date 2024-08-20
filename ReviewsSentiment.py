import spacy
import regex as re
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
spacy.cli.download("en_core_web_sm")

# LOAD DATASET AND CONVERT TO DATAFRAME
data = load_dataset('rotten_tomatoes')
data_train = data["train"].to_pandas()
data_test = data["test"].to_pandas()
data_val = data["validation"].to_pandas()
df_data = pd.concat([data_train, data_test, data_val])

# LOAD STOPWORDS
spacy_en = spacy.load('en_core_web_sm')
stopwords = spacy_en.Defaults.stop_words

# CLEAN AND TOKENIZE TEXT FUNCTIONS
def clean_text(text):
    regex = "[.!?/\-_*:\",'()@#$%^]"
    text = text.lower()
    text = re.sub(regex, " ", text)
    text = re.sub("\s+", " ", text)
    return text

def tokenize_text(text):
    tokens = spacy_en(text)
    result = []
    for token in tokens:
        lemma = token.lemma_
        if(lemma not in stopwords):
            result.append(lemma)
    return result

# CLEANING DATA
df_data["text"] = df_data["text"].apply(clean_text)

# VECTORIZE TEXT

count_vectorize = CountVectorizer(tokenizer=tokenize_text)
vectors = count_vectorize.fit_transform(df_data["text"])

# SPLITTING DATA TO TRAIN AND TEST
x = vectors.toarray()
y = df_data["label"].values

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)

# TRAINING MODEL
gnb = GaussianNB()
gnb.fit(x_train, y_train)

# PREDICTION
y_pred = gnb.predict(x_valid)

correct = (y_valid == y_pred).sum()
print(f"Recognized {correct}/{x_valid.shape[0]}")
print(f"{correct/x_valid.shape[0] * 100}%")

# TESTING MODEL
review_positive = "It was an amazing movie, I had great time!"
review_negative = "Bad acting, uninteresting writing, no redeeming qualities"
review_not_obvious = "The actors did their best, but with a script like that, there's only so much you can do."

test1 = clean_text(review_positive)
vector1 = count_vectorize.transform([test1]).toarray()
result1 = gnb.predict(vector1)
print(review_positive)
print("0 - negative, 1 - positive")
print(f"Recognized: {result1}")

test2 = clean_text(review_negative)
vector2 = count_vectorize.transform([test2]).toarray()
result2 = gnb.predict(vector2)
print(review_negative)
print("0 - negative, 1 - positive")
print(f"Recognized: {result2}")

test3 = clean_text(review_not_obvious)
vector3 = count_vectorize.transform([test3]).toarray()
result3 = gnb.predict(vector3)
print(review_not_obvious)
print("0 - negative, 1 - positive")
print(f"Recognized: {result3}")


