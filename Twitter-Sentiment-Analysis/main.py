import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load Dataset
df = pd.read_csv(
    'training.1600000.processed.noemoticon.csv.zip',
    encoding='latin-1',
    header=None
)

df = df[[0, 5]]
df.columns = ['polarity', 'text']

# Step 2: Keep only Positive and Negative tweets
df = df[df.polarity != 2]
df['polarity'] = df['polarity'].map({0: 0, 4: 1})

# Step 3: Clean text
def clean_text(text):
    return text.lower()

df['clean_text'] = df['text'].apply(clean_text)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'],
    df['polarity'],
    test_size=0.2,
    random_state=42
)

# Step 5: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train_tfidf, y_train)
bnb_pred = bnb.predict(X_test_tfidf)

print("BernoulliNB Accuracy:", accuracy_score(y_test, bnb_pred))
print(classification_report(y_test, bnb_pred))

# Step 7: Support Vector Machine
svm = LinearSVC(max_iter=1000)
svm.fit(X_train_tfidf, y_train)
svm_pred = svm.predict(X_test_tfidf)

print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# Step 8: Logistic Regression
logreg = LogisticRegression(max_iter=100)
logreg.fit(X_train_tfidf, y_train)
logreg_pred = logreg.predict(X_test_tfidf)

print("Logistic Regression Accuracy:", accuracy_score(y_test, logreg_pred))
print(classification_report(y_test, logreg_pred))

# Step 9: Sample Predictions
sample_tweets = [
    "I love this!",
    "I hate that!",
    "It was okay, not great."
]

sample_vec = vectorizer.transform(sample_tweets)

print("\nSample Predictions (1=Positive, 0=Negative)")
print("BernoulliNB:", bnb.predict(sample_vec))
print("SVM:", svm.predict(sample_vec))
print("Logistic Regression:", logreg.predict(sample_vec))
