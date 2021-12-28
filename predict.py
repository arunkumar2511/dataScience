import joblib
from sklearn.feature_extraction.text import CountVectorizer

try:
    model = joblib.load("model.joblib") 
    vectorizer = CountVectorizer(stop_words='english', min_df=0.0001)
    X = vectorizer.fit_transform(["what the fuck"])
    print(X)
    result = model.predict(X)
    print(result)
except Exception as ex:
    print(ex)