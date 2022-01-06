import joblib
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

try:
    model = joblib.load("model.joblib") 
    data = pd.read_csv('dataset.csv')
    data.sort_values("text", inplace = True)
    data.drop_duplicates(subset ="text",
                     keep = False, inplace = True)
    texts = data.iloc[:15442]['text'] 
    vectorizer = CountVectorizer(stop_words='english',vocabulary=texts, min_df=0.0001)
    print("vect")
    X = vectorizer.transform(["Going to the creek with my bitches tomorrow"])
    print("x====>",X)
    result = model.predict(X)
    print("result====>",result)
except Exception as ex:
    print(ex)