import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
import joblib

data = pd.read_csv('dataset.csv')
texts = data['text'].astype(str)
y = data['is_offensive']
print(f"records found in dataset :- {data['text'].count()}")
vectorizer = CountVectorizer(stop_words='english', min_df=0.0001)
print("----------- vectorization done ----------------------------")
X = vectorizer.fit_transform(texts)
model = LinearSVC(class_weight="balanced", dual=False, tol=1e-2, max_iter=1e5)
cclf = CalibratedClassifierCV(base_estimator=model)
cclf.fit(X, y)
print("--------------------- model trained ----------------------------")
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(cclf, 'model.joblib') 
