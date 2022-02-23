import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
import pickle


np.random.seed(500)
Corpus = pd.read_csv(r"dataset.csv", encoding='latin-1')
Corpus['text'].dropna(inplace=True)
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

def text_preprocessing(text):
    # Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    text = str(text).lower()

    # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
    text_words_list = word_tokenize(text)

    # Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(text_words_list):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
    return str(Final_words)
print("=========================================== Text Processing Started =========================================")
Corpus['text_final'] = Corpus['text'].map(text_preprocessing)
print("=========================================== Text Processing Completed =========================================")
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'], Corpus['is_offensive'],
                                                                    test_size=0.3)
                                                                    
Encoder = LabelEncoder()
Encoder.fit(Train_Y)
Train_Y = Encoder.transform(Train_Y)
Test_Y = Encoder.transform(Test_Y)
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
print("=========================================== Training the model Started =========================================")
SVM.fit(Train_X_Tfidf, Train_Y)
print("=========================================== Training the model completed =========================================")
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)
filename = 'labelencoder_fitted.pkl'
pickle.dump(Encoder, open(filename, 'wb'))

# saving TFIDF Vectorizer to disk
filename = 'Tfidf_vect_fitted.pkl'
pickle.dump(Tfidf_vect, open(filename, 'wb'))

# saving the both models to disk
filename = 'svm_trained_model.sav'
pickle.dump(SVM, open(filename, 'wb'))
print("Training completed")