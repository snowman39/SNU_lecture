import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostRegressor, cv

import nltk
import string
import re

data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/wine/wine_train.csv', index_col='id')
test_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/wine/wine_test.csv', index_col='id')

# description과 title이 같은 중복값 제거
data=data.drop_duplicates(['description','title'])
data=data.reset_index(drop=True)
data=data.fillna(-1)

# NLP
data['description']= data['description'].str.lower()
data['description']= data['description'].apply(lambda elem: re.sub('[^a-zA-Z]',' ', elem))

test_data['description']= test_data['description'].str.lower()
test_data['description']= test_data['description'].apply(lambda elem: re.sub('[^a-zA-Z]',' ', elem))

tokenizer = RegexpTokenizer(r'\w+')
words_descriptions = data['description'].apply(tokenizer.tokenize)

test_words_descriptions = test_data['description'].apply(tokenizer.tokenize)

data['description_lengths']= [len(tokens) for tokens in words_descriptions]

test_data['description_lengths']= [len(tokens) for tokens in test_words_descriptions]

# 불용어(stopwords) 제거
stopword_list = stopwords.words('english')
ps = PorterStemmer()

words_descriptions = words_descriptions.apply(lambda elem: [word for word in elem if not word in stopword_list])
words_descriptions = words_descriptions.apply(lambda elem: [ps.stem(word) for word in elem])
data['description_cleaned'] = words_descriptions.apply(lambda elem: ' '.join(elem))

test_words_descriptions = test_words_descriptions.apply(lambda elem: [word for word in elem if not word in stopword_list])
test_words_descriptions = test_words_descriptions.apply(lambda elem: [ps.stem(word) for word in elem])
test_data['description_cleaned'] = test_words_descriptions.apply(lambda elem: ' '.join(elem))

def year_match(name):
    m = re.findall("(\d{4})",name)
    if len(m) == 0:
        return None
    else:
        return m[0]

data['year'] = data['title'].apply(year_match)
test_data['year'] = test_data['title'].apply(year_match)

def prepare_dataframe(vect, data):
    y = data['points']
    X = data.drop(columns=['points','taster_twitter_handle','description','title'])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=52)

    vectorized = vect.fit_transform(X_train['description_cleaned']).toarray()
    vectorized = pd.DataFrame(vectorized)
    X_train = X_train.drop(columns=['description_cleaned'])
    X_train = X_train.fillna(-1)
    print(X_train.columns)
    X_train = pd.concat([X_train.reset_index(drop=True), vectorized.reset_index(drop=True)], axis=1)

    vectorized_valid = vect.transform(X_valid['description_cleaned']).toarray()
    vectorized_valid = pd.DataFrame(vectorized_valid)
    X_valid = X_valid.drop(columns=['description_cleaned'])
    X_valid = X_valid.fillna(-1)
    print(X_valid.columns)
    X_valid = pd.concat([X_valid.reset_index(drop=True), vectorized_valid.reset_index(drop=True)], axis=1)
    
    categorical_features_indices =[0,1,3,4,5,6,7,8,10]
    
    return X_train, y_train, X_valid, y_valid, categorical_features_indices

def prepare_dataframe_test(vect, data):
    vectorized=vect.transform(data['description_cleaned']).toarray()
    vectorized=pd.DataFrame(vectorized)

    X=data.drop(columns=['taster_twitter_handle','description','description_cleaned','title'])
    X=X.fillna(-1)
    print(X.columns)
    X=pd.concat([X.reset_index(drop=True),vectorized.reset_index(drop=True)],axis=1)

    return X

#model definintion and training.
def perform_model(X_train, y_train, X_valid, y_valid, categorical_features_indices, name):
    model = CatBoostRegressor(
        random_seed = 100,
        loss_function = 'RMSE',
        iterations=2000,
    )
    
    model.fit(
        X_train, y_train,
        cat_features = categorical_features_indices,
        verbose=False,
        eval_set=(X_valid, y_valid)
    )
    
    train_preds = model.predict(X_train)
    print('RMSE: {}'.format(mean_squared_error(y_train, train_preds, squared=False)))

    valid_preds = model.predict(X_valid)
    print('RMSE: {}'.format(mean_squared_error(y_valid, valid_preds, squared=False)))

    test_preds = model.predict(X_test)

    my_submission = pd.DataFrame({'id': test_data.index, 'points': test_preds})
    my_submission.to_csv('wine_my_submission.csv', index=False)

vect = CountVectorizer(analyzer='word', token_pattern=r'\w+',max_features=500)
training_variable = prepare_dataframe(vect, data)
X_test = prepare_dataframe_test(vect, test_data)
perform_model(*training_variable, 'Bag of Words Counts')
