from keras import models
from keras import layers
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

print("start")

train_file = '/content/drive/MyDrive/Colab Notebooks/wine/wine_train.csv'
test_file = '/content/drive/MyDrive/Colab Notebooks/wine/wine_test.csv'

train_data_full = pd.read_csv(train_file, index_col='id')
test_data_full = pd.read_csv(test_file, index_col='id')

# predictor variable과 target variable을 분리
train_targets = train_data_full['points']
train_data_full.drop(['points'], axis=1, inplace=True)

feature_columns = ['country', 'region_1', 'price', 'province', 'taster_name', 'variety']
train_raw_data = train_data_full[feature_columns].copy()
test_raw_data = test_data_full[feature_columns].copy()

# preprocessing train data
categorical_columns = ['country', 'region_1', 'province', 'taster_name', 'variety']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))])

categorical_train_data = pd.DataFrame(categorical_transformer.fit_transform(train_raw_data[categorical_columns]))
categorical_train_data.index = train_raw_data.index

numerical_train_data = train_raw_data.drop(categorical_columns, axis=1)
my_imputer = SimpleImputer(strategy='mean')
imputed_train_data = pd.DataFrame(my_imputer.fit_transform(numerical_train_data))
imputed_train_data.index = numerical_train_data.index

train_data = pd.concat([imputed_train_data, categorical_train_data], axis=1)
# preprocessing test data
categorical_test_data = pd.DataFrame(categorical_transformer.transform(test_raw_data[categorical_columns]))
categorical_test_data.index = test_raw_data.index

numerical_test_data = test_raw_data.drop(categorical_columns, axis=1)
imputed_test_data = pd.DataFrame(my_imputer.transform(numerical_test_data))
imputed_test_data.index = numerical_test_data.index

test_data = pd.concat([imputed_test_data, categorical_test_data], axis=1)

# data normalize
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

# test data를 정규화할 때에도 train data 의 mean , std 사용
test_data -= mean 
test_data /= std

def build_model():
  model = models.Sequential()
  model.add(layers.Dense(64, activation='relu',
                         input_shape=(train_data.shape[1],)))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(1))
  model.compile(optimizer='rmsprop', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
  return model

# cross validation
k = 5
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
  print('처리중인 폴드 #', i)
  val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
  val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]

  partial_train_data = np.concatenate(
      [train_data[:i * num_val_samples],
       train_data[(i+1) * num_val_samples:]],
       axis=0
  )
  partial_train_targets = np.concatenate(
      [train_targets[:i * num_val_samples],
       train_targets[(i+1) * num_val_samples:]],
       axis=0
  )

  model = build_model()
  model.fit(partial_train_data, partial_train_targets,
            epochs=num_epochs, batch_size=1, verbose=2)
  val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
  all_scores.append(val_mae)

test_preds = model.predict(test_data)

# Submission 파일을 생성한다
my_submission = pd.DataFrame({'id': test_data.index, 'points': test_preds.ravel()})
my_submission.to_csv('wine_my_submission.csv', index=False)
