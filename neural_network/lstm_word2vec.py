import numpy as np
np.random.seed(42)
import pandas as pd
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from tensorflow.keras.layers import LSTM, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras import backend as K
import re
from sklearn.model_selection import StratifiedKFold
from gensim.models import KeyedVectors
from underthesea import word_tokenize
from pyvi import ViTokenizer

main_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(main_dir, 'data')
submiss_dir = os.path.join(main_dir, 'submission')
embedding_dir = os.path.join(main_dir, 'word_embedding')

train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'), index_col=0)
test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'), index_col=0)
submiss = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

etnlp_file = os.path.join(embedding_dir, 'W2V_ner.vec')
etnlp_word2vec = KeyedVectors.load_word2vec_format(etnlp_file)

sonvx_file = os.path.join(embedding_dir, 'baomoi.model.bin')
sonvx_word2vec = KeyedVectors.load_word2vec_format(sonvx_file, binary=True)

def preprocess(text, pyvi=False):
    if pyvi:
        text = ViTokenizer.tokenize(text)
    else:
        text = word_tokenize(text, format="text")
    words_only = re.findall('\w+', text)   
    text = ' '.join(words_only)
    text = re.sub('[\d ]+', ' ', text).lower()  
    return text

x_train = train_data.text.apply(preprocess)
y_train = train_data.label
x_test = test_data.text.apply(preprocess)

def build_matrix(word_index, word2vec):
    embedding_matrix = np.zeros((len(word_index) + 1, word2vec.vector_size))
    for word, i in word_index.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)
    return embedding_matrix

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

MAX_LEN = 300
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(x_train) + list(x_test))

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

embedding_models = [etnlp_word2vec, sonvx_word2vec]
embedding_matrix = np.concatenate(
    [build_matrix(tokenizer.word_index, f) for f in embedding_models], axis=-1)

LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

def get_model(embedding_matrix=embedding_matrix):
    words = Input(shape=(MAX_LEN,))

    x = Embedding(*embedding_matrix.shape, 
                  weights=[embedding_matrix],
                  trainable=False)(words)

    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(LSTM(LSTM_UNITS, 
                           return_sequences=True))(x)
    x = Bidirectional(LSTM(LSTM_UNITS,
                           return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS,
                                activation='relu')(hidden)])
    
    result = Dense(1, activation='sigmoid')(hidden)
    
    model = Model(inputs=words, outputs=result)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_score])
    return model

def run_model(model, n_epochs, batch_size, X_train, y_train, val_data, X_test):
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_data=val_data,
              verbose=False)
    prediction = model.predict(x_test)
    return prediction

batch_size = 16
epochs_per_fold = 3
n_folds = 10
checkpoint_predictions = []

cv = StratifiedKFold(n_splits=n_folds, shuffle=True)

for i, (train,valid) in enumerate(cv.split(x_train, y_train)):
    
    print(f'TRAIN ON FOLD {i+1}/{n_folds}')
    model = get_model()
    for epoch in range(epochs_per_fold):
        prediction = run_model(model, 1, batch_size,
                               x_train[train], y_train[train],
                               (x_train[valid], y_train[valid]),
                                x_test)
        checkpoint_predictions.append(model.predict(x_test))
        

threshold = 0.5

weights = [2, 4, 1]*n_folds
predict_proba = np.average(checkpoint_predictions, weights=weights, axis=0)

submiss.label = (predict_proba > threshold).astype(np.int)
submiss.to_csv(os.path.join(submiss_dir, 'word2vec_LSTM.csv'), index=False)    # 0.89786