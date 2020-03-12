import numpy as np
np.random.seed(42)
import pandas as pd
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from tensorflow.keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras import backend as K
import re
from tqdm import tqdm
import pickle
from sklearn.model_selection import StratifiedKFold

main_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(main_dir, 'data')
submiss_dir = os.path.join(main_dir, 'submission')
embedding_dir = os.path.join(main_dir, 'word_embedding')

train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'), index_col=0)
test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'), index_col=0)
submiss = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))



def preprocess(text,):
    words_only = re.findall('\w+', text)   
    text = ' '.join(words_only)
    text = re.sub('[\d ]+', ' ', text).lower()  
    return text

X_train = train_data.text.apply(preprocess)
y_train = train_data.label
X_test = test_data.text.apply(preprocess)

maxlen = 300
embed_size = 300
max_features = 20000

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

#EMBEDDING_FILE = os.path.join(embedding_dir, 'cc.vi.300.vec')
#def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
# embeddings = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in tqdm(open(EMBEDDING_FILE)))
#pickle.dump(embeddings, open(os.path.join(embedding_dir, 'fasttext.pkl'), 'wb'))

embeddings  = pickle.load(open(os.path.join(embedding_dir, 'fasttext.pkl'), 'rb'))    # path to vietnamese fasttext model

word_index = tokenizer.word_index
nb_words = len(word_index)+1
embedding_matrix = np.zeros((nb_words, embed_size))

for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector

def f1_score(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(nb_words, embed_size, 
                  weights=[embedding_matrix],
                  trainable=True)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(256, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(1, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[f1_score])
    return model

def run_model(model, n_epochs, batch_size, X_train, y_train, val_data, X_test):
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_data=val_data,
              verbose=True)
    prediction = model.predict(x_test)
    return prediction

epochs_per_fold = 2
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

weights = [1, 1.5]*n_folds
predict_proba = np.average(checkpoint_predictions, weights=weights, axis=0)

threshold = 0.5
submiss.label = (predict_proba > threshold).astype(np.int)
submiss.to_csv(os.path.join(submiss_dir, 'gru_fasttext.csv'), index=False)     # 0.89468