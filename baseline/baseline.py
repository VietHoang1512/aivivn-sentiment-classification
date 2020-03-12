import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from underthesea import word_tokenize
from sklearn.linear_model import LogisticRegression
from pyvi import ViTokenizer

main_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(main_dir, 'data')
submiss_dir = os.path.join(main_dir, 'submission')

def preprocess(text, pyvi=False):
    text = re.sub('\n', ' ', text)
    if pyvi:
        text = ViTokenizer.tokenize(text)
    else:
        text = word_tokenize(text, format="text")
    words_only = re.findall('\w+', text)   
    text = ' '.join(words_only)
    text = re.sub('[\d ]+', ' ', text).lower()  
    return text


train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'), index_col=0)
test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'), index_col=0)
submiss = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

train_clean = train_data.text.apply(preprocess)
test_clean = test_data.text.apply(preprocess)

tfidf = TfidfVectorizer(ngram_range=(1,2), 
                        #min_df=2
                        )

train_vec = tfidf.fit_transform(train_clean)
test_vec = tfidf.transform(test_clean)

model = LogisticRegression(solver='lbfgs')
model.fit(train_vec, train_data.label)

submiss.label = model.predict(test_vec)
submiss.to_csv(os.path.join(submiss_dir, 'baseline.csv'), index=False)         # 0.87814