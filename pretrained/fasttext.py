import os
import pandas as pd
import re
import fasttext

main_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(main_dir, 'data')
submiss_dir = os.path.join(main_dir, 'submission')

def preprocess(text):
    words_only = re.findall('\w+', text)   
    text = ' '.join(words_only)
    text = re.sub('[\d ]+', ' ', text).lower()  
    return text

train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'), index_col=0)
test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'), index_col=0)
submiss = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

for index, comment in train_data.iterrows():
    train_data.loc[index,'formatted_text'] = f'__label__{comment.label} ' + preprocess(comment.text)

test_data['formatted_text'] = test_data.text.apply(preprocess)

with open ('text.train', 'w') as f:
    f.write('\n'.join(train_data.formatted_text))
    
model = fasttext.train_supervised(input="text.train", lr=0.005, epoch=50, wordNgrams=2)

submiss.label = [model.predict(text)[0][0][-1] for text in test_data.formatted_text]
submiss.label = submiss.label.astype(int)

submiss.to_csv(os.path.join(submiss_dir, 'fasttext.csv'), index=False)         # 0.88359