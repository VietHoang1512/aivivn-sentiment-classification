import os
import pandas as pd
from tqdm import tqdm
from underthesea import sentiment


main_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(main_dir, 'data')
submiss_dir = os.path.join(main_dir, 'submission')

test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'), index_col=0)
submiss = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

submiss.label = [sentiment(text) for text in tqdm(test_data.text, position=0, leave=False)]

map_sentiment = {'positive':0, 
                 'negative':1}


submiss.label = submiss.label.map(map_sentiment)

submiss.to_csv(os.path.join(submiss_dir, 'underthesea.csv'), index=False)      # 0.88966