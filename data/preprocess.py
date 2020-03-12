from tqdm import tqdm
import pandas as pd
import re

with open('train.crash', 'r', encoding='utf8') as f:
    raw_train = f.read().strip().split('\n\ntrain_')
raw_train[0] = raw_train[0].strip('train_')

train_ids = []
train_labels = []
train_raw = []
for train in tqdm(raw_train):
    train_ids.append(int(train[:6]))
    train_labels.append(int(train[-1]))
    train_raw.append(re.sub('\n', '.', train[8:-3]))
    
train = pd.DataFrame({'text':train_raw,
                      'label':train_labels},
                     index=train_ids)
train.to_csv('train.csv')

with open('test.crash', 'r', encoding='utf8') as f:
    raw_test = f.read().strip().split('\n\ntest_')
raw_test[0] = raw_test[0].strip('test_')

test_ids = []
test_raw = []
for test in tqdm(raw_test):
    test_ids.append(int(test[:6]))
    test_raw.append(re.sub('\n', '.',test[8:-1]))
    
test = pd.DataFrame({'text':test_raw},
                     index=test_ids)
test.to_csv('test.csv')