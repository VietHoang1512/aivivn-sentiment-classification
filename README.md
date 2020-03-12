# Aivivn sentiment classification
- The official competition had been held [here](https://www.aivivn.com/contests/1)

- Late submission can be evaluated in [training ground](https://www.aivivn.com/contests/6)

- My solution has achieved the #5 position on the training ground (a little bit higher than the official competition's silver medalist's score)

![Picture1](https://user-images.githubusercontent.com/52401767/76520473-899e9e80-6495-11ea-9f9c-ce07b1155ca0.png)

## 1/ Implementation explain
Position on the official competition and training ground leader board (calculated by f1-score) 

| Model          | F1-score | Official competition | Training ground|
| -------------- |:--------:| :-------------------:|:--------------:|                
| Baseline       | 0.87814  | 78                   | 35             |
| Fasttext       | 0.88359  | 58                   | 27             |
| Underthesea    | 0.88966  | 26                   | 22             |
| GRU & fasttext | 0.89468  | 6                    | 12             | 
| LSTM & word2vec| 0.89786  | 2                    | 5              |

### 1.1/ [Baseline](https://github.com/KingLeo2000/aivivn-sentiment-classification/blob/master/baseline/baseline.py)
Naive approach for any text classification task (TF-IDF + Logistic Regression)
### 1.2/ [Fasttext](https://github.com/KingLeo2000/aivivn-sentiment-classification/blob/master/pretrained/fasttext.py)
Pretrained [source](https://fasttext.cc/docs/en/supervised-tutorial.html)
### 1.3/ [Underthesea]
Pretrained source and [usage](https://github.com/undertheseanlp/underthesea#7-sentiment-analysis)
### 1.4/ [GRU & fasttext]
Build a neural network up on [fasttext](https://fasttext.cc/docs/en/crawl-vectors.html) embdedding layer (and continue training these weight)
### 1.5/[LSTM & word2vec]
Build a neural network up on [sonvx word2vec](https://github.com/sonvx/word2vecVN) and [ETNLP word2vec](https://github.com/vietnlp/etnlp) (freeze these weight when training)

The natural and step-by-step approach can be found [here](https://drive.google.com/drive/folders/1wTbvJnc62440yaz75yexzWrof1Mee7ST)
## References:
- [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
- [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- [How does FastText classifier work under the hood?](https://towardsdatascience.com/fasttext-bag-of-tricks-for-efficient-text-classification-513ba9e302e7)
- [Phân loại văn bản tự động bằng Machine Learning như thế nào?](https://viblo.asia/p/phan-loai-van-ban-tu-dong-bang-machine-learning-nhu-the-nao-4P856Pa1ZY3)

## Last updated : 12/3/2020 (on progress)
