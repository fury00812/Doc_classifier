# Doc_classifier
複数の手法による文書分類モデルの実装

## ナイーブベイズ 
### training
```
python train.py --model naive_bayes --train_data $DATA_PATH/data.json --save_path $OUT_PATH/NaiveBayes/naive_bayes.pkl
```
### prediction
```
python predict.py --model $OUT_PATH/NaiveBayes/naive_bayes.pkl
```

## ニューラルネットワーク(bag-of-words)
scikit-learnのMLPClassifierを用いたニューラルネットワークによる多クラス分類
### training
```
python train.py --model mlp_bow --train_data $DATA_PATH/data.json --save_path $OUT_PATH/MLP/mlp_bow.pkl
```
### prediction
```
python predict.py --model $OUT_PATH/MLP/mlp_bow.pkl
```

## ニューラルネットワーク(tf-idf)
上記に加えtf-idfを用いた多クラス分類
### training
```
python train.py --model mlp_tfidf --train_data $DATA_PATH/data.json --save_path $OUT_PATH/MLP/mlp_tfidf.pkl
```
### prediction
```
python predict.py --model $OUT_PATH/MLP/mlp_tfidf.pkl
```
