
todo
- tokenizer作成（sentencepiece, char)
- データセットのダウンロード&トークン化(train.binの作成)
- train

## tokenizer作成
sentencepiece_learn_vocab.py
(charOscar4e6.modelはgitに含まれているため実行不要)

charで分割するシンプルなtokenizerをsentencepieceで作成


## データセット作成(data/oscar_deduplicated_ja_slice)

data/openwebtext/prepare.pyを参考に作成

prepare.pyだとデータが大きすぎて止まったり、ローカルPCでつらみがあったので、prepare_split.pyで複数のtrain{i}.binを作成し、integrate_bin.pyで１つのtrain.binに統合して作成。

## 学習
train.py

１つのminibatchのみ使用するように113行あたりでベタガキ
configとしてはconfig/train_gpt2_oscarja_charvocab.py　を使用するように実行時に引数で与える（なぜか消えていたので記憶を頼りに書き直しているので一部違う可能性あり)