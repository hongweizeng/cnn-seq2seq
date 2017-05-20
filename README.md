# cnn-seq2seq

# PyTorch Implementation of [Convlutional Sequence to Sequence learning](https://arxiv.org/abs/1705.03122)


## 1. Data

### **_[the open parallel corpus](http://opus.lingfil.uu.se/)_**

1.1 [EUROPARL v7 - European Parliament Proceedings](http://opus.lingfil.uu.se/Europarl.php) ([Europarlv7.tar.gz](http://opus.lingfil.uu.se/download.php?f=Europarl/Europarlv7.tar.gz) - 8.4 GB)


## 2. Preprocess

### test files
\# -files: it is a directory, and contains train.src, train.tgt, valid.src, valid.tgt, test.src, test.tgt

\# -save_data: it save the .train.pt file. 
python preprocess.py -files /home/zeng/conversation/OpenNMT-py/data/test/ -save_data /home/zeng/data/test/test