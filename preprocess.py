import argparse
import torch
import re
import itertools
from collections import Counter
import numpy as np
import Constants

parser = argparse.ArgumentParser(description='preprocess.py')

##
## **Preprocess Options**
##

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-dataset', type=str, default="",
                    help="dataset available, [f8k|f30k|coco].")
parser.add_argument('-image_train_file', type=str, default="/home/zeng/parlAI/data/f8k/f8k_train_ims.npy",
                    help="Path to the training source data")
parser.add_argument('-caption_train_file', type=str, default="/home/zeng/parlAI/data/f8k/f8k_train_caps.txt",
                    help="Path to the training target data")
parser.add_argument('-image_valid_file', type=str, default="/home/zeng/parlAI/data/f8k/f8k_dev_ims.npy",
                    help="Path to the training source data")
parser.add_argument('-caption_valid_file', type=str, default="/home/zeng/parlAI/data/f8k/f8k_dev_caps.txt",
                    help="Path to the training target data")
parser.add_argument('-image_test_file', type=str, default="/home/zeng/parlAI/data/f8k/f8k_test_ims.npy",
                    help="Path to the training source data")
parser.add_argument('-caption_test_file', type=str, default="/home/zeng/parlAI/data/f8k/f8k_test_caps.txt",
                    help="Path to the training target data")

parser.add_argument('-save_data', type=str, default="./",
                    help="Output file for the prepared data")

parser.add_argument('-maximum_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")

parser.add_argument('-vocab',
                    help="Path to an existing vocabulary")

parser.add_argument('-seq_length', type=int, default=50,
                    help="Maximum sequence length")
parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=1000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

torch.manual_seed(opt.seed)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def build_vocab(sequence, maximum_vocab_size=50000):
    word_count = Counter(itertools.chain(*sequence)).most_common(maximum_vocab_size)
    word2count = dict([(word[0], word[1]) for word in word_count])
    
    word2index = dict([(word, index+4) for index, word in enumerate(word2count)])
    word2index[0], word2index[1], word2index[2], word2index[3] = 0, 1, 2, 3
    index2word = dict([(index+4, word) for index, word in enumerate(word2count)])
    word2index[Constants.PAD_WORD], word2index[Constants.BOS_WORD], word2index[Constants.EOS_WORD], word2index[Constants.UNK_WORD] = \
    Constants.PAD, Constants.BOS, Constants.EOS, Constants.UNK

    index2word[Constants.PAD], index2word[Constants.BOS], index2word[Constants.EOS], index2word[Constants.UNK] = \
    Constants.PAD_WORD, Constants.BOS_WORD, Constants.EOS_WORD, Constants.UNK_WORD
    return word2count, word2index, index2word

def makeData(images, captions, word2index, shuffle=opt.shuffle):
    assert len(images) == len(captions)
    images = torch.FloatTensor(images)
    for idx in range(len(images)):
        captions[idx] = torch.LongTensor([word2index[word] if word in word2index else Constants.PAD for word in captions[idx]])

    if shuffle == 1:
        print "... shuffling sentences"
        perm = torch.randperm(len(images))
        images = [images[idx] for idx in perm]
        captions = [captions[idx] for idx in perm]

    return images, captions


def load_image_and_caption(image_file, caption_file):
    """
    Load captions and image features
    Possible options: f8k, f30k, coco
    """
    captions = []
    # Image
    images = np.load(image_file)

    with open(caption_file, "rb") as f:
        for line in f:
            captions.append(line.strip().split())

    return images, captions


def main():

    print "Loading data ..."

    image_train, caption_train = load_image_and_caption(opt.image_train_file, opt.caption_train_file)
    image_valid, caption_valid = load_image_and_caption(opt.image_valid_file, opt.caption_valid_file)
    image_test, caption_test = load_image_and_caption(opt.image_test_file, opt.caption_test_file)


    word2count, word2index, index2word = build_vocab(caption_train + caption_valid + caption_test, opt.maximum_vocab_size)

    print('Preparing training ...')
    train = {}
    train['image'], train['caption'] = makeData(image_train, caption_train, word2index)

    print('Preparing validation ...')
    valid = {}
    valid['image'], valid['caption'] = makeData(image_valid, caption_valid, word2index)

    print('Preparing testing ...')
    valid = {}
    valid['image'], valid['caption'] = makeData(image_test, caption_test, word2index)

    print "saving data to \'" + opt.dataset + ".train.pt\'..."
    save_data = {
        "train": train,
        "valid": valid,
        "test": valid,
        "word2index": word2index
    }
    torch.save(save_data, opt.dataset + ".train.pt")

if __name__ == "__main__":
    main()
