from __future__ import division
import torch
import math
import random

import torch
from torch.autograd import Variable

# def batch_iter(images, captions, batch_size, num_epochs, train=True, shuffle=False):
#     """
#     Generates a batch iterator for a dataset.
#     data: data["train"]["x"], data["train"]["y"] or data["test"]["x"], data["test"]["y"]
#     """
#     assert len(images) == len(captions)
#     data_size = len(images)
#     num_batches_per_epoch = int(data_size/batch_size) + 1
#     for epoch in range(num_epochs):
#         # Sort the data at each epoch by sentence length
#         if shuffle:
#             shuffle_indices = torch.randperm(torch.arange(data_size))
#             shuffled_x, shuffled_y = images[shuffle_indices], captions[shuffle_indices]
#         else:
#             shuffled_x, shuffled_y = images, captions
#         for batch_num in range(num_batches_per_epoch):
#             start_index = batch_num * batch_size
#             end_index = min((batch_num + 1) * batch_size, data_size)
#             yield shuffled_x[start_index:end_index], shuffled_y[start_index:end_index]

class Dataset(object):

    def __init__(self, images, captions, batch_size, cuda, volatile=False):

        self.images = images
        self.captions = captions
        assert (len(self.images) == len(self.captions))
        self.batch_size = batch_size
        self.numBatches = math.ceil(len(self.images)/batch_size)
        self.volatile = volatile
        self.cuda = cuda

    def _batchify(self, data, align_right=False, PADDING_TOKEN=0):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(PADDING_TOKEN)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])
        return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        images = self._batchify(
            self.images[index*self.batchSize:(index+1)*self.batchSize])

        captions = self._batchify(
            self.captions[index*self.batchSize:(index+1)*self.batchSize])


        # within batch sorting by decreasing length for variable length rnns
        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        return wrap(images), wrap(captions)

    def __len__(self):
        return self.numBatches


    def shuffle(self):
        data = list(zip(self.images, self.captions))
        self.images, self.captions = zip(*[data[i] for i in torch.randperm(len(data))])

