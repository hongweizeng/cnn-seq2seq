import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

import Constants

class Encoder(nn.Module):
    def __init__(self, opt, vocab_size):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = opt.embedding_size
        self.hidden_size = opt.hidden_size

        self.in_channels = 1
        self.out_channels = opt.hidden_size * 2
        self.kernel_size = opt.kernel_size
        self.kernel = (opt.kernel_size, opt.hidden_size * 2)
        self.stride = 1
        self.padding = (opt.kernel_size / 2, 0)
        self.layers = opt.enc_layers

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.affine = nn.Linear(self.embedding_size, 2*self.hidden_size)
        self.softmax = nn.Softmax()

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride,self.padding)

        self.mapping = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        # self.attn = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_size * 2)

    def forward(self, input):
        inputs = self.embedding(input[0])
        _inputs = inputs.view(-1, inputs.size(2))
        outputs = self.affine(_inputs)
        outputs = outputs.view(inputs.size(0), inputs.size(1), -1).t()

        for i in range(self.layers):
            outputs = outputs.unsqueeze(1)
            outputs = self.conv(outputs)
            outputs = F.relu(outputs)
            outputs = outputs.squeeze(3).transpose(1,2)
            A, B = outputs.split(self.hidden_size, 2)
            A2 = A.contiguous().view(-1, A.size(2))
            B2 = B.contiguous().view(-1, B.size(2))
            attn = torch.mul(A2, self.softmax(B2))
            _attn = self.mapping(attn)

            outputs = _attn.view(A.size(0), A.size(1), -1)
        outputs = torch.sum(outputs, 1).squeeze(1)

        # attn_inputs = . + .

        return outputs, attn

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)



class Decoder(nn.Module):
    """
        Encoder:
        Args:
        Input: seq_len, batch_size
        return:
        """

    def __init__(self, opt, vocab_size):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = opt.embedding_size
        self.hidden_size = opt.hidden_size


        self.in_channels = 1
        self.out_channels = 1#opt.out_channels
        # self.kernel_size = opt.kernel_size
        self.kernel = (opt.kernel_size, opt.hidden_size * 2)
        self.stride = 1
        self.padding = opt.kernel_size - 1
        self.layers = 1

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.affine = nn.Linear(self.embedding_size, 2 * self.hidden_size)
        self.softmax = nn.Softmax()

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel, self.stride, self.padding)
        self.mapping = nn.Linear(self.hidden_size, 2*self.hidden_size)

    # attn_src: src_seq_len, hidden_size
    def forward(self, source, attn_src, target):
        inputs = self.embedding(target)
        outputs = self.affine(inputs)

        for i in torch.range(self.layers):
            conv_out = self.conv(outputs)
            conv_out = conv_out[:self.kernel_size - 1]  # remove the last k - 1 elements
            A, B = conv_out.split(self.hidden_size)
            conv_out = torch.mul(A, self.softmax(B))

            # This is the residual connection,
            # for the output of the conv will add kernel_size/2 elements 
            # before and after the origin input
            if i > 0:
                conv_out = conv_out + outputs

            outputs = self.mapping(conv_out) + outputs

            attn = torch.mm(outputs, attn_src.t()) # trt_seq_len, src_seq_len
            attn = nn.Softmax(attn) # trt_seq_len, src_seq_len;

            source_attn = self.mapping(attn_src) + source # src_seq_len, 2 * hidden_size
            c = torch.mm(attn, source_attn) # trt_seq_len, 2 * hidden_size

            outputs = outputs + c

        return outputs

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)


class NMTModel(nn.Module):
    """
    NMTModel:
    Input:
        encoder:
        decoder:
        attention:
        generator:
    return:
    """
    def __init__(self, encoder, decocer):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decocer = decocer
    
    def forward(self, source, target):
        attn_src, attn_inputs = self.encoder(source) # src_seq_len, embedding_size
        
        out = self.decocer(source, attn_src, target) # trt_seq_len, embedding_size

        return out

