import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

import Constants

class Encoder(nn.Module):
    """
        Args:
            input: seq_len, batch
        Returns:
            attn: batch, seq_len, hidden_size
            outputs: batch, seq_len, hidden_size

    """
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
        self.padding = ((opt.kernel_size -1) / 2, 0)
        self.layers = opt.enc_layers

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.affine = nn.Linear(self.embedding_size, 2*self.hidden_size)
        self.softmax = nn.Softmax()

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel, self.stride,self.padding)

        self.mapping = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        # self.attn = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_size * 2)

    def forward(self, input):
        inputs = self.embedding(input[0])
        _inputs = inputs.view(-1, inputs.size(2)) 
        _outputs = self.affine(_inputs)
        _outputs = _outputs.view(inputs.size(0), inputs.size(1), -1).t() 
        outputs = _outputs
        for i in range(self.layers):
            outputs = outputs.unsqueeze(1) # batch, 1, seq_len, 2*hidden
            outputs = self.conv(outputs) # batch, out_channels, seq_len, 1
            outputs = F.relu(outputs)
            outputs = outputs.squeeze(3).transpose(1,2) # batch, seq_len, 2*hidden
            A, B = outputs.split(self.hidden_size, 2) # A, B: batch, seq_len, hidden
            A2 = A.contiguous().view(-1, A.size(2)) # A2: batch * seq_len, hidden
            B2 = B.contiguous().view(-1, B.size(2)) # B2: batch * seq_len, hidden
            attn = torch.mul(A2, self.softmax(B2)) # attn: batch * seq_len, hidden
            attn2 = self.mapping(attn) # attn2: batch * seq_len, 2 * hidden
            outputs = attn2.view(A.size(0), A.size(1), -1) # outputs: batch, seq_len, 2 * hidden
        # outputs = torch.sum(outputs, 2).squeeze(2)
        out = attn2.view(A.size(0), A.size(1), -1) + _outputs # batch, seq_len, 2 * hidden_size
        # print "_outputs", _outputs
        # print "out", out

        return attn, out

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)



class Decoder(nn.Module):
    """
    Decoder
        Args:
            Input: seq_len, batch_size
        return:
            out:
    """

    def __init__(self, opt, vocab_size):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = opt.embedding_size
        self.hidden_size = opt.hidden_size


        self.in_channels = 1
        self.out_channels = opt.hidden_size * 2
        self.kernel_size = opt.kernel_size
        self.kernel = (opt.kernel_size, opt.hidden_size * 2)
        self.stride = 1
        self.padding = (opt.kernel_size - 1, 0)
        self.layers = 1 #opt.dec_layers

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.affine = nn.Linear(self.embedding_size, 2 * self.hidden_size)
        self.softmax = nn.Softmax()

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel, self.stride, self.padding)
        
        self.mapping = nn.Linear(self.hidden_size, 2*self.hidden_size)

        self.softmax = nn.Softmax()
    # attn_src: src_seq_len, hidden_size
    def forward(self, source, target, enc_attn, source_seq_out):
        inputs = self.embedding(target)
        _inputs = inputs.view(-1, inputs.size(2))
        outputs = self.affine(_inputs)
        outputs = outputs.view(inputs.size(0), inputs.size(1), -1).t()
        for i in range(self.layers):
            outputs = outputs.unsqueeze(1) # batch, 1, seq_len, 2*hidden
            outputs = self.conv(outputs) # batch, out_channels, seq_len + self.kernel_size - 1, 1
            outputs = outputs.narrow(2, 0, outputs.size(2)-self.kernel_size) # remove the last k elements

            # This is the residual connection,
            # for the output of the conv will add kernel_size/2 elements 
            # before and after the origin input
            if i > 0:
                conv_out = conv_out + outputs

            outputs = F.relu(outputs)
            outputs = outputs.squeeze(3).transpose(1,2) # batch, seq_len, 2*hidden
            A, B = outputs.split(self.hidden_size, 2) # A, B: batch, seq_len, hidden
            A2 = A.contiguous().view(-1, A.size(2)) # A2: batch * seq_len, hidden
            B2 = B.contiguous().view(-1, B.size(2)) # B2: batch * seq_len, hidden
            dec_attn = torch.mul(A2, self.softmax(B2)) # attn: batch * seq_len, hidden

            dec_attn2 = self.mapping(dec_attn)
            dec_attn2 = dec_attn2.view(A.size(0), A.size(1), -1)

            enc_attn = enc_attn.view(A.size(0), -1, A.size(2)) # enc_attn1: batch, seq_len_src, hidden_size
            dec_attn = dec_attn.view(A.size(0), -1, A.size(2)) # dec_attn1: batch, seq_len_tgt, hidden_size

            

            _attn_matrix = torch.bmm(dec_attn, enc_attn.transpose(1,2)) # attn_matrix: batch, seq_len_tgt, seq_len_src
            attn_matrix = self.softmax(_attn_matrix.view(-1, _attn_matrix.size(2)))
            attn_matrix = attn_matrix.view(_attn_matrix.size(0), _attn_matrix.size(1), -1) # normalized attn_matrix: batch, seq_len_tgt, seq_len_src

            attns = torch.bmm(attn_matrix, source_seq_out) # attns: batch, seq_len_tgt, 2 * hidden_size
            outputs = dec_attn2 + attns # outpus: batch, seq_len_tgt - 1, 2 * hidden_size
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
        # attn: batch, seq_len, hidden
        # out: batch, seq_len, 2 * hidden_size
        attn, source_seq_out = self.encoder(source)

        # batch, seq_len_tgt, hidden_size
        out = self.decocer(source, target, attn, source_seq_out)

        return out

