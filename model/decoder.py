import torch as t
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, params, highway):
        super(Decoder, self).__init__()

        self.params = params
        self.hw1 = highway
        self.encoding_rnn = nn.LSTM(input_size=self.params.word_embed_size,
                                       hidden_size=self.params.encoder_rnn_size,  # the board of hidden later?
                                       num_layers=self.params.encoder_num_layers,   # the number of layers
                                       batch_first=True,    # 批处理
                                       bidirectional=True)  # 双向

        self.decoding_rnn = nn.LSTM(input_size=self.params.latent_variable_size
                                        + self.params.word_embed_size,
                                       hidden_size=self.params.decoder_rnn_size,
                                       num_layers=self.params.decoder_num_layers,
                                       batch_first=True)
        self.h_to_initial_state = nn.Linear(self.params.encoder_rnn_size * 2,   # hidden to initial
            self.params.decoder_num_layers * self.params.decoder_rnn_size)
        # 一种线性变换 Linear的创建需要两个参数，inputSize 和 outputSize
        # inputSize：输入节点数
        # outputSize：输出节点数
        # 第零阶张量（ {\displaystyle r=0} r=0）为标量，第一阶张量（ {\displaystyle r=1} r=1）为矢量， 第二阶张量（ {\displaystyle r=2} r=2）则成为矩阵
        # weight : Tensor ， outputSize × inputSize
        # bias: Tensor ，outputSize
        # y = wx+b
        self.c_to_initial_state = nn.Linear(self.params.encoder_rnn_size * 2,   # context to initial
            self.params.decoder_num_layers * self.params.decoder_rnn_size)

        self.fc = nn.Linear(self.params.decoder_rnn_size, self.params.vocab_size)

    def build_initial_state(self, input):
        [batch_size, seq_len, embed_size] = input.size()    # torch.Size([5, 3])
        input = input.view(-1, embed_size)  # change the column to embed_size
        input = self.hw1(input) # highway
        input = input.view(batch_size, seq_len, embed_size)  # change to the initial state

        _, cell_state = self.encoding_rnn(input)    # put input into the encoder,_ means a temporary variable
        [h_state, c_state] = cell_state             # a vector
        h_state = h_state.view(self.params.encoder_num_layers, 2, batch_size, self.params.encoder_rnn_size)[-1]
        c_state = c_state.view(self.params.encoder_num_layers, 2, batch_size, self.params.encoder_rnn_size)[-1]
        # the above state change the row
        
        # with shapes (batch, 2 * encoder_rnn_size)
        h_state = h_state.permute(1,0,2).contiguous().view(batch_size, -1)
        c_state = c_state.permute(1,0,2).contiguous().view(batch_size, -1)
        # the above 将tensor的维度换位。
        # contiguous：view只能用在contiguous的variable上。如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy。
        '''
            x = torch.Tensor(2,3)
            y = x.permute(1,0)
            y.view(-1) # 报错，因为x和y指针指向相同
            y = x.permute(1,0).contiguous()
            y.view(-1) # OK
        '''
        # shapes (num_layers, batch, decoder_rnn_size)        
        h_initial = self.h_to_initial_state(h_state).view(batch_size,
            self.params.decoder_num_layers, self.params.decoder_rnn_size).permute(1,0,2).contiguous()
        c_initial = self.c_to_initial_state(c_state).view(batch_size, 
            self.params.decoder_num_layers, self.params.decoder_rnn_size).permute(1,0,2).contiguous()

        return (h_initial, c_initial)


    def forward(self, encoder_input, decoder_input, z, drop_prob, initial_state=None):
        """
        :param encoder_input: tensor with shape of [batch_size, seq_len, embed_size]
        :param decoder_input: tensor with shape of [batch_size, seq_len, embed_size]
        :param z: sequence context with shape of [batch_size, latent_variable_size]
        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout
        :param initial_state: initial state of decoder rnn

        :return: unnormalized logits of sentense words distribution probabilities
                    with shape of [batch_size, seq_len, vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """
        
        if initial_state is None:
            # build initial context with source input.
            assert not encoder_input is None    # if encoder_input is none , then give AssertionError
            initial_state = self.build_initial_state(encoder_input)

        [batch_size, seq_len, _] = decoder_input.size() 
        '''
            a.size()
        >>> torch.Size([2, 3])
            [b,c] = a.size()
        >>> b = 2
        >>> c = 3
        '''

        # print(initial_state[0].size())
        '''
            decoder rnn is conditioned on context via additional bias = W_cond * z to every input token
        '''
        decoder_input = F.dropout(decoder_input, drop_prob) # dropout,drop_prob means 不保留节点数的比例

        z = t.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.latent_variable_size)
        # seq_len z connect together as the input metrix
        # Cat对数据沿着某一维度进行拼接。cat后数据的总维数不变.比如下面代码对两个2维tensor（分别为2*3,1*3）进行拼接，拼接完后变为3*3还是2维的tensor。
        # torch.cat((x,y),0) 0 means row , 1 means column ,2 means z?
        decoder_input = t.cat([decoder_input, z], 2)

        rnn_out, final_state = self.decoding_rnn(decoder_input, initial_state)

        rnn_out = rnn_out.contiguous().view(-1, self.params.decoder_rnn_size)
        result = self.fc(rnn_out)
        result = result.view(batch_size, seq_len, self.params.vocab_size)

        return result, final_state
