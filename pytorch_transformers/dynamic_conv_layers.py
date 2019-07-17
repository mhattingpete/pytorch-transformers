from torch import nn
from torch.nn import functional as F
import torch

class LightweightConv1dTBC(nn.Module):
    '''Lightweight Convolution assuming the input is TxBxC
    Args:
        input_size: # of channels of the input
        kernel_size: convolution channels
        padding_l: padding to the left when using "same" padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout: the drop rate of the DropConnect to drop the weight
        weight_softmax: normalize the weight with softmax before the convolution
        bias: use bias
    Shape:
        Input: TxBxC, i.e. (timesteps, batch_size, input_size)
        Output: TxBxC, i.e. (timesteps, batch_size, input_size)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(input_size)`
    '''
    def __init__(self, input_size, kernel_size=1, padding_l=None, num_heads=1,
                 weight_dropout=0., bias=False):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_dropout = weight_dropout

        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None

        self.reset_parameters()

        self.onnx_trace = False

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, x):
        '''Assuming the input, x, of the shape T x B x C and producing an output in the shape T x B x C
        args:
            x: Input of shape T x B x C, i.e. (timesteps, batch_size, input_size)
        '''
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size
        weight = self.weight.view(H, K)
        weight = F.softmax(weight, dim=1).type_as(weight)
        weight = weight.view(1, H, K).expand(T*B, H, K).contiguous()
        weight = weight.view(T, B*H, K).transpose(0, 1)
        x = x.view(T, B*H, R).transpose(0, 1)
        P = self.padding_l
        if K > T and P == K-1:
            weight = weight.narrow(2, K-T, T)
            K, P = T, T-1
        # turn the convolution filters into band matrices
        weight_expanded = weight.new_zeros(B*H, T, T+K-1, requires_grad=False)
        weight_expanded.as_strided((B*H, T, K), (T*(T+K-1), T+K, 1)).copy_(weight)
        weight_expanded = weight_expanded.narrow(2, P, T)
        weight_expanded = F.dropout(weight_expanded, self.weight_dropout, training=self.training)
        output = torch.bmm(weight_expanded, x)
        output = output.transpose(0, 1).contiguous().view(T, B, C)
        return output, weight_expanded

class DynamicConv1dTBC(nn.Module):
    '''
    Dynamic lightweight convolution taking T x B x C inputs
    Args:
        input_size: # of channels of the input
        kernel_size: convolution channels
        padding_l: padding to the left when using "same" padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout: the drop rate of the DropConnect to drop the weight
        bias: use bias
        conv_bias: bias of the convolution
    Shape:
        Input: TxBxC, i.e. (timesteps, batch_size, input_size)
        Output: TxBxC, i.e. (timesteps, batch_size, input_size)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(input_size)`
    '''
    def __init__(self, input_size, kernel_size=1, padding_l=None, num_heads=1,
                 weight_dropout=0., bias=False, conv_bias=False):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_dropout = weight_dropout
        self.bias = bias

        self.weight_linear = nn.Linear(self.input_size, num_heads * kernel_size * 1, bias=bias)
        if conv_bias:
            self.conv_bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.conv_bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_linear.weight)
        if self.bias:
            nn.init.constant_(self.weight_linear.bias, 0.)
        if self.conv_bias is not None:
            nn.init.constant_(self.conv_bias, 0.)

    def forward(self, x):
        '''Turn the convolution filters into band matrices and do matrix multiplication.
        This is faster when the sequence is short, but less memory efficient.
        This is not used in the decoder during inference.
        '''
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size
        weight = self.weight_linear(x).view(T*B*H, -1)
        weight = weight.narrow(1, 0, K).contiguous()
        weight = weight.view(T, B*H, K).transpose(0, 1)
        x = x.view(T, B*H, R).transpose(0, 1)
        # turn the convolution filters into band matrices
        weight_expanded = weight.new(B*H, T, T+K-1).fill_(float('-inf'))
        weight_expanded.as_strided((B*H, T, K), (T*(T+K-1), T+K, 1)).copy_(weight)
        weight_expanded = weight_expanded.narrow(2, self.padding_l, T)
        # normalize the weight over valid positions like self-attention
        weight_expanded = F.softmax(weight_expanded, dim=2)
        weight_expanded = F.dropout(weight_expanded, self.weight_dropout, training=self.training, inplace=False)
        output = torch.bmm(weight_expanded, x)
        output = output.transpose(0, 1).contiguous().view(T, B, C)
        return output, weight_expanded

class LightweightConv(LightweightConv1dTBC):
    def __init__(self,config):
        kernel_size = config.kernel_size
        padding_l = kernel_size // 2 if kernel_size % 2 == 1 else ((kernel_size - 1) // 2, kernel_size // 2)
        super().__init__(input_size=config.hidden_size, kernel_size=kernel_size, padding_l=padding_l,
        num_heads=config.num_attention_heads, bias=False, weight_dropout=config.attention_probs_dropout_prob)
        self.output_attentions = config.output_attentions
        if config.use_glu:
           self.input_projection = nn.Sequential(nn.Linear(config.hidden_size,2*config.hidden_size), nn.GLU())
        else:
           self.input_projection = nn.Sequential(nn.Linear(config.hidden_size,config.hidden_size))

    def forward(self, hidden_states, attention_mask, head_mask=None):
        hidden_states = self.input_projection(hidden_states)
        #if attention_mask is not None:
        #   hidden_states = hidden_states.masked_fill(attention_mask, 0)
        context_layer, attention_probs = super().forward(hidden_states)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs

class DynamicConv(DynamicConv1dTBC):
    def __init__(self,config):
        kernel_size = config.kernel_size
        padding_l = kernel_size // 2 if kernel_size % 2 == 1 else ((kernel_size - 1) // 2, kernel_size // 2)
        super().__init__(input_size=config.hidden_size, kernel_size=kernel_size, padding_l=padding_l,
        num_heads=config.num_attention_heads, bias=False, conv_bias=False, weight_dropout=config.attention_probs_dropout_prob)
        self.output_attentions = config.output_attentions
        if config.use_glu:
           self.input_projection = nn.Sequential(nn.Linear(config.hidden_size,2*config.hidden_size), nn.GLU())
        else:
           self.input_projection = nn.Sequential(nn.Linear(config.hidden_size,config.hidden_size))

    def forward(self, hidden_states, attention_mask, head_mask=None):
        hidden_states = self.input_projection(hidden_states)
        #if attention_mask is not None:
        #   hidden_states = hidden_states.masked_fill(attention_mask, 0)
        context_layer, attention_probs = super().forward(hidden_states)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs
