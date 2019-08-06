import torch
from torch import nn

class QueryMLP(nn.Module):

    def __init__(self, sizes, num_heads, output_size, bias=True, batchnorm=False, initializer_range=0.02):
        super().__init__()
        assert len(sizes) >= 2
        assert num_heads*output_size == sizes[-1]
        self.num_heads = num_heads
        self.input_size = sizes[0]
        self.output_size = output_size
        self.initializer_range = initializer_range
        mlp = []
        pairs = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        for i, (dim_in, dim_out) in enumerate(pairs):
            mlp.append(nn.Linear(dim_in, dim_out, bias=bias))
            if batchnorm:
                mlp.append(nn.BatchNorm1d(dim_out))
            else:
                mlp.append(nn.LayerNorm(dim_out))
            if i < len(pairs)-1:
                mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)
        self.apply(self.reset_parameters)

    def reset_parameters(self, module):
        """ 
        Initialize the weights.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            #nn.init.kaiming_uniform_(module.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        assert x.shape[-1] == self.input_size
        x = x.contiguous().view(-1, self.input_size) if x.dim() > 2 else x
        bs = len(x)
        return self.mlp(x).view(bs * self.num_heads, self.output_size)