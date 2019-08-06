import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .utils import get_gaussian_keys, get_uniform_keys
from .query import QueryMLP

MemoryDefaultDict = {
    "mem_sparse" : False,
    "mem_k_dim" : 16,
    "mem_v_dim" : -1,
    "mem_heads" : 2,
    "mem_knn" : 8,
    "mem_keys_type" : "uniform",
    "mem_n_keys" : 256,
    "mem_keys_normalized_init" : False,
    "mem_keys_learn" : True,
    "mem_query_layer_sizes" : "0,128,0",
    "mem_query_bias" : True,
    "mem_query_batchnorm" : False,
    "mem_input_dropout" : 0,
    "mem_query_dropout" : 0,
    "mem_value_dropout" : 0
}

def add_memory_config(config):
    defaults = MemoryDefaultDict
    for key, value in defaults.items():
        if key not in config.__dict__:
            config.__dict__[key] = value
    return config

class MemoryLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        mem_config = add_memory_config(config)
        self.input_size = mem_config.hidden_size
        output_size = mem_config.hidden_size
        query_layer_sizes = [int(x) for x in filter(None, mem_config.mem_query_layer_sizes.split(','))]
        
        self.size = mem_config.mem_n_keys ** 2
        self.v_dim = mem_config.mem_v_dim if mem_config.mem_v_dim > 0 else output_size
        self.learn_keys = mem_config.mem_keys_learn
        self.keys_normalized_init = mem_config.mem_keys_normalized_init
        self.keys_type = mem_config.mem_keys_type
        self.k_dim = mem_config.mem_k_dim
        self.heads = mem_config.mem_heads
        self.knn = mem_config.mem_knn
        self.input_dropout = mem_config.mem_input_dropout
        self.query_dropout = mem_config.mem_query_dropout
        self.value_dropout = mem_config.mem_value_dropout

        self.values = nn.EmbeddingBag(self.size, self.v_dim, mode='sum', sparse=mem_config.mem_sparse)

        # query network
        assert len(query_layer_sizes) > 0

        # layer sizes / number of features
        l_sizes = list(query_layer_sizes)
        assert len(l_sizes) >= 2 and l_sizes[0] == l_sizes[-1] == 0
        l_sizes[0] = self.input_size
        l_sizes[-1] = self.heads * self.k_dim

        # query network
        self.query_proj = QueryMLP(l_sizes, num_heads=self.heads, output_size=self.k_dim, bias=mem_config.mem_query_bias, batchnorm=mem_config.mem_query_batchnorm)

        # initialize model parameters
        self._init()

    def _create_keys(self):
        """
        This function creates keys and returns them.
        I guess you could see that from the name of the function and the fact that is has a return statement.
        """
        assert self.keys_type in ['gaussian', 'uniform']
        half = self.k_dim // 2
        n_keys = int(self.size ** 0.5)

        # random keys from Gaussian or uniform distributions
        init = get_gaussian_keys if self.keys_type == 'gaussian' else get_uniform_keys
        keys = torch.from_numpy(np.array([
            init(n_keys, half, self.keys_normalized_init, seed=(2 * i + j))
            for i in range(self.heads)
            for j in range(2)
        ])).view(self.heads, 2, n_keys, half)
        return keys

    def _init(self):
        # values initialization
        nn.init.normal_(self.values.weight, mean=0, std=self.v_dim ** -0.5)
        # initialize keys
        keys = self._create_keys()
        # learned or fixed keys
        if self.learn_keys:
            self.keys = nn.Parameter(keys)
        else:
            self.register_buffer('keys', keys)

    def get_indices(self, query, knn):
        """
        Generate scores and indices given unnormalized queries.
        """
        assert query.dim() == 2 and query.size(1) == self.k_dim
        bs = len(query)
        query = query.view(-1, self.heads, self.k_dim)
        outputs = [
            self._get_indices(query[:, i], knn, self.keys[i][0], self.keys[i][1])
            for i in range(self.heads)
        ]
        scores = torch.cat([s.unsqueeze(1) for s, _ in outputs], 1).view(bs, knn)
        indices = torch.cat([idx.unsqueeze(1) for _, idx in outputs], 1).view(bs, knn)
        return scores, indices

    def _get_indices(self, query, knn, keys1, keys2):
        """
        Generate scores and indices given keys and unnormalized queries.
        """
        assert query.dim() == 2 and query.size(1) == self.k_dim
        assert len(keys1) == len(keys2)
        bs = query.size(0)
        half = self.k_dim // 2
        n_keys = len(keys1)

        # split query for product quantization
        q1 = query[:, :half]                                                                                          # (bs, half)
        q2 = query[:, half:]                                                                                          # (bs, half)

        # compute indices with associated scores
        scores1 = F.linear(q1, keys1, bias=None)                                                                      # (bs, n_keys ** 0.5)
        scores2 = F.linear(q2, keys2, bias=None)                                                                      # (bs, n_keys ** 0.5)
        scores1, indices1 = scores1.topk(knn, dim=1, largest=True, sorted=True)                                       # (bs, knn) ** 2
        scores2, indices2 = scores2.topk(knn, dim=1, largest=True, sorted=True)                                       # (bs, knn) ** 2

        # cartesian product on best candidate keys
        all_scores = (
            scores1.view(bs, knn, 1).expand(bs, knn, knn) +
            scores2.view(bs, 1, knn).expand(bs, knn, knn)
        ).view(bs, -1)                                                                                                # (bs, knn ** 2)
        all_indices = (
            indices1.view(bs, knn, 1).expand(bs, knn, knn) * n_keys +
            indices2.view(bs, 1, knn).expand(bs, knn, knn)
        ).view(bs, -1)                                                                                                # (bs, knn ** 2)

        # select overall best scores and indices
        scores, best_indices = torch.topk(all_scores, k=knn, dim=1, largest=True, sorted=True)                        # (bs, knn)
        indices = all_indices.gather(1, best_indices)                                                                 # (bs, knn)

        # return scores with indices
        assert scores.shape == indices.shape == (bs, knn)
        return scores, indices

    def forward(self, input):
        """
        Read from the memory.
        """
        # input dimensions
        assert input.shape[-1] == self.input_size
        prefix_shape = input.shape[:-1]

        # compute query / store it
        bs = np.prod(prefix_shape)
        input = F.dropout(input, p=self.input_dropout, training=self.training)    # input shape
        query = self.query_proj(input)                                            # (bs * heads, k_dim)
        query = F.dropout(query, p=self.query_dropout, training=self.training)    # (bs * heads, k_dim)
        assert query.shape == (bs * self.heads, self.k_dim)

        # get indices
        scores, indices = self.get_indices(query, self.knn)                       # (bs * heads, knn) ** 2

        # re-scoring
        scores = F.softmax(scores.float(), dim=-1).type_as(scores)                # (bs * heads, knn)

        # merge heads / knn (since we sum heads)
        indices = indices.view(bs, self.heads * self.knn)                         # (bs, heads * knn)
        scores = scores.view(bs, self.heads * self.knn)                           # (bs, heads * knn)

        # weighted sum of values
        output = self.values(
            indices,
            per_sample_weights=scores.to(self.values.weight.data)
        ).to(scores)                                                              # (bs, v_dim)
        output = F.dropout(output, p=self.value_dropout, training=self.training)  # (bs, v_dim)

        # reshape output
        if len(prefix_shape) >= 2:
            output = output.view(prefix_shape + (self.v_dim,))                    # (..., v_dim)

        # store indices / scores (eval mode only - for usage statistics)
        if not self.training:
            self.last_indices = indices.view(bs, self.heads, self.knn).detach().cpu()
            self.last_scores = scores.view(bs, self.heads, self.knn).detach().cpu().float()
        return output