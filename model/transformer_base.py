import torch
from torch import nn
from torch.nn import ModuleList
from taylor_series_linear_attention import TaylorSeriesLinearAttn
from local_attention import LocalMHA
from x_transformers.x_transformers import RMSNorm, FeedForward, LayerIntermediates


from model.nanogpt import configure_optimizers


class TransformerBase(nn.Module):
    
    def __init__(self):
        super().__init__()

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for n,p in self.named_parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        return configure_optimizers(self.named_parameters(), weight_decay, learning_rate, betas, device_type)


class TransformerNet(nn.Module):
    def __init__(self, config,dim = 512):
        super().__init__()
        self.config = config

        use_linear_attn = config.use_linear_attn
        attn_kwargs = dict(
            causal = False,
            prenorm = True,
            dropout = config.attn_dropout,
            window_size = config.local_attn_window_size,
        )
        local_attn_kwargs = dict(
            heads = config.local_attn_heads,
            dim_head = config.local_attn_dim_head,
        )
        linear_attn_kwargs = dict(
            heads = config.linear_attn_heads,
            dim_head = config.linear_attn_dim_head,
        )

        curr_dim = dim
        self.encoder_attn_blocks = ModuleList([])
        for _ in range(config.attn_encoder_depth):
            self.encoder_attn_blocks.append(nn.ModuleList([
                TaylorSeriesLinearAttn(curr_dim, prenorm = True, **linear_attn_kwargs) if use_linear_attn else None,
                LocalMHA(dim = curr_dim, **attn_kwargs, **local_attn_kwargs),
                nn.Sequential(RMSNorm(curr_dim), FeedForward(curr_dim, glu = True, dropout = config.ff_dropout))
            ]))

    def forward(self, x, mask):
        x = x.permute(0, 2, 1)
        mask = mask.reshape(x.shape[0],-1)
        for linear_attn, local_attn, ff in self.encoder_attn_blocks:
            if linear_attn is not None:
                x = linear_attn(x, mask) + x
            x = local_attn(x, mask) + x
            x = ff(x) + x
        x = x.permute(0, 2, 1)
        return x
