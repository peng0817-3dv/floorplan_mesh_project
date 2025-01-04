import torch_geometric
import torch_scatter
import torch
import torch.nn as nn
from torch.nn import ModuleList

import pytorch_lightning as pl
import hydra
from dataset.floorplan_triangles import FPTriangleNodes, FPTriangleNodesDataloader
from lightning_utilities.core.rank_zero import rank_zero_only
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from model.decoder import resnet34_decoder
from model.encoder import get_conv, GraphEncoder
from trainer import create_trainer, step, create_conv_batch
from util.positional_encoding import get_embedder
from taylor_series_linear_attention import TaylorSeriesLinearAttn
from local_attention import LocalMHA
from x_transformers.x_transformers import RMSNorm, FeedForward, LayerIntermediates


class AddTransformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.train_dataset = FPTriangleNodes(config, 'train', config.scale_augment, config.shift_augment)
        self.val_dataset = FPTriangleNodes(config, 'val', config.scale_augment, config.shift_augment)

        # 网络的编码器为GraphEncoder，解码器为resnet34_decoder
        self.encoder = GraphEncoder(no_max_pool=config.g_no_max_pool, aggr=config.g_aggr, graph_conv=config.graph_conv, use_point_features=config.use_point_feats, output_dim=576)
        self.decoder = resnet34_decoder(512, config.num_cls, config.ce_output)
        self.transformer = TransformerNet(config,dim = 512)
        self.post_quant = torch.nn.Linear(config.embed_dim * 3, 512)
        self.register_buffer('smoothing_weight', torch.tensor([2, 10, 200, 10, 2], dtype=torch.float32).unsqueeze(0).unsqueeze(0))

        self.automatic_optimization = False
        # 初始化结束，可视化一下真实数据
        # self.visualize_groundtruth()

    def configure_optimizers(self):
        # parameters = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.pre_quant.parameters()) + list(self.post_quant.parameters()) + list(self.vq.parameters())
        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.post_quant.parameters()) + list(self.transformer.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=self.config.lr, amsgrad=True, weight_decay=self.config.weight_decay)
        max_steps = int(self.config.max_epoch * len(self.train_dataset) / self.config.batch_size)
        print('Max Steps | First cycle:', max_steps)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, first_cycle_steps=max_steps, cycle_mult=1.0,
            max_lr=self.config.lr, min_lr=self.config.min_lr,
            warmup_steps=self.config.warmup_steps, gamma=1.0
        )
        return [optimizer], [scheduler]

    def create_conv_batch(self, encoded_features, batch, batch_size):
        return create_conv_batch(encoded_features, batch, batch_size, self.device)

    def training_step(self, data, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        scheduler.step()  # type: ignore
        # 输入encoder(sage conv) 得到编码后的特征
        encoded_x = self.encoder(data.x, data.edge_index, data.batch)  # output shape: (N(triangle_num), 576)
        encoded_x = self.post_quant(encoded_x) # N x 576 -> N x 512
        # 关于data.batch的解释： https://blog.csdn.net/qq_45770988/article/details/129772968
        # 将geometric_torch式的batch转化为普通totch式的batch
        encoded_x_conv, conv_mask = self.create_conv_batch(encoded_x, data.batch, self.config.batch_size)
        # 输入到decoder(resnet34) 得到预测的token
        encoded_x_conv = self.transformer(encoded_x_conv,conv_mask)  # B x 512 x max_graph_size
        decoded_x_conv = self.decoder(encoded_x_conv) # B x 512 x max_graph_size => B X max_graph_size X 9 X num_tokens
        # 此处变为B X max_graph_size X num_cls

        decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-1])[conv_mask, :] # num_triangles x num_cls(32)
        loss = torch.nn.functional.cross_entropy(decoded_x.reshape(-1, decoded_x.shape[-1]), data.y.reshape(-1), reduction='mean')


        acc = self.get_accuracy(decoded_x, data.y) # 计算准确度
        self.log("train/cross_entropy_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,batch_size=self.config.batch_size)
        self.log("train/acc", acc.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,batch_size=self.config.batch_size)
        loss = loss / self.config.gradient_accumulation_steps  # scale the loss to account for gradient accumulation
        self.manual_backward(loss)
        # accumulate gradients of `n` batches
        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            step(optimizer, [self.encoder, self.decoder, self.post_quant,self.transformer])
            optimizer.zero_grad(set_to_none=True)  # type: ignore
        self.log("lr", optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)  # type: ignore

    def validation_step(self, data, batch_idx):
        encoded_x = self.encoder(data.x, data.edge_index, data.batch)
        encoded_x = self.post_quant(encoded_x)
        encoded_x_conv, conv_mask = self.create_conv_batch(encoded_x, data.batch, self.config.batch_size)
        encoded_x_conv = self.transformer(encoded_x_conv,conv_mask)
        decoded_x_conv = self.decoder(encoded_x_conv)

        decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-1])[conv_mask, :] # num_triangles x num_cls(32)
        loss = torch.nn.functional.cross_entropy(decoded_x.reshape(-1, decoded_x.shape[-1]), data.y.reshape(-1), reduction='mean')
        # y_coords = data.y / (self.config.num_tokens - 3) - 0.5
        # loss_c = torch.nn.functional.mse_loss(decoded_c, y_coords, reduction='mean')
        acc = self.get_accuracy(decoded_x, data.y)
        # acc_triangle = self.get_triangle_accuracy(decoded_x, data.y)
        if not torch.isnan(loss).any():
            self.log(f"val/cross_entropy_loss", loss.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True,batch_size=self.config.batch_size)
        if not torch.isnan(acc).any():
            self.log(f"val/acc", acc.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True,batch_size=self.config.batch_size)


    def inference_data(self, data):
        encoded_x = self.encoder(data.x.to(self.device), data.edge_index.to(self.device), torch.zeros([data.x.shape[0]],device=self.device).long())
        encoded_x = self.post_quant(encoded_x)
        encoded_x_conv, conv_mask = self.create_conv_batch(encoded_x, torch.zeros([data.x.shape[0]], device=self.device).long(), 1)
        encoded_x_conv = self.transformer(encoded_x_conv,conv_mask)
        decoded_x_conv = self.decoder(encoded_x_conv)
        decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-1])[conv_mask, :]
        predict_labels = decoded_x.argmax(-1).reshape(-1)
        return predict_labels.detach().cpu().numpy()

    @rank_zero_only
    def on_validation_epoch_end(self):
        pass

    def visualize_groundtruth(self):
        '''
        借由感兴趣的类，可视化训练集和验证集的ground truth
        output_dir_image_train: 训练集真实图像输出目录
        output_dir_image_val: 验证集真实图像输出目录
        从数据集中均匀取出num_val_samples个样本，并将其真实数据坐标转换为mesh，然后将其可视化保存为jpg文件
        '''
        # category_names = [""] + [x[0].strip("_") for x in self.interesting_categories]
        for didx, dataset in enumerate([self.train_dataset,self.val_dataset]): # didx <=> data index
            output_dir_image = self.output_dir_image_train if didx == 0 else self.output_dir_image_val
            for k in range(self.config.num_val_samples):
                # 均匀取数据
                data = dataset.get(k * (len(dataset) // self.config.num_val_samples) % len(dataset))
                if self.config.ce_output:
                    dataset.save_sample_data_by_idx(k * (len(dataset) // self.config.num_val_samples) % len(dataset),\
                                                is_quantized=True, save_path_root=output_dir_image)
                else:
                    dataset.save_sample_data_by_idx(k * (len(dataset) // self.config.num_val_samples) % len(dataset), \
                                                    is_quantized=False, save_path_root=output_dir_image)

    def train_dataloader(self):
        return FPTriangleNodesDataloader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=not self.config.overfit, num_workers=self.config.num_workers, pin_memory=False)

    def val_dataloader(self):
        return FPTriangleNodesDataloader(self.val_dataset,batch_size=self.config.batch_size, shuffle=True, drop_last=True, num_workers=self.config.num_workers)

    def get_accuracy(self, x, y):
        return (x.argmax(-1).reshape(-1) == y.reshape(-1)).sum() / y.shape[0]


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


@hydra.main(config_path='../config', config_name='add_transformer', version_base='1.2')
def main(config):
    # 创建一个lightning提供的Trainer实例
    trainer = create_trainer("TriangleTokens", config)
    model = AddTransformer(config)
    trainer.fit(model, ckpt_path=config.resume)



if __name__ == '__main__':
    main()