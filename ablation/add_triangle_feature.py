import torch_geometric
import torch_scatter
import torch
import torch.nn as nn
from torch.nn import ModuleList

import pytorch_lightning as pl
import hydra
from dataset.floorplan_triangles import FPTriangleWithGeneratedFeaturesNodes, FPTriangleNodesDataloader
from lightning_utilities.core.rank_zero import rank_zero_only
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from model.decoder import resnet34_decoder
from model.encoder import GraphEncoder, get_conv
from trainer import create_trainer, step, create_conv_batch

from taylor_series_linear_attention import TaylorSeriesLinearAttn
from local_attention import LocalMHA
from x_transformers.x_transformers import RMSNorm, FeedForward, LayerIntermediates

from util.positional_encoding import get_embedder


class AddTriangleFeature(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.train_dataset = FPTriangleWithGeneratedFeaturesNodes(config, 'train', config.scale_augment, config.shift_augment)
        self.val_dataset = FPTriangleWithGeneratedFeaturesNodes(config, 'val', config.scale_augment, config.shift_augment)

        # 网络的编码器为GraphEncoder，解码器为resnet34_decoder
        self.encoder = GraphEncoderAdaptExtraFeatures(no_max_pool=config.g_no_max_pool, aggr=config.g_aggr, graph_conv=config.graph_conv, use_point_features=config.use_point_feats, output_dim=576)
        self.decoder = resnet34_decoder(512, config.num_cls, config.ce_output)
        self.post_quant = torch.nn.Linear(config.embed_dim * 3, 512)
        self.register_buffer('smoothing_weight', torch.tensor([2, 10, 200, 10, 2], dtype=torch.float32).unsqueeze(0).unsqueeze(0))

        self.automatic_optimization = False
        # 初始化结束，可视化一下真实数据
        # self.visualize_groundtruth()

    def configure_optimizers(self):
        # parameters = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.pre_quant.parameters()) + list(self.post_quant.parameters()) + list(self.vq.parameters())
        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.post_quant.parameters())
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
            step(optimizer, [self.encoder, self.decoder, self.post_quant])
            optimizer.zero_grad(set_to_none=True)  # type: ignore
        self.log("lr", optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)  # type: ignore

    def validation_step(self, data, batch_idx):
        encoded_x = self.encoder(data.x, data.edge_index, data.batch)
        encoded_x = self.post_quant(encoded_x)
        encoded_x_conv, conv_mask = self.create_conv_batch(encoded_x, data.batch, self.config.batch_size)
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


class GraphEncoderAdaptExtraFeatures(nn.Module):
    def __init__(self, no_max_pool=True, aggr='mean', graph_conv="edge", use_point_features=False, output_dim=512):
        super().__init__()
        self.no_max_pool = no_max_pool
        self.use_point_features = use_point_features
        self.embedder, self.embed_dim = get_embedder(10)
        self.conv = graph_conv # 卷积类型
        self.gc1 = get_conv(self.conv, self.embed_dim * 3 + 7 + 4, 64, aggr=aggr)
        self.gc2 = get_conv(self.conv, 64, 128, aggr=aggr)
        self.gc3 = get_conv(self.conv, 128, 256, aggr=aggr)
        self.gc4 = get_conv(self.conv, 256, 256, aggr=aggr)
        self.gc5 = get_conv(self.conv, 256, output_dim, aggr=aggr)

        self.norm1 = torch_geometric.nn.BatchNorm(64)
        self.norm2 = torch_geometric.nn.BatchNorm(128)
        self.norm3 = torch_geometric.nn.BatchNorm(256)
        self.norm4 = torch_geometric.nn.BatchNorm(256)

        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        # 位置特征做positional embedding, 如果要改为2维，则嵌入函数需要修改
        x_0 = self.embedder(x[:, :3])
        x_1 = self.embedder(x[:, 3:6])
        x_2 = self.embedder(x[:, 6:9]) #  pe后特征63维 = 3维位置 + （10维sin + 10维cos） * 一个点的三维坐标
        x_confidence = x[:, 9:16]  # 法向量特征
        x_ar = x[:, 16:17]  # 面积特征
        x_an_0 = x[:, 17:18]  # 内角1
        x_an_1 = x[:, 18:19]  # 内角2
        x_an_2 = x[:, 19:]  # 内角3
        # 196维特征 = 63维位置（顶点1） + 63维位置（顶点2） + 63维位置（顶点3） + 3维法向量 + 1维面积 + 3维内角
        x = torch.cat([x_0, x_1, x_2, x_confidence, x_ar, x_an_0, x_an_1, x_an_2], dim=-1)
        #
        x = self.relu(self.norm1(self.gc1(x, edge_index)))
        x = self.norm2(self.gc2(x, edge_index))
        # 单独取了图卷积第二层未relu的特征视作点特征
        point_features = x
        x = self.relu(x)
        x = self.relu(self.norm3(self.gc3(x, edge_index)))
        x = self.relu(self.norm4(self.gc4(x, edge_index)))
        # 第5层没有使用norm和relu
        x = self.gc5(x, edge_index)
        if not self.no_max_pool:
            x = torch_scatter.scatter_max(x, batch, dim=0)[0]
            x = x[batch, :]
        if self.use_point_features: # 配置一般为False，不使用
            return torch.cat([x, point_features], dim=-1)
        return x

@hydra.main(config_path='../config', config_name='add_triangle_feature', version_base='1.2')
def main(config):
    # 创建一个lightning提供的Trainer实例
    trainer = create_trainer("TriangleTokens", config)
    model = AddTriangleFeature(config)
    trainer.fit(model, ckpt_path=config.resume)



if __name__ == '__main__':
    main()