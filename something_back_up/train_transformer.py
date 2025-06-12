import random

import omegaconf
import trimesh
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import pytorch_lightning as pl
import hydra
from easydict import EasyDict
from lightning_utilities.core.rank_zero import rank_zero_only
from pathlib import Path
import torch

from something_back_up.quantized_soup import QuantizedSoupTripletsCreator
from something_back_up.triangles import TriangleNodesWithFacesAndSequenceIndices, TriangleNodesWithFacesDataloader
from model.transformer import QuantSoupTransformer
from trainer import create_trainer, step, get_rvqvae_v0_decoder
from util.misc import accuracy
from util.visualization import plot_vertices_and_faces
from util.misc import get_parameters_from_state_dict


class QuantSoupModelTrainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vq_cfg = omegaconf.OmegaConf.load(Path(config.vq_resume).parents[1] / "config.yaml")
        self.save_hyperparameters()
        self.train_dataset = TriangleNodesWithFacesAndSequenceIndices(config, 'train', config.scale_augment, config.shift_augment, config.ft_category)
        self.val_dataset = TriangleNodesWithFacesAndSequenceIndices(config, 'val', config.scale_augment_val, False, config.ft_category)
        print("Dataset Lengths:", len(self.train_dataset), len(self.val_dataset))
        print("Batch Size:", self.config.batch_size)
        print("Dataloader Lengths:", len(self.train_dataset) // self.config.batch_size, len(self.val_dataset) // self.config.batch_size)
        model_cfg = get_qsoup_model_config(config, self.vq_cfg.embed_levels)
        self.model = QuantSoupTransformer(model_cfg, self.vq_cfg)
        self.sequencer = QuantizedSoupTripletsCreator(self.config, self.vq_cfg)
        self.sequencer.freeze_vq()
        # print('compiling model...')
        # self.model = torch.compile(model)  # requires PyTorch 2.0
        self.output_dir_image = Path(f'runs/{self.config.experiment}/image')
        self.output_dir_image.mkdir(exist_ok=True, parents=True)
        self.output_dir_mesh = Path(f'runs/{self.config.experiment}/mesh')
        self.output_dir_mesh.mkdir(exist_ok=True, parents=True)
        if self.config.ft_resume is not None:
            self.model.load_state_dict(get_parameters_from_state_dict(torch.load(self.config.ft_resume, map_location='cpu')['state_dict'], "model"))
        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(
            self.config.weight_decay, self.config.lr,
            (self.config.beta1, self.config.beta2), 'cuda'
        )
        max_steps = int(self.config.max_epoch * len(self.train_dataset) / self.config.batch_size / 2)
        print('Max Steps | First cycle:', max_steps)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, first_cycle_steps=max_steps, cycle_mult=1.0,
            max_lr=self.config.lr, min_lr=self.config.min_lr,
            warmup_steps=self.config.warmup_steps, gamma=1.0
        )
        return [optimizer], [scheduler]

    def training_step(self, data, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        scheduler.step()  # type: ignore
        if self.config.force_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.config.force_lr
        sequence_in, sequence_out, pfin, pfout = self.sequencer(data.x, data.edge_index, data.batch, data.faces, data.num_vertices.sum(), data.js)
        logits, loss = self.model(sequence_in, pfin, pfout, self.sequencer, targets=sequence_out)
        acc = accuracy(logits.detach(), sequence_out, ignore_label=2, device=self.device)
        self.log("train/ce_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/acc", acc.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        loss = loss / self.config.gradient_accumulation_steps  # scale the loss to account for gradient accumulation
        self.manual_backward(loss)
        # accumulate gradients of `n` batches
        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            step(optimizer, [self.model])
            optimizer.zero_grad(set_to_none=True)  # type: ignore
        self.log("lr", optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)  # type: ignore

    def validation_step(self, data, batch_idx):
        sequence_in, sequence_out, pfin, pfout = self.sequencer(data.x, data.edge_index, data.batch, data.faces, data.num_vertices.sum(), data.js)
        logits, loss = self.model(sequence_in, pfin, pfout, self.sequencer, targets=sequence_out)
        acc = accuracy(logits.detach(), sequence_out, ignore_label=2, device=self.device)
        if not torch.isnan(loss).any():
            self.log("val/ce_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        if not torch.isnan(acc).any():
            self.log("val/acc", acc.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    @rank_zero_only
    def on_validation_epoch_end(self):
        decoder = get_rvqvae_v0_decoder(self.vq_cfg, self.config.vq_resume, self.device)
        for k in range(self.config.num_val_samples):
            data = self.val_dataset.get(random.randint(0, len(self.val_dataset) - 1))
            soup_sequence, face_in_idx, face_out_idx, target = self.sequencer.get_completion_sequence(
                data.x.to(self.device),
                data.edge_index.to(self.device),
                data.faces.to(self.device),
                data.num_vertices,
                12
            )
            y = self.model.generate_with_beamsearch(
                soup_sequence, face_in_idx, face_out_idx, self.sequencer, self.config.max_val_tokens, use_kv_cache=True
            )
            if y is None:
                continue
            gen_vertices, gen_faces = self.sequencer.decode(y[0], decoder)
            plot_vertices_and_faces(gen_vertices, gen_faces, self.output_dir_image / f"{self.global_step:06d}_{k}.jpg")

            try:
                trimesh.Trimesh(vertices=gen_vertices, faces=gen_faces, process=False).export(self.output_dir_mesh / f"{self.global_step:06d}_{k}.obj")
            except Exception as e:
                pass  # sometimes the mesh is invalid (ngon) and we don't want to crash

    def train_dataloader(self):
        return TriangleNodesWithFacesDataloader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=not self.config.overfit, num_workers=self.config.num_workers, pin_memory=True)

    def val_dataloader(self):
        return TriangleNodesWithFacesDataloader(self.val_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=False, num_workers=self.config.num_workers)


def get_qsoup_model_config(config, vq_embed_levels):
    cfg = EasyDict({
        'block_size': config.block_size,
        'n_embd': config.model.n_embd,
        'dropout': config.model.dropout,
        'n_layer': config.model.n_layer,
        'n_head': config.model.n_head,
        'bias': config.model.bias,
        'finemb_size': vq_embed_levels * 3,
        'foutemb_size': config.block_size * 3,
    })
    return cfg


@hydra.main(config_path='../config', config_name='meshgpt', version_base='1.2')
def main(config):
    trainer = create_trainer("MeshTriSoup", config)
    model = QuantSoupModelTrainer(config)
    trainer.fit(model, ckpt_path=config.resume)


if __name__ == '__main__':
    main()
