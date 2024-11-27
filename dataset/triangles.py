from typing import Mapping, Sequence

import hydra
import omegaconf
import torch
import numpy as np
import trimesh
from torch.utils.data import Dataset, default_collate
from pathlib import Path
import torch.utils.data
import pickle

from torch_geometric.data.data import BaseData
from tqdm import tqdm

from dataset import sort_vertices_and_faces, quantize_coordinates
from util.misc import normalize_vertices, scale_vertices, shift_vertices
from torch_geometric.data import Dataset as GeometricDataset, Batch
from torch_geometric.data import Data as GeometricData
from torch_geometric.loader.dataloader import Collater as GeometricCollator


class TriangleNodes(GeometricDataset):

    def __init__(self, config, split, scale_augment, shift_augment, force_category, use_start_stop=False, only_backward_edges=False):
        super().__init__()
        data_path = Path(config.dataset_root)
        self.cached_vertices = []
        self.cached_faces = []
        self.names = []
        self.scale_augment = scale_augment
        self.shift_augment = shift_augment
        self.low_augment = config.low_augment
        self.use_start_stop = use_start_stop
        self.ce_output = config.ce_output
        self.only_backward_edges = only_backward_edges
        self.num_tokens = config.num_tokens - 3 # num token(131) - 3 = 128
        with open(data_path, 'rb') as fptr:
            data = pickle.load(fptr)
            if force_category is not None:
                for s in ['train', 'val']:
                    data[f'vertices_{s}'] = [data[f'vertices_{s}'][i] for i in range(len(data[f'vertices_{s}'])) if data[f'name_{s}'][i].split('_')[0] == force_category]
                    data[f'faces_{s}'] = [data[f'faces_{s}'][i] for i in range(len(data[f'faces_{s}'])) if data[f'name_{s}'][i].split('_')[0] == force_category]
                    data[f'name_{s}'] = [data[f'name_{s}'][i] for i in range(len(data[f'name_{s}'])) if data[f'name_{s}'][i].split('_')[0] == force_category]
                if len(data[f'vertices_val']) == 0:
                    data[f'vertices_val'] = data[f'vertices_train']
                    data[f'faces_val'] = data[f'faces_train']
                    data[f'name_val'] = data[f'name_train']

            if not config.overfit:
                self.names = data[f'name_{split}']
                self.cached_vertices = data[f'vertices_{split}']
                self.cached_faces = data[f'faces_{split}']
            else:
                multiplier = 16 if split == 'val' else 512
                self.names = data[f'name_train'][:1] * multiplier
                self.cached_vertices = data[f'vertices_train'][:1] * multiplier
                self.cached_faces = data[f'faces_train'][:1] * multiplier

        print(len(self.cached_vertices), "meshes loaded")

    def len(self):
        return len(self.cached_vertices)

    def get_all_features_for_shape(self, idx):
        vertices = self.cached_vertices[idx]
        faces = self.cached_faces[idx]
        if self.scale_augment:
            if self.low_augment:
                x_lims = (0.9, 1.1)
                y_lims = (0.9, 1.1)
                z_lims = (0.9, 1.1)
            else:
                x_lims = (0.75, 1.25)
                y_lims = (0.75, 1.25)
                z_lims = (0.75, 1.25)
            vertices = scale_vertices(vertices, x_lims=x_lims, y_lims=y_lims, z_lims=z_lims)
        vertices = normalize_vertices(vertices)
        if self.shift_augment:
            vertices = shift_vertices(vertices)
        triangles, normals, areas, angles, vertices, faces = create_feature_stack(vertices, faces, self.num_tokens)
        features = np.hstack([triangles, normals, areas, angles])
        face_neighborhood = np.array(trimesh.Trimesh(vertices=vertices, faces=faces, process=False).face_neighborhood)  # type: ignore
        # 取前9个特征，实质就是坐标特征
        target = torch.from_numpy(features[:, :9]).float()
        if self.use_start_stop:
            features = np.concatenate([np.zeros((1, features.shape[1])), features], axis=0)
            target = torch.cat([target, torch.ones(1, 9) * 0.5], dim=0)
            face_neighborhood = face_neighborhood + 1
        if self.only_backward_edges:
            face_neighborhood = face_neighborhood[face_neighborhood[:, 1] > face_neighborhood[:, 0], :]
            # face_neighborhood = modify so that only edges in backward direction are present
        # 是否对标签做离散化
        if self.ce_output:
            target = quantize_coordinates(target, self.num_tokens)
        return features, target, vertices, faces, face_neighborhood

    def get(self, idx):
        features, target, _, _, face_neighborhood = self.get_all_features_for_shape(idx)
        return GeometricData(x=torch.from_numpy(features).float(), y=target, edge_index=torch.from_numpy(face_neighborhood.T).long())


class TriangleNodesWithFaces(TriangleNodes):

    def __init__(self, config, split, scale_augment, shift_augment, force_category):
        super().__init__(config, split, scale_augment, shift_augment, force_category)

    def get(self, idx):
        features, target, vertices, faces, face_neighborhood = self.get_all_features_for_shape(idx)
        return GeometricData(x=torch.from_numpy(features).float(), y=target,
                             edge_index=torch.from_numpy(face_neighborhood.T).long(),
                             num_vertices=vertices.shape[0], faces=torch.from_numpy(np.array(faces)).long())


class TriangleNodesWithFacesDataloader(torch.utils.data.DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=None, exclude_keys=None, **kwargs):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)
        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=FaceCollator(follow_batch, exclude_keys),
            **kwargs,
        )


class FaceCollator(GeometricCollator):

    def __init__(self, follow_batch, exclude_keys):
        super().__init__(follow_batch, exclude_keys)

    def __call__(self, batch):
        elem = batch[0]

        num_vertices = 0
        for b_idx in range(len(batch)):
            batch[b_idx].faces += num_vertices
            num_vertices += batch[b_idx].num_vertices

        if isinstance(elem, BaseData):
            return Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')

    def collate(self, batch):  # pragma: no cover
        raise NotImplementedError

class TriangleNodesWithSequenceIndices(TriangleNodes):

    vq_depth_factor = 1

    def __init__(self, config, split, scale_augment, shift_augment, force_category):
        super().__init__(config, split, scale_augment=scale_augment, shift_augment=shift_augment, force_category=force_category)
        vq_cfg = omegaconf.OmegaConf.load(Path(config.vq_resume).parents[1] / "config.yaml")
        self.vq_depth = vq_cfg.embed_levels
        self.block_size = config.block_size
        max_inner_face_len = 0
        self.padding = int(config.padding * self.block_size)
        self.sequence_stride = config.sequence_stride
        for i in range(len(self.cached_vertices)):
            self.cached_vertices[i] = np.array(self.cached_vertices[i])
            for j in range(len(self.cached_faces[i])):
                max_inner_face_len = max(max_inner_face_len, len(self.cached_faces[i][j]))
        print('Longest inner face sequence', max_inner_face_len)
        assert max_inner_face_len == 3, f"Only triangles are supported, but found a face with {max_inner_face_len}."
        self.sequence_indices = []
        max_face_sequence_len = 0
        min_face_sequence_len = 1e7
        for i in range(len(self.cached_faces)):
            sequence_len = len(self.cached_faces[i]) * self.vq_depth * self.vq_depth_factor + 1 + 1
            max_face_sequence_len = max(max_face_sequence_len, sequence_len)
            min_face_sequence_len = min(min_face_sequence_len, sequence_len)
            self.sequence_indices.append((i, 0, False))
            for j in range(config.sequence_stride, max(1, sequence_len - self.block_size + self.padding + 1), config.sequence_stride):  # todo: possible bug? +1 added recently
                self.sequence_indices.append((i, j, True if split == 'train' else False))
            if sequence_len > self.block_size: 
                self.sequence_indices.append((i, sequence_len - self.block_size, False))
        print('Length of', split, len(self.sequence_indices))
        print('Shortest face sequence', min_face_sequence_len)
        print('Longest face sequence', max_face_sequence_len)

    def len(self):
        return len(self.sequence_indices)
    
    def get(self, idx):
        i, j, randomness = self.sequence_indices[idx]
        if randomness:
            sequence_len = len(self.cached_faces[i]) * self.vq_depth * self.vq_depth_factor + 1 + 1
            j = min(max(0, j + np.random.randint(-self.sequence_stride // 2, self.sequence_stride // 2)), sequence_len - self.block_size + self.padding)
        features, target, _, _, face_neighborhood = self.get_all_features_for_shape(i)
        return GeometricData(x=torch.from_numpy(features).float(), y=target, edge_index=torch.from_numpy(face_neighborhood.T).long(), js=torch.tensor(j).long())
    
    def plot_sequence_lenght_stats(self):
        sequence_lengths = []
        for i in range(len(self.cached_faces)):
            sequence_len = len(self.cached_faces[i]) * self.vq_depth * self.vq_depth_factor + 1 + 1
            sequence_lengths.append(sequence_len)
        import matplotlib.pyplot as plt
        plt.hist(sequence_lengths, bins=32)
        plt.ylim(0, 100)
        plt.show()
        return sequence_lengths


class TriangleNodesWithFacesAndSequenceIndices(TriangleNodesWithSequenceIndices):
    vq_depth_factor = 3
    def __init__(self, config, split, scale_augment, shift_augment, force_category):
        super().__init__(config, split, scale_augment, shift_augment, force_category)

    def get(self, idx):
        i, j, randomness = self.sequence_indices[idx]
        if randomness:
            sequence_len = len(self.cached_faces[i]) * self.vq_depth * self.vq_depth_factor + 1 + 1
            j = min(max(0, j + np.random.randint(-self.sequence_stride // 2, self.sequence_stride // 2)), sequence_len - self.block_size + self.padding)
        features, target, vertices, faces, face_neighborhood = self.get_all_features_for_shape(i)
        return GeometricData(x=torch.from_numpy(features).float(),
                             y=target, mesh_name=self.names[i], edge_index=torch.from_numpy(face_neighborhood.T).long(),
                             js=torch.tensor(j).long(), num_vertices=vertices.shape[0],
                             faces=torch.from_numpy(np.array(faces)).long())


class Triangles(Dataset):

    def __init__(self, config, split, scale_augment, shift_augment):
        super().__init__()
        data_path = Path(config.dataset_root)
        self.cached_vertices = []
        self.cached_faces = []
        self.names = []
        self.scale_augment = scale_augment
        self.shift_augment = shift_augment
        with open(data_path, 'rb') as fptr:
            data = pickle.load(fptr)
            if not config.overfit:
                self.names = data[f'name_{split}']
                self.cached_vertices = data[f'vertices_{split}']
                self.cached_faces = data[f'faces_{split}']
            else:
                multiplier = 1 if split == 'val' else 500
                self.names = data[f'name_train'][:1] * multiplier
                self.cached_vertices = data[f'vertices_train'][:1] * multiplier
                self.cached_faces = data[f'faces_train'][:1] * multiplier

        print(len(self.cached_vertices), "meshes loaded")
        self.features = None
        self.setup_triangles_for_epoch()

    def __len__(self):
        return self.features.shape[0]

    def setup_triangles_for_epoch(self):
        all_features = []
        for idx in tqdm(range(len(self.cached_vertices)), desc="refresh augs"):
            vertices = self.cached_vertices[idx]
            faces = self.cached_faces[idx]
            if self.scale_augment:
                vertices = scale_vertices(vertices)
            vertices = normalize_vertices(vertices)
            if self.shift_augment:
                vertices = shift_vertices(vertices)
            all_features.append(create_feature_stack(vertices, faces)[0])
        self.features = np.vstack(all_features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'target': self.features[idx, :9]
        }

    def get_all_features_for_shape(self, idx):
        vertices = self.cached_vertices[idx]
        faces = self.cached_faces[idx]
        feature_stack = create_feature_stack(vertices, faces)[0]
        return torch.from_numpy(feature_stack).float(), torch.from_numpy(feature_stack[:, :9]).float()


def normal(triangles):
    # The cross product of two sides is a normal vector
    if torch.is_tensor(triangles):
        return torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=1)
    else:
        return np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], axis=1)


def area(triangles):
    # The norm of the cross product of two sides is twice the area
    if torch.is_tensor(triangles):
        return torch.norm(normal(triangles), dim=1) / 2
    else:
        return np.linalg.norm(normal(triangles), axis=1) / 2


def angle(triangles):
    v_01 = triangles[:, 1] - triangles[:, 0]
    v_02 = triangles[:, 2] - triangles[:, 0]
    v_10 = -v_01
    v_12 = triangles[:, 2] - triangles[:, 1]
    v_20 = -v_02
    v_21 = -v_12
    if torch.is_tensor(triangles):
        return torch.stack([angle_between(v_01, v_02), angle_between(v_10, v_12), angle_between(v_20, v_21)], dim=1)
    else:
        return np.stack([angle_between(v_01, v_02), angle_between(v_10, v_12), angle_between(v_20, v_21)], axis=1)


def angle_between(v0, v1):
    v0_u = unit_vector(v0)
    v1_u = unit_vector(v1)
    if torch.is_tensor(v0):
        return torch.arccos(torch.clip(torch.einsum('ij,ij->i', v0_u, v1_u), -1.0, 1.0))
    else:
        return np.arccos(np.clip(np.einsum('ij,ij->i', v0_u, v1_u), -1.0, 1.0))


def unit_vector(vector):
    if torch.is_tensor(vector):
        return vector / (torch.norm(vector, dim=-1)[:, None] + 1e-8)
    else:
        return vector / (np.linalg.norm(vector, axis=-1)[:, None] + 1e-8)


def create_feature_stack(vertices, faces, num_tokens):
    # 此处的对数据按照顶点和面进行排序（y, x, z），并同步重新排列了faces的顺序
    # 并且整个排序的基础是建立在使用num_tokens定义的量级，将坐标进行离散化后排序的结果，其返回的坐标也是在归一空间的离散化整合分布
    vertices, faces = sort_vertices_and_faces(vertices, faces, num_tokens)
    # need more features: positions, angles, area, cross_product
    # triangles [face_num, 3, 3]
    triangles = vertices[faces, :]
    # 得到法向量、mesh面积、角度后顺便把triangles的形状变成[face_num, 3 * 3]
    triangles, normals, areas, angles = create_feature_stack_from_triangles(triangles)
    return triangles, normals, areas, angles, vertices, faces


def create_feature_stack_from_triangles(triangles):
    t_areas = area(triangles) * 1e3
    t_angles = angle(triangles) / float(np.pi)
    t_normals = unit_vector(normal(triangles))
    return triangles.reshape(-1, 9), t_normals.reshape(-1, 3), t_areas.reshape(-1, 1), t_angles.reshape(-1, 3)

@hydra.main(config_path='../config', config_name='meshgpt', version_base='1.2')
def main(config):
    train_dataset = TriangleNodesWithFaces(config, 'train', config.scale_augment, config.shift_augment, None)
    data_0 = train_dataset.get(0)
    print(data_0)

if __name__ == '__main__':
    main()