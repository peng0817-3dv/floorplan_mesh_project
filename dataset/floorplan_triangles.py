import os
import pickle
from pathlib import Path

import numpy as np
import shapefile
import torch
import trimesh
from torch_geometric.data import Dataset as GeometricDataset
from tqdm import tqdm

from dataset import sort_vertices_and_faces,sort_vertices_and_faces_and_labels_and_features
from dataset.triangles import FaceCollator
from util.analyse_dataset import random_split, analyse_dataset_split
from util.misc import scale_vertices, normalize_vertices, shift_vertices, rotate_vertices, mirror_vertices
from util.visualization import plot_vertices_and_faces_with_labels,export_mesh_to_obj
from util.s3d_data_load import read_s3d_mesh_info
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from torch_geometric.data import Data as GeometricData
from torch_geometric.loader.dataloader import Collater as GeometricCollator


class FPOriginTriangleNodes(GeometricDataset):
    def __init__(self, config,split):
        super().__init__()
        data_path = Path(config.dataset_root)
        self.cached_vertices = []
        self.cached_faces = []
        self.extra_features = []

        data_cache_path = os.path.join(config.dataset_root, f"cache.pkl")
        if os.path.exists(data_cache_path):
            with open(data_cache_path, "rb") as f:
                data = pickle.load(f)
            print(f"load data from cache,cache path:{data_cache_path}")
        else:
            data = {"features": [], "vertices": [], "faces": [], "labels": []}
            for scene_name in tqdm(os.listdir(data_path)):
                scene_full_path = os.path.join(data_path, scene_name)
                if not os.path.isdir(scene_full_path):
                    continue
                vertices, faces, face_feature, labels = read_s3d_mesh_info(scene_full_path)
                data["features"].append(face_feature)
                data["vertices"].append(vertices)
                data["faces"].append(faces)
                data["labels"].append(labels)
            with open(data_cache_path, "wb") as f:
                pickle.dump(data,f)

        train_size = int(len(data["faces"]) * config.train_ratio)
        val_size = int(len(data["faces"]) * config.val_ratio)

        if split == "train":
            self.extra_features = data[f'features'][:train_size]
            self.cached_vertices = data[f'vertices'][:train_size]
            self.cached_faces = data[f'faces'][:train_size]
            self.labels = data[f'labels'][:train_size]
        elif split == "val":
            self.extra_features = data[f'features'][train_size:train_size+val_size]
            self.cached_vertices = data[f'vertices'][train_size:train_size+val_size]
            self.cached_faces = data[f'faces'][train_size:train_size+val_size]
            self.labels = data[f'labels'][train_size:train_size+val_size]
        elif split == "test":
            self.extra_features = data[f'features'][train_size+val_size:]
            self.cached_vertices = data[f'vertices'][train_size+val_size:]
            self.cached_faces = data[f'faces'][train_size+val_size:]
            self.labels = data[f'labels'][train_size+val_size:]


    def get_feature(self, idx):
        vertices = self.cached_vertices[idx]
        faces = self.cached_faces[idx]
        extra_features = self.extra_features[idx]
        target = self.labels[idx]
        face_neighborhood = np.array(trimesh.Trimesh(vertices=vertices, faces=faces, process=False).face_neighborhood)
        triangles = vertices[faces, :].reshape(-1,9)
        features = np.hstack([triangles, extra_features])
        return features, target, face_neighborhood

    def get(self, idx):
        features, target, face_neighborhood = self.get_feature(idx)
        return GeometricData(x=torch.from_numpy(features).float(), y=target,
                             edge_index=torch.from_numpy(face_neighborhood.T).long(),
                             )

    def len(self):
        return len(self.cached_vertices)


class FPTriangleNodes(GeometricDataset):
    def __init__(self, config, split,scale_augment=False,shift_augment=False):
        super().__init__()
        data_path = Path(config.dataset_root)
        self.cached_vertices = []
        self.cached_faces = []
        self.extra_features = []
        self.labels = []
        self.names = []
        # self.only_backward_edges = only_backward_edges
        # self.num_tokens = config.num_tokens
        self.discrete_size = config.discrete_size
        self.scale_augment = scale_augment
        self.shift_augment = shift_augment
        self.low_augment = config.low_augment

        data_cache_path = os.path.join(config.dataset_root, f"cache.pkl")
        split_analysis_path = os.path.join(config.dataset_root, f"split_analysis.png")
        if os.path.exists(data_cache_path):
            with open(data_cache_path, "rb") as f:
                data = pickle.load(f)
            print(f"load data from cache,cache path:{data_cache_path}")
        else:
            data = {"features": [], "vertices": [], "faces": [], "labels": [],"names": [],"split_idx":[]}
            for scene_name in tqdm(os.listdir(data_path)):
                if not os.path.isdir(os.path.join(data_path, scene_name)):
                    continue
                scene_full_path = os.path.join(data_path, scene_name)
                vertices, faces, face_feature, labels = read_s3d_mesh_info(scene_full_path)
                data["features"].append(face_feature)
                data["vertices"].append(vertices)
                data["faces"].append(faces)
                data["labels"].append(labels)
                data["names"].append(scene_name)

            data["split_idx"] = random_split(len(data["labels"]), config.train_ratio, config.val_ratio)
            analyse_dataset_split(data["split_idx"], data["labels"], split_analysis_path)
            with open(data_cache_path, "wb") as f:
                pickle.dump(data,f)

        if split == "train":
            self.extra_features = [data[f'features'][i] for i in data["split_idx"][0]]
            self.cached_vertices = [data[f'vertices'][i] for i in data["split_idx"][0]]
            self.cached_faces = [data[f'faces'][i] for i in data["split_idx"][0]]
            self.labels = [data[f'labels'][i] for i in data["split_idx"][0]]
            self.names = [data[f'names'][i] for i in data["split_idx"][0]]
        elif split == "val":
            self.extra_features = [data[f'features'][i] for i in data["split_idx"][1]]
            self.cached_vertices = [data[f'vertices'][i] for i in data["split_idx"][1]]
            self.cached_faces = [data[f'faces'][i] for i in data["split_idx"][1]]
            self.labels = [data[f'labels'][i] for i in data["split_idx"][1]]
            self.names = [data[f'names'][i] for i in data["split_idx"][1]]
        elif split == "test":
            self.extra_features = [data[f'features'][i] for i in data["split_idx"][2]]
            self.cached_vertices = [data[f'vertices'][i] for i in data["split_idx"][2]]
            self.cached_faces = [data[f'faces'][i] for i in data["split_idx"][2]]
            self.labels = [data[f'labels'][i] for i in data["split_idx"][2]]
            self.names = [data[f'names'][i] for i in data["split_idx"][2]]

        if split == "train":
            self.data_augmentation()
        print(len(self.cached_vertices), "meshes loaded loading for", split)


    def len(self):
        return len(self.cached_vertices)

    def get_name(self, idx):
        return self.names[idx]


    def get_all_features_for_shape(self, idx):
        vertices = self.cached_vertices[idx]
        faces = self.cached_faces[idx]
        confidence = self.extra_features[idx]
        labels = self.labels[idx]
        reverse_op = []
        if self.scale_augment:
            if self.low_augment:
                x_lims = (0.9, 1.1)
                y_lims = (0.9, 1.1)
                z_lims = (0.9, 1.1)
            else:
                x_lims = (0.75, 1.25)
                y_lims = (0.75, 1.25)
                z_lims = (0.75, 1.25)
            vertices,scale_rev = scale_vertices(vertices, x_lims=x_lims, y_lims=y_lims, z_lims=z_lims)
            reverse_op.append(['*',scale_rev])
        vertices,rev_1,rev_2 = normalize_vertices(vertices)
        reverse_op.append(['+',rev_1])
        reverse_op.append(['*',rev_2])
        if self.shift_augment:
            vertices,shift_rev = shift_vertices(vertices)
            reverse_op.append(['+',shift_rev])

        reverse_op = reverse_op[::-1]
        # 注意该排序会同时做离散化操作
        vertices, faces,labels,confidence = \
            sort_vertices_and_faces_and_labels_and_features(vertices, faces, labels, confidence, self.discrete_size)
        triangles = vertices[faces, :].reshape(-1,9)
        # triangles, normals, areas, angles, vertices, faces = create_feature_stack(vertices, faces, self.num_tokens)
        features = np.hstack([triangles, confidence])
        face_neighborhood = np.array(trimesh.Trimesh(vertices=vertices, faces=faces, process=False).face_neighborhood)  # type: ignore
        target = torch.from_numpy(labels).long() - 1
        return features, target, vertices, faces, face_neighborhood,reverse_op

    def get(self, idx):
        features, target, vertices, faces, face_neighborhood,reverse_op = self.get_all_features_for_shape(idx)
        return GeometricData(x=torch.from_numpy(features).float(), y=target,
                             edge_index=torch.from_numpy(face_neighborhood.T).long(),
                             num_vertices=vertices.shape[0], faces=torch.from_numpy(np.array(faces)).long(),
                             reverse_op=reverse_op)

    def save_sample_data_by_idx(self,idx,is_quantized=False,save_path_root = os.getcwd()):
        vertices = self.cached_vertices[idx]
        faces = self.cached_faces[idx]
        labels = self.labels[idx]
        features = self.extra_features[idx]
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

        if is_quantized:
            vertices, faces, labels, features = \
                sort_vertices_and_faces_and_labels_and_features(vertices, faces, labels, features, self.num_tokens)

        suffer = "quantized" if is_quantized else "non_quantized"
        export_mesh_to_obj(vertices, faces, f"{save_path_root}/GT_scene_{idx}_{suffer}.obj")
        plot_vertices_and_faces_with_labels(vertices, faces, labels, f"{save_path_root}/GT_scene_{idx}_{suffer}.png")


    def data_augmentation(self):
        augmented_vertices = self.cached_vertices.copy()
        augmented_faces = self.cached_faces.copy()
        augmented_labels = self.labels.copy()
        augmented_features = self.extra_features.copy()
        augemnted_names = self.names.copy()

        transformations = [
            lambda v: rotate_vertices(v, 90),
            lambda v: rotate_vertices(v, 180),
            lambda v: rotate_vertices(v, 270),
            lambda v: mirror_vertices(v, 'x'),
            lambda v: mirror_vertices(v, 'y')
        ]

        for transform in transformations:
            augmented_vertices.extend([
                transform(v) for v in self.cached_vertices
            ])
            augmented_faces.extend(self.cached_faces)
            augmented_labels.extend(self.labels)
            augmented_features.extend(self.extra_features)
            augemnted_names.extend(self.names)

        print(f" augmented data size: {len(augmented_vertices)},origin data size: {len(self.cached_vertices)}")
        self.cached_vertices = augmented_vertices
        self.cached_faces = augmented_faces
        self.labels = augmented_labels
        self.extra_features = augmented_features
        self.names = augemnted_names


class FPTriangleNodesDataloader(torch.utils.data.DataLoader):
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


class FPTriangleWithGeneratedFeaturesNodes(FPTriangleNodes):
    def __init__(self, config, split,scale_augment=False,shift_augment=False):
        GeometricDataset.__init__(self)
        data_path = Path(config.dataset_root)
        self.cached_vertices = []
        self.cached_faces = []
        self.extra_features = []
        self.labels = []
        self.names = []
        # self.only_backward_edges = only_backward_edges
        self.num_tokens = config.num_tokens
        self.scale_augment = scale_augment
        self.shift_augment = shift_augment
        self.low_augment = config.low_augment


        data_cache_path = os.path.join(config.dataset_root, f"extra_features_cache.pkl")
        if os.path.exists(data_cache_path):
            with open(data_cache_path, "rb") as f:
                data = pickle.load(f)
            print(f"load data from cache,cache path:{data_cache_path}")
        else:
            data = {"features": [], "vertices": [], "faces": [], "labels": [],"names": []}
            for scene_name in tqdm(os.listdir(data_path)):
                scene_full_path = os.path.join(data_path, scene_name)
                if not os.path.isdir(scene_full_path):
                    continue
                vertices, faces, face_feature, labels = read_s3d_mesh_info(scene_full_path)
                data["features"].append(face_feature)
                data["vertices"].append(vertices)
                data["faces"].append(faces)
                data["labels"].append(labels)
                data["names"].append(scene_name)
            with open(data_cache_path, "wb") as f:
                pickle.dump(data,f)

        train_size = int(len(data["faces"]) * config.train_ratio)
        val_size = int(len(data["faces"]) * config.val_ratio)
        test_size = len(data["faces"]) - train_size - val_size
        if split == "train":
            self.extra_features = data[f'features'][:train_size]
            self.cached_vertices = data[f'vertices'][:train_size]
            self.cached_faces = data[f'faces'][:train_size]
            self.labels = data[f'labels'][:train_size]
            self.names = data[f'names'][:train_size]
        elif split == "val":
            self.extra_features = data[f'features'][train_size:train_size+val_size]
            self.cached_vertices = data[f'vertices'][train_size:train_size+val_size]
            self.cached_faces = data[f'faces'][train_size:train_size+val_size]
            self.labels = data[f'labels'][train_size:train_size+val_size]
            self.names = data[f'names'][train_size:train_size+val_size]
        elif split == "test":
            self.extra_features = data[f'features'][train_size+val_size:]
            self.cached_vertices = data[f'vertices'][train_size+val_size:]
            self.cached_faces = data[f'faces'][train_size+val_size:]
            self.labels = data[f'labels'][train_size+val_size:]
            self.names = data[f'names'][train_size+val_size:]

        if split == "train":
            self.data_augmentation()
        print(len(self.cached_vertices), "meshes loaded loading for", split)



    def get_all_features_for_shape(self, idx):
        vertices = self.cached_vertices[idx]
        faces = self.cached_faces[idx]
        confidence = self.extra_features[idx]
        labels = self.labels[idx]
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
        # 注意该排序会同时做离散化操作
        vertices, faces,labels,confidence = \
            sort_vertices_and_faces_and_labels_and_features(vertices, faces, labels, confidence, self.num_tokens)
        triangles = vertices[faces, :]
        # triangles, normals, areas, angles, vertices, faces = create_feature_stack(vertices, faces, self.num_tokens)
        triangles, areas, angles = create_feature_stack_from_triangles(triangles)

        features = np.hstack([triangles, confidence, areas, angles])
        face_neighborhood = np.array(trimesh.Trimesh(vertices=vertices, faces=faces, process=False).face_neighborhood)  # type: ignore
        target = torch.from_numpy(labels).long() - 1
        return features, target, vertices, faces, face_neighborhood


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


def create_feature_stack_from_triangles(triangles):
    t_areas = area(triangles) * 1e3
    t_angles = angle(triangles) / float(np.pi)
    return triangles.reshape(-1, 9), t_areas.reshape(-1, 1), t_angles.reshape(-1, 3)