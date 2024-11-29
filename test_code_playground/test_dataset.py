import os
import pickle
from pathlib import Path
import trimesh
from trimesh.exchange import obj
import numpy as np
from dataset import sort_vertices_and_faces
from dataset.floorplan_triangles import FPTriangleNodes
import hydra

from util.misc import scale_vertices, normalize_vertices, shift_vertices


class test_Dataset:
    def __init__(self, config, split, scale_augment, shift_augment, low_augment, num_tokens,sample_size):
        self.config = config
        self.split = split
        self.scale_augment = scale_augment
        self.shift_augment = shift_augment
        self.low_augment = low_augment
        self.num_tokens = num_tokens
        self.sample_size = sample_size
        data_path = Path(config.dataset_root)
        with open(data_path, 'rb') as fptr:
            data = pickle.load(fptr)
            self.names = data[f'name_{split}'][:sample_size]
            self.cached_vertices = data[f'vertices_{split}'][:sample_size]
            self.cached_faces = data[f'faces_{split}'][:sample_size]
    def view_data_by_idx(self, idx,is_quantized=False):
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

        if is_quantized:
            vertices, faces = sort_vertices_and_faces(vertices, faces, self.num_tokens)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces,process=False)
            triangles = vertices[faces, :]
        else:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            triangles = vertices[faces, :]

        label_str = "quantized" if is_quantized else "non_quantized"
        current_path = os.getcwd()
        print("当前工作路径:", current_path)
        mesh.export(f"{current_path}/mesh_{idx}_{label_str}.obj")


@hydra.main(config_path='../config', config_name='meshgpt', version_base='1.2')
def main(config):
    sample_size = 32

    config.dataset_root = 'G:/workspace_plane2DDL/RefCode/meshgpt-official-version/MeshGPT/data/structure3d'

    dataset = FPTriangleNodes(config, 'train')
    data = dataset.get(0)
    print(data)



if __name__ == '__main__':
    main()