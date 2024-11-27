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
from util.misc import scale_vertices, normalize_vertices, shift_vertices

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from torch_geometric.data import Data as GeometricData
from torch_geometric.loader.dataloader import Collater as GeometricCollator

# 约定-平面shp文件夹下各子文件的文件名
DATA_VERTICE_FILENAME = "vertexes.shp"
DATA_EDGE_FILENAME = "edges.shp"
DATA_FACE_FILENAME = "poly.shp"
# 约定-vertexes.shp文件中的属性字段名
PROPERTY_VERTICE_CONFIDENCE = "confidence"
# 约定-edges.shp文件中的属性字段名
PROPERTY_EDGE_CONFIDENCE = "confidence"
PROPERTY_EDGE_P0 = "pnt0"  # 边所连接的端点1的序号
PROPERTY_EDGE_P1 = "pnt1"  # 边所连接的端点2的序号
# 约定-edges.shp文件中的属性字段名
PROPERTY_FACE_CONFIDENCE = "confidence"
PROPERTY_FACE_P0 = "pnt0"  # 三角面的顶点1的序号
PROPERTY_FACE_P1 = "pnt1"  # 三角面的顶点2的序号
PROPERTY_FACE_P2 = "pnt2"  # 三角面的顶点3的序号
PROPERTY_FACE_LABEL = "label"  # 三角面的顶点3的序号

def get_vertices_data(vertices_file):
    """
    从点shp文件中提取单个平面图中的所有点信息
    :param vertices_file: 平面图 点shp文件的路径
    :return: (points, confidences) 返回两个list，第一个list装载所有点的坐标对（p_x,p_y）,
    : 第二个list装载所有点的置信度
    """
    sf = shapefile.Reader(vertices_file)
    shapes = sf.shapes()
    records = sf.records()
    # shapes中的每一个shape，其points属性中只有一个点，故通过points[0]可以拿到该点
    # 因此points[0][0]拿到该点的x坐标，points[0][1]拿到该点的y坐标，我们用一个元组(px,py)来记录单个点
    vertices = [(float(shape.points[0][0]), float(shape.points[0][1]), 0.0) for shape in shapes]
    vertices_confidence = [float(r[PROPERTY_VERTICE_CONFIDENCE]) for r in records]
    return vertices, vertices_confidence


def get_edges_data(edge_file):
    """
    从边shp文件中提取单个平面图中的所有边信息
    :param edge_file: 平面图 边shp文件的路径
    :return: (edges, confidences, point_id_to_edge_id),返回两个list和一个dict,
    :第一个list装载所有边的两端点序号，第二个list装载置信度，dict装载点序号到边的快速索引
    """
    sf = shapefile.Reader(edge_file)
    shapes = sf.shapes()
    records = sf.records()
    edges = [(r[PROPERTY_EDGE_P0], r[PROPERTY_EDGE_P1]) for r in records]

    # point_id_to_edges_id = { point_id : [edge1_id,edge2_id,...] }
    # 如此，可以根据点的id快速锁定该点连接的边的id，从而，我们可以依据两个点id快速锁定一个边的id
    point_id_to_edges_id = {}
    for index, r in enumerate(records):
        point1_id = r[PROPERTY_EDGE_P0]
        point2_id = r[PROPERTY_EDGE_P1]

        # 以防止首次访问point1_id，没有找到key出错
        if point1_id in point_id_to_edges_id:
            point_id_to_edges_id[point1_id].append(index)
        else:
            point_id_to_edges_id[point1_id] = [index]
        # 同理
        if point2_id in point_id_to_edges_id:
            point_id_to_edges_id[point2_id].append(index)
        else:
            point_id_to_edges_id[point2_id] = [index]

    edges_confidence = [float(r[PROPERTY_EDGE_CONFIDENCE]) for r in records]
    # 知道可以用一个循环解决三个lit的生成，但是上面这么写简洁一些，可读性好一点
    return edges, edges_confidence, point_id_to_edges_id


def get_faces_data(face_file):
    """
    从面shp文件中提取单个平面中的所有三角形的信息
    :param face_file: 平面图 面shp文件的路径
    :return: (faces, faces_confidences, faces_label), 返回三个NumPy数组,
             第一个数组装载所有三角形的顶点序号，
             第二个数组装载所有三角形的置信度，
             第三个是该三角形的label
    """
    sf = shapefile.Reader(face_file)
    records = sf.records()

    # 使用一次遍历提取所有需要的信息
    faces = []
    faces_confidences = []
    faces_label = []
    hasNagative = False

    for r in records:
        faces.append((r[PROPERTY_FACE_P0], r[PROPERTY_FACE_P1], r[PROPERTY_FACE_P2]))
        faces_confidences.append(float(r[PROPERTY_FACE_CONFIDENCE]))

        # 如果 label 为 -2，则将其改为 -1，然后把所有值都加1，保证范围在0以上
        label = r[PROPERTY_FACE_LABEL]
        if label <= -2:
            label = -1
        if label == -1:
            hasNagative = True

        faces_label.append(label)

    if hasNagative == True:
        faces_label += 1

    return faces, faces_confidences, faces_label


def get_edge_id_of_face(face: tuple, point_id_to_edge_id):
    """
    找出该面对应的edge编号
    :param face:
    :param point_id_to_edge_id:
    :return:
    """
    v0, v1, v2 = face[0], face[1], face[2]
    edges_on_v0 = point_id_to_edge_id[v0]
    edges_on_v1 = point_id_to_edge_id[v1]
    edges_on_v2 = point_id_to_edge_id[v2]
    # 根据两个顶点之间关联的边的交集锁定 两个点之间的边
    e0 = list(set(edges_on_v1).intersection(edges_on_v2))
    e1 = list(set(edges_on_v0).intersection(edges_on_v2))
    e2 = list(set(edges_on_v0).intersection(edges_on_v1))
    # 两个点之间的边 应该有且仅有一条
    assert len(e0) == 1
    assert len(e1) == 1
    assert len(e2) == 1
    e0, e1, e2 = e0[0], e1[0], e2[0]
    return e0, e1, e2


def read_vertexes_and_faces(obj_root_path):
    vertices_file = os.path.join(obj_root_path, DATA_VERTICE_FILENAME)
    vertices, v_confidence = get_vertices_data(vertices_file)
    edge_file = os.path.join(obj_root_path, DATA_EDGE_FILENAME)
    edges, e_confidence, point_id_to_edge_id = get_edges_data(edge_file)
    face_file = os.path.join(obj_root_path, DATA_FACE_FILENAME)
    faces, f_confidence, labels = get_faces_data(face_file)
    face_feature = [0] * len(faces)  # 构造faces同等长度的列表，初值填充0
    for index, (v1, v2, v3) in enumerate(faces):
        # 找到每个面的顶点所关联的边
        e1, e2, e3 = get_edge_id_of_face(faces[index], point_id_to_edge_id)
        face_feature[index] = [
            v_confidence[v1],
            v_confidence[v2],
            v_confidence[v3],
            e_confidence[e1],
            e_confidence[e2],
            e_confidence[e3],
            f_confidence[index],
        ]

    vertices = np.array(vertices)
    faces = np.array(faces)
    face_feature = np.array(face_feature)
    labels = np.array(labels)
    return vertices, faces, face_feature, labels


class FPTriangleNodes(GeometricDataset):
    def __init__(self, config, split):
        super().__init__()
        data_path = Path(config.dataset_root)
        self.cached_vertices = []
        self.cached_faces = []
        self.extra_features = []
        # self.only_backward_edges = only_backward_edges
        self.num_tokens = config.num_tokens
        self.scale_augment = False
        self.shift_augment = False
        self.low_augment = False
        self.split_ratio = 0.8

        data = {"features": [], "vertices": [], "faces": [], "labels": []}
        for scene_name in tqdm(os.listdir(data_path)):
            scene_full_path = os.path.join(data_path, scene_name)
            vertices, faces, face_feature, labels = read_vertexes_and_faces(scene_full_path)
            data["features"].append(face_feature)
            data["vertices"].append(vertices)
            data["faces"].append(faces)
            data["labels"].append(labels)
            train_size = int(len(os.listdir(data_path)) * self.split_ratio)
        if split == "train":
            self.extra_features = data[f'features'][:train_size]
            self.cached_vertices = data[f'vertices'][:train_size]
            self.cached_faces = data[f'faces'][:train_size]
            self.labels = data[f'labels'][:train_size]
        elif split == "val":
            self.extra_features = data[f'features'][train_size:]
            self.cached_vertices = data[f'vertices'][train_size:]
            self.cached_faces = data[f'faces'][train_size:]
            self.labels = data[f'labels'][train_size:]
        print(len(self.cached_vertices), "meshes loaded")

    def len(self):
        return len(self.cached_vertices)

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
        triangles = vertices[faces, :].reshape(-1,9)
        # triangles, normals, areas, angles, vertices, faces = create_feature_stack(vertices, faces, self.num_tokens)
        features = np.hstack([triangles, confidence])
        face_neighborhood = np.array(trimesh.Trimesh(vertices=vertices, faces=faces, process=False).face_neighborhood)  # type: ignore
        target = torch.from_numpy(labels).int()
        return features, target, vertices, faces, face_neighborhood

    def get(self, idx):
        features, target, vertices, faces, face_neighborhood = self.get_all_features_for_shape(idx)
        return GeometricData(x=torch.from_numpy(features).float(), y=target,
                             edge_index=torch.from_numpy(face_neighborhood.T).long(),
                             num_vertices=vertices.shape[0], faces=torch.from_numpy(np.array(faces)).long())

    def save_sample_data_by_idx(self,idx,is_quantized=False):
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
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces,process=False)

        else:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        tris = np.array([[vertices[face[0]][:2], vertices[face[1]][:2], vertices[face[2]][:2]] for face in faces])

        # attribute = np.array(labels)
        attribute = np.array([f[6] for f in features])
        tris = PolyCollection(tris, array=attribute, cmap='RdYlGn')
        fig, ax = plt.subplots()
        ax.add_collection(tris)
        ax.autoscale()

        suffer = "quantized" if is_quantized else "non_quantized"
        current_path = os.getcwd()
        print("存放obj文件路径:", current_path)
        # mesh.show()
        mesh.export(f"{current_path}/mesh_{idx}_{suffer}.obj")

        plt.savefig(f'{current_path}/mesh_{idx}_{suffer}.png',dpi=1200)


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
