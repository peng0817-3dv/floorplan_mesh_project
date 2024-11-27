import numpy as np
import torch
import networkx as nx


newface_token = 0
stopface_token = 1
padface_token = 2


def get_shifted_sequence(sequence):
    non_special = np.flatnonzero(np.isin(sequence, [0, 1, 2], invert=True))
    if non_special.shape[0] > 0:
        idx = non_special[0]
        val = sequence[idx]
        sequence[non_special] -= (val - 3)
    return sequence


def read_faces(text):
    all_lines = text.splitlines()
    all_face_lines = [x for x in all_lines if x.startswith('f ')]
    all_faces = [[int(y.split('/')[0]) - 1 for y in x.strip().split(' ')[1:]] for x in all_face_lines]
    return all_faces


def read_vertices(text):
    all_lines = text.splitlines()
    all_vertex_lines = [x for x in all_lines if x.startswith('v ')]
    all_vertices = np.array([[float(y) for y in x.strip().split(' ')[1:]] for x in all_vertex_lines])
    assert all_vertices.shape[1] == 3, 'vertices should have 3 coordinates'
    return all_vertices


def quantize_coordinates(coords, num_tokens=256):
    if torch.is_tensor(coords):
        coords = torch.clip((coords + 0.5), 0, 1) * num_tokens  # type: ignore
        coords_quantized = coords.round().long()
    else:
        coords = np.clip((coords + 0.5), 0, 1) * num_tokens  # type: ignore
        coords_quantized = coords.round().astype(int)
    return coords_quantized


def face_to_cycles(face):
    """Find cycles in face."""
    g = nx.Graph()
    for v in range(len(face) - 1):
        g.add_edge(face[v], face[v + 1])
    g.add_edge(face[-1], face[0])
    return list(nx.cycle_basis(g))


def sort_vertices_and_faces(vertices_, faces_, num_tokens=256):
    vertices = np.clip((vertices_ + 0.5), 0, 1) * num_tokens  # type: ignore
    vertices_quantized_ = vertices.round().astype(int)

    vertices_quantized_ = vertices_quantized_[:, [2, 0, 1]] # 将xyz轴进行交换，使得xyz轴的顺序为Z,X,Y
    # 对于一维数组或者列表，np.unique() 函数 去除其中重复的元素 ，并按元素 由小到大 返回一个新的无元素重复的元组或者列表。
    vertices_quantized, unique_inverse = np.unique(vertices_quantized_, axis=0, return_inverse=True)
    # （n,3[z,x,y]) 转置为 （3[z,x,y],n）方便使用np.lexsort()排序
    #  np.lexsort()将会把数组按照y,x,z的顺序进行排序，返回排序后的索引值
    sort_inds = np.lexsort(vertices_quantized.T)

    vertices_quantized = vertices_quantized[sort_inds]
    # 执行前 （n,3[z,x,y])[以y,x,z的顺序排序]
    # 执行后 （n,3[y,x,z])[以y,x,z的顺序排序]
    vertices_quantized = np.stack([vertices_quantized[:, 2], vertices_quantized[:, 1], vertices_quantized[:, 0]], axis=-1)

    # Re-index faces and tris to re-ordered vertices.
    # 顶点经过了一次排序，我们将面中的顶点号进行相应的映射，以确保faces中的索引能索引到排序后的顶点上
    faces = [np.argsort(sort_inds)[unique_inverse[f]] for f in faces_]
    # Merging duplicate vertices and re-indexing the faces causes some faces to
    # contain loops (e.g [2, 3, 5, 2, 4]). Split these faces into distinct
    # sub-faces.
    sub_faces = []
    for f in faces:
        cliques = face_to_cycles(f)
        for c in cliques:
            c_length = len(c)
            # Only append faces with more than two verts.
            if c_length > 2:
                d = np.argmin(c)
                # Cyclically permute faces just that first index is the smallest.
                sub_faces.append([c[(d + i) % c_length] for i in range(c_length)])
    faces = sub_faces
    # Sort faces by lowest vertex indices. If two faces have the same lowest
    # index then sort by next lowest and so on.
    # 将面也进行排序，以第一个顶点号、第二个顶点号、第三个顶点号的顺序进行排序
    faces.sort(key=lambda f: tuple(sorted(f)))

    # After removing degenerate faces some vertices are now unreferenced.
    # Remove these.
    num_verts = vertices_quantized.shape[0]
    # 依据faces判断顶点序号的连接状态
    vert_connected = np.equal(
        np.arange(num_verts)[:, None], np.hstack(faces)[None]).any(axis=-1)
    vertices_quantized = vertices_quantized[vert_connected]
    # Re-index faces and tris to re-ordered vertices.
    vert_indices = (
            np.arange(num_verts) - np.cumsum(1 - vert_connected.astype('int')))
    # 再次重新调整faces的索引
    faces = [vert_indices[f].tolist() for f in faces]
    # 回归到[-1,1]范围内
    vertices = vertices_quantized / num_tokens - 0.5
    # order: Z, Y, X --> X, Y, Z
    # 上述是原注释，但是我疑惑，明明应该是将y,x,z转换为 z,x,y才对,如果要按照注释的话，则输入的点的坐标应该是y,z,x,但过于违背常识，倾向于认为注释错了
    vertices = np.stack([vertices[:, 1], vertices[:, 0], vertices[:, 2]], axis=-1)
    return vertices, faces