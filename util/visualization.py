import json
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import trimesh
from PIL import Image
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely import Polygon

from dataset import newface_token, stopface_token, padface_token, sort_vertices_and_faces_and_labels_and_features
import shapefile
from pycocotools.coco import COCO
from skimage import io
from util.s3d_data_load import global_label_colors

# 约定-平面shp文件夹下各子文件的文件名
DATA_VERTICE_FILENAME = "vertexes.shp"
DATA_EDGE_FILENAME = "edges.shp"
DATA_FACE_FILENAME = "poly.shp"

# 约定-edges.shp文件中的属性字段名
PROPERTY_FACE_CONFIDENCE = "confidence"
PROPERTY_FACE_P0 = "pnt0"  # 三角面的顶点1的序号
PROPERTY_FACE_P1 = "pnt1"  # 三角面的顶点2的序号
PROPERTY_FACE_P2 = "pnt2"  # 三角面的顶点3的序号
PROPERTY_FACE_LABEL = "label"  # 面片label


colors_12 = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58230",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd7b4"
]


def visualize_points(points, vis_path, colors=None):
    if colors is None:
        Path(vis_path).write_text("\n".join(f"v {p[0]} {p[1]} {p[2]} 127 127 127" for p in points))
    else:
        Path(vis_path).write_text("\n".join(f"v {p[0]} {p[1]} {p[2]} {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}" for i, p in enumerate(points)))


def tokens_to_vertices(token_sequence, num_tokens):
    try:
        end = token_sequence.index(num_tokens + 1)
    except ValueError:
        end = len(token_sequence)
    token_sequence = token_sequence[:end]
    token_sequence = token_sequence[:(len(token_sequence) // 3) * 3]
    vertices = (np.array(token_sequence).reshape(-1, 3)) / num_tokens - 0.5
    # order: Z, Y, X --> X, Y, Z
    vertices = np.stack([vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=-1)
    return vertices


def visualize_quantized_mesh_vertices(token_sequence, num_tokens, output_path):
    vertices = tokens_to_vertices(token_sequence, num_tokens)
    plot_vertices(vertices, output_path)


def visualize_quantized_mesh_vertices_and_faces(token_sequence_vertex, token_sequence_face, num_tokens, output_path):
    vertices, faces = tokens_to_mesh(token_sequence_vertex, token_sequence_face, num_tokens)
    plot_vertices_and_faces(vertices, faces, output_path)


def plot_vertices(vertices, output_path):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    plt.xlim(-0.35, 0.35)
    plt.ylim(-0.35, 0.35)
    # Don't mess with the limits!
    plt.autoscale(False)
    ax.set_axis_off()
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='g', s=10)
    ax.set_zlim(-0.35, 0.35)
    ax.view_init(25, -120, 0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close("all")


def plot_vertices_and_faces(vertices, faces, output_path):
    ngons = [[vertices[v, :].tolist() for v in f] for f in faces]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.xlim(-0.45, 0.45)
    plt.ylim(-0.45, 0.45)
    # Don't mess with the limits!
    plt.autoscale(False)
    ax.set_axis_off()
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='black', s=10)
    polygon_collection = Poly3DCollection(ngons)
    polygon_collection.set_alpha(0.3)
    polygon_collection.set_color('b')
    ax.add_collection(polygon_collection)
    ax.set_zlim(-0.35, 0.35)
    ax.view_init(25, -120, 0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close("all")


def plot_vertices_and_faces_with_labels(vertices, faces, labels,output_path):
    # ngons = [[vertices[v, :].tolist() for v in f] for f in faces]
    ngons = np.array([[vertices[face[0]][:2], vertices[face[1]][:2], vertices[face[2]][:2]] for face in faces])
    attribute = np.array(labels)
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', global_label_colors, N=len(global_label_colors))
    tris = PolyCollection(ngons, array=attribute, cmap=custom_cmap)
    fig, ax = plt.subplots()
    ax.add_collection(tris)
    ax.autoscale()
    plt.savefig(output_path, dpi=1200)
    plt.close("all")


def plot_ground_truth_and_prediction(vertices, faces, gt_labels, pred_labels, output_path):
    ngons = np.array([[vertices[face[0]][:2], vertices[face[1]][:2], vertices[face[2]][:2]] for face in faces])
    gt_attribute = np.array(gt_labels)
    pred_attribute = np.array(pred_labels)
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', global_label_colors, N=len(global_label_colors))
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_aspect('equal')
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_aspect('equal')
    tris_gt = PolyCollection(ngons, array=gt_attribute, cmap=custom_cmap)
    tris_pred = PolyCollection(ngons, array=pred_attribute, cmap=custom_cmap)
    ax1.add_collection(tris_gt)
    ax2.add_collection(tris_pred)
    plt.savefig(output_path)
    plt.close("all")

def export_mesh_to_obj(vertices, faces, output_path):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(output_path)


def export_face_to_obj(faces, output_path):
    def save_line_segments_to_obj(filename, vertices, lines):
        with open(filename, 'w') as file:
            # 写入顶点数据
            for vertex in vertices:
                file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

            # 写入线段数据
            for line in lines:
                # 线段格式是用 'l' 来表示线段，注意索引从 1 开始
                file.write(f"l {line[0]} {line[1]}\n")

    lines = []
    vertices = []
    vertices_id = 1
    for face in faces:
        for i in range(len(face) - 1):
            lines.append([i + vertices_id, (i + 1) % (len(face)-1) + vertices_id])
            vertices.append(face[i])
        vertices_id = vertices_id + len(face) - 1
    save_line_segments_to_obj(output_path, vertices, lines)


def save_face_shp_with_label(output_path,vertices,faces,label):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    faces_file = os.path.join(output_path, DATA_FACE_FILENAME)

    writer = shapefile.Writer(faces_file)
    writer.field(PROPERTY_FACE_LABEL, "N", 5)  # 修改为字符串类型
    writer.field(PROPERTY_FACE_P0, "N", 5)
    writer.field(PROPERTY_FACE_P1, "N", 5)
    writer.field(PROPERTY_FACE_P2, "N", 5)

    for face_id, face in enumerate(faces):
        # 获取面的顶点坐标
        p0 = vertices[face[0]]
        p1 = vertices[face[1]]
        p2 = vertices[face[2]]

        # 创建多边形
        polygon = [
            [p0[0], p0[1]],
            [p1[0], p1[1]],
            [p2[0], p2[1]],
            [p0[0], p0[1]]  # 闭合多边形
        ]

        writer.poly([polygon])
        writer.record(
            label[face_id],
            face[0],
            face[1],
            face[2],
        )
    # 保存并关闭文件
    writer.close()


def export_mesh_to_shp(vertices,faces,labels,output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    vertices_file = os.path.join(output_path, DATA_VERTICE_FILENAME)
    faces_file = os.path.join(output_path, DATA_FACE_FILENAME)

    writer = shapefile.Writer(vertices_file)
    writer.field("PROPERTY_VERTICE_X", "N", 5, 0)
    writer.field("PROPERTY_VERTICE_Y", "N", 5, 0)
    for vertex in vertices:
        writer.point(vertex[0], vertex[1])
    for vertex_id, vertex in enumerate(vertices):
        writer.record(vertex[0], vertex[1])
    writer.close()

    writer = shapefile.Writer(faces_file)
    writer.field(PROPERTY_FACE_LABEL, "N", 5)  # 修改为字符串类型
    writer.field(PROPERTY_FACE_P0, "N", 5)
    writer.field(PROPERTY_FACE_P1, "N", 5)
    writer.field(PROPERTY_FACE_P2, "N", 5)

    for face_id, face in enumerate(faces):
        # 获取面的顶点坐标
        p0 = vertices[face[0]]
        p1 = vertices[face[1]]
        p2 = vertices[face[2]]

        # 创建多边形
        polygon = [
            [p0[0], p0[1]],
            [p1[0], p1[1]],
            [p2[0], p2[1]],
            [p0[0], p0[1]]  # 闭合多边形
        ]

        writer.poly([polygon])
        writer.record(
            labels[face_id],
            face[0],
            face[1],
            face[2],
        )
    # 保存并关闭文件
    writer.close()

def visualize_quantized_mesh_vertices_gif(token_sequence, num_tokens, output_dir):
    vertices = tokens_to_vertices(token_sequence, num_tokens)
    visualize_mesh_vertices_gif(vertices, output_dir)


def visualize_mesh_vertices_gif(vertices, output_dir):
    for i in range(1, len(vertices), 1):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
        plt.xlim(-0.35, 0.35)
        plt.ylim(-0.35, 0.35)
        # Don't mess with the limits!
        plt.autoscale(False)
        ax.set_axis_off()
        ax.scatter(vertices[:i, 0], vertices[:i, 1], vertices[:i, 2], c='g', s=10)
        ax.set_zlim(-0.35, 0.35)
        ax.view_init(25, -120, 0)
        plt.tight_layout()
        plt.savefig(output_dir / f"{i:05d}.png")
        plt.close("all")
    create_gif(output_dir, 40, output_dir / "vis.gif")


def visualize_quantized_mesh_vertices_and_faces_gif(token_sequence_vertex, token_sequence_face, num_tokens, output_dir):
    visualize_quantized_mesh_vertices_gif(token_sequence_vertex, num_tokens, output_dir)
    vertices, faces = tokens_to_mesh(token_sequence_vertex, token_sequence_face, num_tokens)
    visualize_mesh_vertices_and_faces_gif(vertices, faces, output_dir)


def visualize_mesh_vertices_and_faces_gif(vertices, faces, output_dir):
    ngons = [[vertices[v, :].tolist() for v in f] for f in faces]
    for i in range(1, len(ngons) + 1, 1):
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection="3d")
        plt.xlim(-0.35, 0.35)
        plt.ylim(-0.35, 0.35)
        # Don't mess with the limits!
        plt.autoscale(False)
        ax.set_axis_off()
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='black', s=10)
        polygon_collection = Poly3DCollection(ngons[:i])
        polygon_collection.set_alpha(0.3)
        polygon_collection.set_color('b')
        ax.add_collection(polygon_collection)
        ax.set_zlim(-0.35, 0.35)
        ax.view_init(25, -120, 0)
        plt.tight_layout()
        plt.savefig(output_dir / f"{len(vertices) + i:05d}.png")
        plt.close("all")
    create_gif(output_dir, 40, output_dir / "vis.gif")


def create_gif(folder, fps, output_path):
    collection_rgb = []
    for f in sorted([x for x in folder.iterdir() if x.suffix == ".png" or x.suffix == ".jpg"]):
        img_rgb = np.array(Image.open(f).resize((384, 384)))
        collection_rgb.append(img_rgb)
    clip = ImageSequenceClip(collection_rgb, fps=fps)
    clip.write_gif(output_path, verbose=False, logger=None)


def tokens_to_mesh(vertices_q, face_sequence, num_tokens):
    vertices = (np.array(vertices_q).reshape(-1, 3)) / num_tokens - 0.5
    # order: Z, Y, X --> X, Y, Z
    vertices = np.stack([vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=-1)
    try:
        end = face_sequence.index(stopface_token)
    except ValueError:
        end = len(face_sequence)
    face_sequence = face_sequence[:end]
    face_sequence = [x for x in face_sequence if x != 2]  # remove padding
    faces = []
    current_face = []
    for i in range(len(face_sequence)):
        if face_sequence[i] == newface_token:
            if len(current_face) > 2:
                faces.append(current_face)
            current_face = []
        else:
            current_face.append(face_sequence[i] - 3)
    if len(current_face) != 0:
        faces.append(current_face)
    return vertices, faces


def visualize_discrete_mesh(vertices, faces, labels, features, output_path,discrete = 128):
    vertices,faces,labels,features = \
        sort_vertices_and_faces_and_labels_and_features(vertices, faces, labels, features,discrete)
    plot_vertices_and_faces_with_labels(vertices, faces, labels, output_path)


def ngon_to_obj(vertices, faces):
    obj = ""
    for i in range(len(vertices)):
        obj += f"v {vertices[i, 0]} {vertices[i, 1]} {vertices[i, 2]}\n"
    for i in range(len(faces)):
        fline = "f"
        for j in range(len(faces[i])):
            fline += f" {faces[i][j] + 1} "
        fline += "\n"
        obj += fline
    return obj


def trisoup_sequence_to_mesh(soup_sequence, num_tokens):
    try:
        end = soup_sequence.index(stopface_token)
    except ValueError:
        end = len(soup_sequence)
    soup_sequence = soup_sequence[:end]
    vertices_q = []
    current_subsequence = []
    for i in range(len(soup_sequence)):
        if soup_sequence[i] == newface_token:
            if len(current_subsequence) >= 9:
                current_subsequence = current_subsequence[:9]
                vertices_q.append(np.array(current_subsequence).reshape(3, 3))
            current_subsequence = []
        elif soup_sequence[i] != padface_token:
            current_subsequence.append(soup_sequence[i] - 3)
    if len(current_subsequence) >= 9:
        current_subsequence = current_subsequence[:9]
        vertices_q.append(np.array(current_subsequence).reshape(3, 3))
    vertices = (np.array(vertices_q).reshape(-1, 3)) / num_tokens - 0.5
    # order: Z, Y, X --> X, Y, Z
    vertices = np.stack([vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=-1)
    faces = np.array(list(range(len(vertices_q) * 3)), dtype=np.int32).reshape(-1, 3)
    return vertices, faces


def ngonsoup_sequence_to_mesh(soup_sequence, num_tokens):
    try:
        end = soup_sequence.index(stopface_token)
    except ValueError:
        end = len(soup_sequence)
    soup_sequence = soup_sequence[:end]
    vertices_q = []
    face_ctr = 0
    faces = []
    current_subsequence = []
    for i in range(len(soup_sequence)):
        if soup_sequence[i] == newface_token:
            current_subsequence = current_subsequence[:len(current_subsequence) // 3 * 3]
            if len(current_subsequence) > 0:
                vertices_q.append(np.array(current_subsequence).reshape(-1, 3))
                faces.append([x for x in range(face_ctr, face_ctr + len(current_subsequence) // 3)])
                face_ctr += (len(current_subsequence) // 3)
            current_subsequence = []
        elif soup_sequence[i] != padface_token:
            current_subsequence.append(soup_sequence[i] - 3)

    current_subsequence = current_subsequence[:len(current_subsequence) // 3 * 3]
    if len(current_subsequence) > 0:
        vertices_q.append(np.array(current_subsequence).reshape(-1, 3))
        faces.append([x for x in range(face_ctr, face_ctr + len(current_subsequence) // 3)])
        face_ctr += (len(current_subsequence) // 3)

    vertices = np.vstack(vertices_q) / num_tokens - 0.5
    # order: Z, Y, X --> X, Y, Z
    vertices = np.stack([vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=-1)
    return vertices, faces


def triangle_sequence_to_mesh(triangles):
    vertices = triangles.reshape(-1, 3)
    faces = np.array(list(range(vertices.shape[0]))).reshape(-1, 3)
    return vertices, faces


def visualization_seg(num_image, json_path, img_path, str=' '):
    # 需要画图的是第num副图片, 对应的json路径和图片路径,
    # str = ' '为类别字符串，输入必须为字符串形式 'str'，若为空，则返回所有类别id
    coco = COCO(json_path)

    catIds = coco.getCatIds()  # 获取指定类别 id
    exclude_cat_ids = [17,18]  # 替换为你要排除的类别 ID

    # 使用列表推导式获取排除指定 ID 之外的所有类别 ID
    include_cat_ids = [cat_id for cat_id in catIds if cat_id not in exclude_cat_ids]
    # print(include_cat_ids)

    imgIds = coco.getImgIds()  # 获取图片i
    # print(catIds,imgIds)
    img = coco.loadImgs(num_image)[0]  # 加载图片,loadImgs() 返回的是只有一个内嵌字典元素的list, 使用[0]来访问这个元素
    image = io.imread(os.path.join(img_path,img['file_name']))

    annIds = coco.getAnnIds(imgIds=img['id'], catIds=include_cat_ids, iscrowd=None)
    anns = coco.loadAnns(annIds)


    plt.imshow(image,cmap="gray")
    coco.showAnns(anns)

    # cv2.imwrite(f"gt_on_densitymap/{imgIds[num_image - 1]}.png", image)
    ax = plt.gca()
    for i, ann in enumerate(anns):
        # 假设每个标注都有一个中心点，这里我们简单地使用bbox的中心作为示例
        bbox = ann['bbox']
        x, y, w, h = bbox
        center_x = x + w / 2
        center_y = y + h / 2
        # 添加顺序数字（根据需求调整位置和样式）
        plt.text(center_x, center_y, i, color='red', fontsize=12, ha='center', va='center')

    # 显示图像
    plt.axis('off')

    plt.show()


def visualization_seg_with_custom_anno(num_image, json_path, img_path, coco_anno_dict):
    # 需要画图的是第num副图片, 对应的json路径和图片路径,
    # str = ' '为类别字符串，输入必须为字符串形式 'str'，若为空，则返回所有类别id
    coco = COCO(json_path)

    catIds = coco.getCatIds()  # 获取指定类别 id
    exclude_cat_ids = [17,18]  # 替换为你要排除的类别 ID

    # 使用列表推导式获取排除指定 ID 之外的所有类别 ID
    include_cat_ids = [cat_id for cat_id in catIds if cat_id not in exclude_cat_ids]
    # print(include_cat_ids)

    imgIds = coco.getImgIds()  # 获取图片i
    # print(catIds,imgIds)
    img = coco.loadImgs(num_image)[0]  # 加载图片,loadImgs() 返回的是只有一个内嵌字典元素的list, 使用[0]来访问这个元素
    image = io.imread(os.path.join(img_path,img['file_name']))
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=include_cat_ids, iscrowd=None)
    ori_anns = coco.loadAnns(annIds)
    anns = coco_anno_dict

    plt.imshow(image,cmap="gray")
    coco.showAnns(anns)

    # cv2.imwrite(f"gt_on_densitymap/{imgIds[num_image - 1]}.png", image)
    ax = plt.gca()
    for i, ann in enumerate(anns):
        # 假设每个标注都有一个中心点，这里我们简单地使用bbox的中心作为示例
        bbox = ann['bbox']
        x, y, w, h = bbox
        center_x = x + w / 2
        center_y = y + h / 2
        # 添加顺序数字（根据需求调整位置和样式）
        plt.text(center_x, center_y, i, color='red', fontsize=12, ha='center', va='center')

    # 显示图像
    plt.axis('off')

    plt.show()



def visualization_bbox1(num_image, json_path, img_path):  # 需要画的第num副图片， 对应的json路径和图片路径
    with open(json_path) as annos:
        annotation_json = json.load(annos)

    print('the annotation_json num_key is:', len(annotation_json))  # 统计json文件的关键字长度
    print('the annotation_json key is:', annotation_json.keys())  # 读出json文件的关键字
    print('the annotation_json num_images is:', len(annotation_json['images']))  # json文件中包含的图片数量

    image_name = annotation_json['images'][num_image - 1]['file_name']  # 读取图片名
    id = annotation_json['images'][num_image - 1]['id']  # 读取图片id

    image_path = os.path.join(img_path, str(image_name).zfill(5))  # 拼接图像路径
    image = cv2.imread(image_path, 1)  # 保持原始格式的方式读取图像
    num_bbox = 0  # 统计一幅图片中bbox的数量

    for i in range(len(annotation_json['annotations'][::])):
        if annotation_json['annotations'][i - 1]['image_id'] == id:
            num_bbox = num_bbox + 1
            x, y, w, h = annotation_json['annotations'][i - 1]['bbox']  # 读取边框
            image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)

    print('The unm_bbox of the display image is:', num_bbox)

    # 显示方式1：用plt.imshow()显示
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) #绘制图像，将CV的BGR换成RGB
    plt.show() #显示图像

    # 显示方式2：用cv2.imshow()显示
    # cv2.namedWindow(image_name, 0)  # 创建窗口
    # cv2.resizeWindow(image_name, 1000, 1000)  # 创建500*500的窗口
    # cv2.imshow(image_name, image)
    # cv2.waitKey(0)


def visualization_bbox_seg(num_image, json_path, img_path, *str):  # 需要画图的是第num副图片， 对应的json路径和图片路径

    coco = COCO(json_path)

    if len(str) == 0:
        catIds = []
    else:
        catIds = coco.getCatIds(catNms=[str[0]])  # 获取给定类别对应的id 的dict（单个内嵌字典的类别[{}]）
        catIds = coco.loadCats(catIds)[0]['id']  # 获取给定类别对应的id 的dict中的具体id

    list_imgIds = coco.getImgIds(catIds=catIds)  # 获取含有该给定类别的所有图片的id
    img = coco.loadImgs(list_imgIds[num_image - 1])[0]  # 获取满足上述要求，并给定显示第num幅image对应的dict
    image = io.imread(img_path + img['file_name'])  # 读取图像
    image_name = img['file_name']  # 读取图像名字
    image_id = img['id']  # 读取图像id
    print(image_id)
    img_annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)  # 读取这张图片的所有seg_id
    img_anns = coco.loadAnns(img_annIds)

    for i in range(len(img_annIds)):
        x, y, w, h = img_anns[i - 1]['bbox']  # 读取边框
        image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)

    #plt.rcParams['figure.figsize'] = (20.0, 20.0)
    plt.imshow(image)
    coco.showAnns(img_anns)
    plt.show()


def plot_floorplan_with_rooms_and_bound(add_coords_faces,bound,title="floorplan",save_path = None):
    polygons = [
        Polygon([point for point in room]) for room in add_coords_faces]
    plt.figure()
    colors = colors_12
    for i,polygon in enumerate(polygons) :
        x, y = polygon.exterior.xy
        plt.fill(x, y, alpha=0.5, fc=colors[i % len(colors)], ec='black')  # 填充多边形
    bound = [
        [bound[0],bound[1]],
        [bound[2],bound[1]],
        [bound[2],bound[3]],
        [bound[0],bound[3]]]
    bound = Polygon(bound)
    x, y = bound.exterior.xy
    plt.plot(x, y, color = 'black')  # 绘制边界

    # 设置图形属性
    plt.title(title)
    plt.axis('equal')  # 确保坐标轴比例一致
    plt.axis('off') #
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


def plot_trimesh_with_labels(trimesh, labels, title="trimesh",save_path = None):
    coords = []
    for face in trimesh['faces']:
        coord = [trimesh['vertices'][i] for i in face]
        coords.append(coord)
    polygons = [
        Polygon(face) for face in coords]
    plt.figure()
    for i,polygon in enumerate(polygons) :
        x, y = polygon.exterior.xy
        try:
            plt.fill(x, y, alpha=1, fc=global_label_colors[labels[i] - 1], ec='black',linewidth=0.2)  # 填充多边形
        except Exception as e:
            print(e)
            return
        # plt.plot(x, y, color='white')  # 绘制边界

    # 设置图形属性
    plt.title(title)
    plt.axis('equal')  # 确保坐标轴比例一致
    plt.axis('off') #
    # plt.show()
    if save_path is not None:
        plt.savefig(save_path,dpi=300)
    plt.close()

def plot_floorplan_with_regions(regions, corners=None, edges=None, scale=256):
    """Draw floorplan map where different colors indicate different rooms
    """
    colors = colors_12

    regions = [(region * scale / 256).round().astype(np.int_) for region in regions]

    # define the color map
    room_colors = [colors[i] for i in range(len(regions))]

    colorMap = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in room_colors]
    colorMap = np.asarray(colorMap)
    if len(regions) > 0:
        colorMap = np.concatenate([np.full(shape=(1, 3), fill_value=0), colorMap], axis=0).astype(
            np.uint8)
    else:
        colorMap = np.concatenate([np.full(shape=(1, 3), fill_value=0)], axis=0).astype(
            np.uint8)
    # when using opencv, we need to flip, from RGB to BGR
    colorMap = colorMap[:, ::-1]

    alpha_channels = np.zeros(colorMap.shape[0], dtype=np.uint8)
    alpha_channels[1:len(regions) + 1] = 150

    colorMap = np.concatenate([colorMap, np.expand_dims(alpha_channels, axis=-1)], axis=-1)

    room_map = np.zeros([scale, scale]).astype(np.int32)
    # sort regions
    if len(regions) > 1:
        avg_corner = [region.mean(axis=0) for region in regions]
        ind = np.argsort(np.square(np.array(avg_corner)).sum(axis=1), axis=0)
        regions = [regions[i] for i in ind]

    for idx, polygon in enumerate(regions):
        cv2.fillPoly(room_map, [polygon], color=idx + 1)

    image = colorMap[room_map.reshape(-1)].reshape((scale, scale, 4))

    pointColor = (0,0,0,255)
    lineColor = (0,0,0,255)

    for region in regions:
        for i, point in enumerate(region):
            if i == len(region)-1:
                cv2.line(image, tuple(point), tuple(region[0]), color=lineColor, thickness=5)
            else:
                cv2.line(image, tuple(point), tuple(region[i+1]), color=lineColor, thickness=5)

    for region in regions:
        for i, point in enumerate(region):
            cv2.circle(image, tuple(point), color=pointColor, radius=12, thickness=-1)
            cv2.circle(image, tuple(point), color=(255, 255, 255, 0), radius=6, thickness=-1)

    return image


def plot_room_map(preds, room_map, im_size=256):
    """Draw room polygons overlaid on the density map
    """
    for i, corner in enumerate(preds):
        if i == len(preds) - 1:
            cv2.line(room_map, (round(corner[0]), round(corner[1])), (round(preds[0][0]), round(preds[0][1])),
                     (252, 252, 0), 2)
        else:
            cv2.line(room_map, (round(corner[0]), round(corner[1])), (round(preds[i + 1][0]), round(preds[i + 1][1])),
                     (252, 252, 0), 2)
        cv2.circle(room_map, (round(corner[0]), round(corner[1])), 2, (0, 0, 255), 2)
        cv2.putText(room_map, str(i), (round(corner[0]), round(corner[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 1, cv2.LINE_AA)

    return room_map


def plot_standard_point_cloud_density(
        points, save_path,
        min_axis_raster_size=800,
        log_density = False,
        acrtan_density = False,
):
    '''
    points: (N, 3) numpy array

    '''

    max_coords = np.max(points, axis=0)
    min_coords = np.min(points, axis=0)

    max_m_min = max_coords - min_coords
    # 填充，向外填充10%
    max_coords = max_coords + 0.1 * max_m_min
    min_coords = min_coords - 0.1 * max_m_min

    max_m_min = max_coords - min_coords

    width_range = max_m_min[0]
    height_range = max_m_min[1]

    img_res = []
    if width_range > height_range:
        img_res = [min_axis_raster_size, int(min_axis_raster_size * height_range / width_range)]
    else:
        img_res = [int(min_axis_raster_size * width_range / height_range), min_axis_raster_size]
    img_res = np.array(img_res)

    # 真实坐标向栅格坐标转换
    coordinates = \
        np.round(
            (points[:, :2] - min_coords[None, :2]) / (max_coords[None, :2] - min_coords[None, :2]) * img_res[None])

    coordinates = np.minimum(np.maximum(coordinates, np.zeros_like(img_res)),
                                img_res - 1)

    density = np.zeros((img_res[1],img_res[0]), dtype=np.float32)
    unique_coordinates, counts = np.unique(coordinates, return_counts=True, axis=0)
    unique_coordinates = unique_coordinates.astype(np.int32)

    density[unique_coordinates[:, 1], unique_coordinates[:, 0]] = counts
    density = density / np.max(density)

    if log_density:
        density = np.log(density + 1)/np.log(2)
    if acrtan_density:
        density = np.arctan(density * 5) / np.arctan(5)

    img_name = 'density_map'
    if log_density:
        img_name = 'log_' + img_name
    elif acrtan_density:
        img_name = 'acrtan_' + img_name
    density_path = os.path.join(save_path, img_name+'.png')
    density_uint8 = (density * 255).astype(np.uint8)
    density_uint8 = 255 - density_uint8
    cv2.imwrite(density_path, density_uint8)




