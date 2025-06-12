import json

import cv2
import numpy as np
import os

import tqdm

from util.visualization import export_face_to_obj


'''
process_vertice_by_op_record: 用于对数据集类输出某场景的顶点坐标进行变换，由于数据集类在载入时会将mesh的坐标变换到0-1范围，
根据其数据集类输出当前场景时额外的操作记录（op_rercord,operation record）来进行变换。存在bug，【弃置】.
process_vertice_by_ori_bound: 用于对数据集类输出某场景的顶点坐标进行变换，由于数据集类在载入时会将mesh的坐标变换到0-1范围，
根据其数据集类输出当前场景时额外的原始边界来进行变换.该函数更方便
'''


def process_vertice_by_op_record(reverse_op,vertices):
    for op in reverse_op:
        if op[0] == '+':
            vertices = np.stack([vertices[:, 0] + op[1][0], vertices[:, 1] + op[1][1], vertices[:, 2]], axis=-1)
        elif op[0] == '*':
            vertices = np.stack([vertices[:, 0] * op[1][0], vertices[:, 1] * op[1][1], vertices[:, 2]], axis=-1)

    return vertices


def process_vertice_by_ori_bound(ori_bound,vertices):
    # 找到每个坐标轴的最小值
    min_coords = np.min(vertices, axis=0)  # 形状为 (3,)，分别是 x, y, z 的最小值
    max_coords = np.max(vertices, axis=0)  # 形状为 (3,)，分别是 x, y, z 的最大值
    x_range = max_coords[0] - min_coords[0]
    y_range = max_coords[1] - min_coords[1]

    # 将点集移动到原点(方便缩放)
    centered_vertices = vertices - min_coords

    # 缩放到目标尺寸
    x_scale = (ori_bound[2] - ori_bound[0]) / x_range
    y_scale = (ori_bound[3] - ori_bound[1]) / y_range
    ori_scale_vertices_x = centered_vertices[:, 0] * x_scale
    ori_scale_vertices_y = centered_vertices[:, 1] * y_scale
    ori_scale_vertices = np.stack([ori_scale_vertices_x, ori_scale_vertices_y, centered_vertices[:, 2]], axis=-1)

    # 移动到目标位置
    ori_vertices = ori_scale_vertices + np.array([ori_bound[0], ori_bound[1], 0])

    return ori_vertices


'''
用于生成增强点云密度图及其一些相关操作
'''

def generate_augmented_point_cloud_density_map(scene_path, annotation_path,sacle_table):
    import laspy
    las_path = os.path.join(scene_path, 'scale.laz')

    scene_id = os.path.basename(scene_path).split('_')[-1]
    pc = laspy.read(las_path)
    number_points = len(pc)
    x = np.reshape(pc.x, (number_points, 1))
    y = np.reshape(pc.y, (number_points, 1))
    z = np.reshape(pc.z, (number_points, 1))
    xyz = np.hstack((x, y, z))
    density, normalization_dict = generate_density_by_distortion(xyz)
    augment_annos = scale_annos_by_augment_pc_scale(annotation_path,sacle_table)
    normalized_annos = normalize_annotations(augment_annos, normalization_dict)

    ### parse annotations
    polys = parse_floor_plan_polys(normalized_annos)

    return polys,normalized_annos,density


def generate_density_by_distortion(point_cloud, width=256, height=256):
    '''
    point_cloud: (N, 3) numpy array
    roomformer要求的缩放，会有失真
    '''


    # z轴反向
    ps = point_cloud * -1
    ps[:,0] *= -1
    ps[:,1] *= -1

    image_res = np.array((width, height))

    max_coords = np.max(ps, axis=0)
    min_coords = np.min(ps, axis=0)
    max_m_min = max_coords - min_coords

    # 填充，向外填充10%
    max_coords = max_coords + 0.1 * max_m_min
    min_coords = min_coords - 0.1 * max_m_min

    normalization_dict = {}
    normalization_dict["min_coords"] = min_coords
    normalization_dict["max_coords"] = max_coords
    normalization_dict["image_res"] = image_res


    # coordinates = np.round(points[:, :2] / max_coordinates[None,:2] * image_res[None])
    # 计算点云在[256，256]图像上的坐标
    coordinates = \
        np.round(
            (ps[:, :2] - min_coords[None, :2]) / (max_coords[None,:2] - min_coords[None, :2]) * image_res[None])
    # 限制坐标范围
    coordinates = np.minimum(np.maximum(coordinates, np.zeros_like(image_res)),
                                image_res - 1)

    #
    density = np.zeros((height, width), dtype=np.float32)
    # 统计每个图像像素格中落下点云的数目
    unique_coordinates, counts = np.unique(coordinates, return_counts=True, axis=0)
    # print(np.unique(counts))
    # counts = np.minimum(counts, 1e2)

    unique_coordinates = unique_coordinates.astype(np.int32)
    # 密度图每个像素的灰度值 = 落在该像素格中的点云数目 / 最大像素格数目
    density[unique_coordinates[:, 1], unique_coordinates[:, 0]] = counts
    density = density / np.max(density)
    return density, normalization_dict


def scale_annos_by_augment_pc_scale(anno_path,scale_table:dict):
    with open(anno_path, 'r') as f:
        annotation_json = json.load(f)
    scene_name = anno_path.split("\\")[-2]
    scene_scale = scale_table[scene_name]
    for line in annotation_json["lines"]:
        point = line["point"]
        point = [coord / scene_scale for coord in point]
        line["point"] = point
    for junction in annotation_json['junctions']:
        point = junction["coordinate"]
        point = [coord / scene_scale for coord in point]
        junction["coordinate"] = point

    rooms = parse_floor_plan_polys(annotation_json)
    junctions = np.array([junc['coordinate'][:3] for junc in annotation_json['junctions']])
    pure_polygons = [poly[0] for poly in rooms]
    polygons_coords = [junctions[poly] for poly in pure_polygons]
    offset = offset_polygons(polygons_coords)

    for line in annotation_json["lines"]:
        point = line["point"]
        point = [point[i] - offset[i] for i in range(3)]
        line["point"] = point
    for junction in annotation_json['junctions']:
        point = junction["coordinate"]
        point = [point[i] - offset[i] for i in range(3)]
        junction["coordinate"] = point

    return annotation_json


def read_scale_table(scale_table_path):
    scale_table = {}
    with open(scale_table_path, 'r') as f:
        for line in f:
            scene_name, scene_scale = line.strip().split(',')
            scale_table[scene_name] = float(scene_scale[:-1])
    return scale_table


def parse_floor_plan_polys(annos):
    planes = []
    for semantic in annos['semantics']:
        for planeID in semantic['planeID']:
            if annos['planes'][planeID]['type'] == 'floor':
                planes.append({'planeID': planeID, 'type': semantic['type']})

    # construct each polygon
    polygons = []
    for plane in planes:
        lineIDs = np.where(np.array(annos['planeLineMatrix'][plane['planeID']]))[0].tolist()
        junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
        polygon = convert_lines_to_vertices(junction_pairs)
        if plane['type'] in['door', 'window']:
            continue
        polygons.append([polygon[0], plane['type']])

    return polygons


def offset_polygons(polygons):
    min_coords = [10000, 10000, 0]
    for polygon in polygons:
        for vertex in polygon:
            min_coords = [min(min_coords[0], vertex[0]),
                          min(min_coords[1],vertex[1]),
                          min_coords[2]]
    return min_coords


def convert_lines_to_vertices(lines):
    """
    convert line representation to polygon vertices

    """
    polygons = []
    lines = np.array(lines)

    polygon = None
    while len(lines) != 0:
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, 0)

        lineID, juncID = np.where(lines == polygon[-1])
        vertex = lines[lineID[0], 1 - juncID[0]]
        lines = np.delete(lines, lineID, 0)

        if vertex in polygon:
            polygons.append(polygon)
            polygon = None
        else:
            polygon.append(vertex)

    return polygons


def normalize_annotations(annotation_json, normalization_dict):
    for line in annotation_json["lines"]:
        point = line["point"]
        point = normalize_point(point, normalization_dict)
        line["point"] = point

    for junction in annotation_json["junctions"]:
        point = junction["coordinate"]
        point = normalize_point(point, normalization_dict)
        junction["coordinate"] = point

    return annotation_json


def normalize_point(point, normalization_dict):

    min_coords = normalization_dict["min_coords"]
    max_coords = normalization_dict["max_coords"]
    image_res = normalization_dict["image_res"]

    point_2d = \
        np.round(
            (point[:2] - min_coords[:2]) / (max_coords[:2] - min_coords[:2]) * image_res)
    point_2d = np.minimum(np.maximum(point_2d, np.zeros_like(image_res)),
                            image_res - 1)

    point[:2] = point_2d.tolist()

    return point


def export_density(density_map, out_folder, scene_id):
    density_path = os.path.join(out_folder, scene_id+'.png')
    density_uint8 = (density_map * 255).astype(np.uint8)
    cv2.imwrite(density_path, density_uint8)


def export_flip_density(density_map, out_folder, scene_id):
    density_path = os.path.join(out_folder, scene_id + '_flip.png')
    density_uint8 = (density_map * 255).astype(np.uint8)
    inverted_density_uint8 = 255 - density_uint8
    cv2.imwrite(density_path, inverted_density_uint8)


'''
由于s3d数据集中前人一些工作生成的预测平面图是密度图坐标形式的，因此，我们需要一些函数获取相关点云的真实边界范围，并利用另一些函数将前人工作的预测平面图转化为真实坐标的obj文件。
'''

class OBJReader:
    def __init__(self, file_path = None):
        # 存储解析后的数据
        self.vertices = []    # 顶点坐标 [x, y, z]
        self.faces = []       # 面数据 [[v_idx, vt_idx, vn_idx], ...]
        if file_path is not None:
            self.read(file_path)

    def read(self, file_path):
        """读取OBJ文件"""
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()
                    keyword = parts[0]
                    values = parts[1:]

                    if keyword == 'v':
                        # 顶点坐标
                        self.vertices.append([float(v) for v in values[:3]])
                    elif keyword == 'f':
                        # 面数据
                        face = []
                        for vertex in values:
                            indices = vertex.split('/')
                            # 注意：OBJ索引从1开始，转换为Python的0-based索引
                            v_idx = int(indices[0]) - 1 if indices[0] else -1
                            vt_idx = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else -1
                            vn_idx = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else -1
                            face.append((v_idx, vt_idx, vn_idx))
                        self.faces.append(face)

            print(f"成功读取OBJ文件: {file_path}")
            return True
        except Exception as e:
            print(f"读取OBJ文件时出错: {e}")
            return False

    def get_face_vertices(self, face_index):
        """返回指定面的顶点坐标"""
        if 0 <= face_index < len(self.faces):
            face = self.faces[face_index]
            return [self.vertices[vertex[0]] for vertex in face if vertex[0] != -1]
        return []

class OBJExporter:
    """将多边形顶点数据导出为OBJ文件的工具类"""

    def __init__(self):
        self.vertices = []  # 存储所有顶点坐标
        self.vertex_index = 1  # 当前顶点索引，从1开始
        self.faces = []  # 存储所有面的定义

    def add_polygon(self, points, material_name=None):
        """
        添加一个多边形

        参数:
        points: 多边形顶点列表，每个顶点是一个三元组(x, y, z)
        material_name: 可选，材质名称
        """
        # 添加顶点
        face_indices = []
        for point in points:
            self.vertices.append(point)
            face_indices.append(self.vertex_index)
            self.vertex_index += 1

        # 添加面定义
        face_data = {
            "indices": face_indices,
            "material": material_name
        }
        self.faces.append(face_data)

    def export(self, obj_file_path):
        """
        导出为OBJ文件

        参数:
        obj_file_path: OBJ文件路径
        mtl_file_path: 可选，MTL文件路径，默认与OBJ同名
        """

        try:
            # 写入OBJ文件
            with open(obj_file_path, 'w') as f:
                # 写入顶点数据
                for vertex in self.vertices:
                    f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')

                # 写入面数据
                current_material = None
                for face in self.faces:
                    # 写入面的顶点索引
                    indices_str = ' '.join([str(i) for i in face["indices"]])
                    f.write(f'f {indices_str}\n')

            print(f"成功导出OBJ文件: {obj_file_path}")

        except Exception as e:
            print(f"导出文件时出错: {e}")


def s3d_bound_shp_to_obj(bound_shp_path, out_folder, scene_id):
    import shapefile
    sf = shapefile.Reader(bound_shp_path)
    shapes = sf.shapes()
    records = sf.records()
    exporter = OBJExporter()
    for i in range(len(shapes)):
        polygon = shapes[i].points
        polygon_obj = []
        for i in range(len(polygon) - 1):
            point = polygon[i]
            polygon_obj.append([point[0], point[1], 0])
        exporter.add_polygon(polygon_obj)

    exporter.export(os.path.join(out_folder, 'rooms_gt.obj'))


def generate_real_world_room_shp_x_y_range(vertex_shp):
    import shapefile
    sf = shapefile.Reader(vertex_shp)
    shapes = sf.shapes()
    points = []
    for i in range(len(shapes)):
        point = shapes[i].points[0]
        points.append(point)
    points = np.array(points)
    x_range = np.max(points[:, 0]) - np.min(points[:, 0])
    y_range = np.max(points[:, 1]) - np.min(points[:, 1])
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    return [x_range, y_range, x_min, y_min]


def generate_dataset_real_world_range_record_from_point_cloud(point_cloud_root,record_path = None):
    scene_names = os.listdir(point_cloud_root)
    scene_names.sort()
    record = {}
    bar = tqdm.tqdm(total=len(scene_names))
    import laspy
    for scene in scene_names:
        if not scene.startswith('scene_'):
            continue
        scene_path = os.path.join(point_cloud_root, scene)
        point_cloud_path = os.path.join(scene_path, 'point_cloud.laz')
        pc = laspy.read(point_cloud_path)
        point_cloud = np.vstack((pc.x, pc.y, pc.z)).T



        x_range = np.max(point_cloud[:, 0]) - np.min(point_cloud[:, 0])
        y_range = np.max(point_cloud[:, 1]) - np.min(point_cloud[:, 1])
        x_min = np.min(point_cloud[:, 0])
        y_min = np.min(point_cloud[:, 1])
        x_max = np.max(point_cloud[:, 0])
        y_max = np.max(point_cloud[:, 1])

        x_min = x_min - 0.1 * x_range
        y_min = y_min - 0.1 * y_range
        x_max = x_max + 0.1 * x_range
        y_max = y_max + 0.1 * y_range
        x_range = x_max - x_min
        y_range = y_max - y_min

        record[scene] = [x_range, y_range, x_min, y_min]
        bar.update(1)
    bar.close()

    record_path = os.path.join(record_path, 'dataset_real_world_range.json')
    import json
    with open(record_path, 'w') as f:
        json.dump(record, f)

def generate_dataset_real_world_range_record(shp_root,record_path = None):
    scene_names = os.listdir(shp_root)
    scene_names.sort()
    record = {}
    bar = tqdm.tqdm(total=len(scene_names))
    for scene in scene_names:
        if not scene.startswith('scene_'):
            continue
        scene_path = os.path.join(shp_root, scene)
        vertex_shp = os.path.join(scene_path, 'vertexes.shp')
        record[scene] = generate_real_world_room_shp_x_y_range(vertex_shp)
        bar.update(1)
    bar.close()
    import json
    if record_path is None:
        parent_path = os.path.dirname(shp_root)
        record_path = os.path.join(parent_path, 'dataset_real_world_range.json')
    with open(record_path, 'w') as f:
        json.dump(record, f)


def from_density_map_result_to_real_world(density_map_rooms,x_range,y_range,x_min,y_min):
    x_step = x_range / 256
    y_step = y_range / 256

    x_discrete_value = [i * x_step + x_step/2 for i in range(256)]
    y_discrete_value = [i * y_step + y_step/2 for i in range(256)]
    rooms = []
    for r in density_map_rooms:
        rooms.append([
            [x_discrete_value[p[0]],y_discrete_value[p[1]]]
            for p in r
        ])

    for r in rooms:
        for p in r:
            p[0] += x_min
            p[1] += y_min
    return rooms


def to_density_map_result_from_real_world(rooms,x_range,y_range,x_min,y_min):
    density_map_rooms = []
    for r in rooms:
        room = []
        for p in r:
            p[0] -= x_min
            p[1] -= y_min
            x_idx = int(p[0] / x_range * 256)
            y_idx = int(p[1] / y_range * 256)
            room.append([x_idx, y_idx])
        density_map_rooms.append(room)
    return density_map_rooms


def from_density_map_result_to_real_world_and_save_in_obj(
        density_map_rooms,
        x_range,
        y_range,
        x_min,
        y_min,
        obj_path
):
    rooms = from_density_map_result_to_real_world(density_map_rooms, x_range, y_range, x_min, y_min)
    exporter = OBJExporter()
    for i in range(len(rooms)):
        polygon = rooms[i]
        polygon_obj = []
        for i in range(len(polygon)):
            point = polygon[i]
            polygon_obj.append([point[0], point[1], 0])
        exporter.add_polygon(polygon_obj)

    exporter.export(obj_path)
    return rooms


if __name__ == '__main__':
    # test_scene = r"...\augment_stru3d_pointcloud\scene_00002"
    test_scene = r""
    # generate_augmented_point_cloud_density_map(test_scene, None)
    # anno_path = r"...\annotation_3d.json"
    anno_path = r""
    # scale_table_path = r"...\scales.txt"
    scale_table_path = r""
    scale_table = read_scale_table(scale_table_path)
    anno_json = scale_annos_by_augment_pc_scale(anno_path,scale_table)
    polygons = parse_floor_plan_polys(anno_json)
    junctions = np.array([junc['coordinate'][:3] for junc in anno_json['junctions']])
    pure_polygons = [poly[0] for poly in polygons]
    adapt_polygons = []
    for poly in pure_polygons:
        adapt_polygon = poly.copy()
        adapt_polygon.append(poly[0])
        adapt_polygons.append(adapt_polygon)
    polygons = [junctions[poly] for poly in adapt_polygons]
    export_face_to_obj(polygons, os.path.join(test_scene, 'test.obj'))