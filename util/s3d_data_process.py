import json

import cv2
import numpy as np
import os

from util.visualization import export_face_to_obj


def process_vertice_by_op_record(reverse_op,vertices):
    for op in reverse_op:
        if op[0] == '+':
            vertices = np.stack([vertices[:, 0] + op[1][0], vertices[:, 1] + op[1][1], vertices[:, 2]], axis=-1)
        elif op[0] == '*':
            vertices = np.stack([vertices[:, 0] * op[1][0], vertices[:, 1] * op[1][1], vertices[:, 2]], axis=-1)

    return vertices


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


if __name__ == '__main__':
    test_scene = r"G:\workspace_plane2DDL\real_point_cloud_dataset\augment_stru3d_pointcloud\scene_00002"
    # generate_augmented_point_cloud_density_map(test_scene, None)
    anno_path = r"G:\workspace_plane2DDL\data_anno\scene_00002\annotation_3d.json"
    scale_table_path = r"G:\workspace_plane2DDL\data_anno_scales\scales.txt"
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