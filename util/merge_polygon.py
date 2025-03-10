import math
import os
import queue
from collections import defaultdict
from datetime import datetime

import numpy as np
import trimesh
from shapely import MultiPoint, Polygon, unary_union, MultiPolygon

from srcipt.generate_coco_stru3d import generate_coco_dict, generate_predict_coco_dict
from util.s3d_data_load import enum_label, get_vertices_coord, get_faces_point_id_and_label
from util.visualization import visualization_seg_with_custom_anno


def is_connected(src, tgt, graph):
    q = queue.Queue()
    visited = [False] * len(graph)
    q.put(src)
    visited[src] = True
    while not q.empty():
        cur_id = q.get()
        if cur_id == tgt:
            return True
        for i in range(len(graph)):
            if not visited[graph[cur_id][i]] and graph[cur_id][i] == 1:
                q.put(i)
                visited[i] = True
    return False


def replace_target_point(neib:list,replace_one:int,target_point:int):
    for i in range(len(neib)):
        if neib[i] == target_point:
            neib[i] = replace_one
            break


class MergePolygonSolution:
    def __init__(self):
        self._points = []
        self._faces = []
        self._labels = []
        self._adjacent_graph = []
        self._merged_polygons = []

    def load_data_from_shp_file(self,shp_file_root):
        points = get_vertices_coord(os.path.join(shp_file_root, 'vertexes.shp'))
        faces, labels = get_faces_point_id_and_label(os.path.join(shp_file_root, 'poly.shp'))
        self.load_data(points, faces, labels)

    def load_data_from_model_inference(self,vertices,faces,labels):
        self._points = [(v[0],v[1]) for v in vertices]
        self._faces = [(f[0],f[1],f[2]) for f in faces]
        self._labels = [int(l) for l in labels]


    def load_data(self,points,faces,labels):
        self._points = points
        self._faces = faces
        self._labels = labels

    def start_work_2(self):
        ajacent_graph = self.generate_adjacent_graph()
        groups = self.split_mesh(ajacent_graph)
        self.merge_mesh(groups)
        self.simplify_mesh()

    def start_work(self):
        ajacent_graph = self.generate_adjacent_graph()
        groups = self.split_mesh(ajacent_graph)
        self.merge_mesh_with_shapely(groups)
        self.simplify_mesh_2()

    def start_work_with_time_analysis(self):
        start_time = datetime.now()
        ajacent_graph = self.generate_adjacent_graph()
        end_time_1 = datetime.now()
        print("generate_adjacent_graph time:", (end_time_1 - start_time))
        start_time = datetime.now()
        groups = self.split_mesh(ajacent_graph)
        end_time_2 = datetime.now()
        print("split_mesh time:", (end_time_2 - start_time).total_seconds())
        start_time = datetime.now()
        self.merge_mesh(groups)
        end_time_3 = datetime.now()
        print("merge_mesh time:", (end_time_3 - start_time).total_seconds())
        start_time = datetime.now()
        self.simplify_mesh()
        end_time_4 = datetime.now()
        print("simplify_mesh time:", (end_time_4 - start_time).total_seconds())

    def get_merged_polygons_with_coords(self):
        return self._merged_polygons

    def get_merged_polygons_as_coco_format(self,cur_img_id):
        multi_point = MultiPoint(self._points)
        bbox = multi_point.bounds
        rooms = self.get_merged_polygons_with_coords()
        coco_dict = generate_predict_coco_dict(rooms, bbox, cur_img_id)
        return coco_dict

    def get_merged_polygons_as_raster_format(self):
        mutli_point = MultiPoint(self._points)
        bbox = mutli_point.bounds
        rooms = self.get_merged_polygons_with_coords()

        min_coord = np.array([bbox[0], bbox[1]])
        max_coord = np.array([bbox[2], bbox[3]])

        norm_rooms = []
        img_res = np.array((256, 256))
        for room in rooms:
            norm_room = []
            for point in room:
                norm_point = np.round((point - min_coord) / (max_coord - min_coord) * img_res)
                norm_room.append(norm_point)
            norm_rooms.append(np.array(norm_room).astype(np.int32))
        return norm_rooms

    def generate_adjacent_graph(self):
        polygon_id_around_point_table = defaultdict(set)
        for i in range(len(self._faces)):
            triangle = self._faces[i]
            polygon_id_around_point_table[triangle[0]].add(i)
            polygon_id_around_point_table[triangle[1]].add(i)
            polygon_id_around_point_table[triangle[2]].add(i)

        mesh = trimesh.Trimesh(
            vertices=self._points, faces=self._faces, process=False)


        def get_shared_edge_neighbors(mesh, face_index):
            """
            获取与指定面共享边的相邻面。
            :param mesh: trimesh.Trimesh 对象
            :param face_index: 指定的面索引
            :return: 共享边的相邻面索引列表
            """
            # 获取指定面的边
            target_face = self._faces[face_index]
            target_edges = [
                [target_face[0],target_face[1]],
                [target_face[1],target_face[2]],
                [target_face[2],target_face[0]]]
            # 用于存储共享边的相邻面
            neighbors = set()

            # 遍历所有面
            for i, face in enumerate(mesh.faces):
                if i == face_index:
                    continue  # 跳过自身
                # 获取当前面的边
                current_faces = mesh.faces[i]
                current_edges = [
                    [current_faces[0],current_faces[1]],
                    [current_faces[1],current_faces[2]],
                    [current_faces[2],current_faces[0]]]

                find = False
                for edge in target_edges:
                    if not find:
                        for current_edge in current_edges:
                            if sorted(edge) == sorted(current_edge):
                                neighbors.add(i)
                                find = True
                                break
                    else:
                        break

            return list(neighbors)

        adjacent_graph = [set() for _ in range(len(self._faces))]
        for i in range(len(self._faces)):
            neib = get_shared_edge_neighbors(mesh, i)
            for j in range(len(neib)):
                adjacent_graph[i].add(neib[j])
        return adjacent_graph

    def generate_around_point_adjacent_graph(self):

        polygon_id_around_point_table = defaultdict(set)
        for i in range(len(self._faces)):
            triangle = self._faces[i]
            polygon_id_around_point_table[triangle[0]].add(i)
            polygon_id_around_point_table[triangle[1]].add(i)
            polygon_id_around_point_table[triangle[2]].add(i)

        adjacent_graph = [set() for _ in range(len(self._faces))]
        for i in range(len(self._faces)):
            for polygon_id in polygon_id_around_point_table[self._faces[i][0]]:
                adjacent_graph[i].add(polygon_id)
            for polygon_id in polygon_id_around_point_table[self._faces[i][1]]:
                adjacent_graph[i].add(polygon_id)
            for polygon_id in polygon_id_around_point_table[self._faces[i][2]]:
                adjacent_graph[i].add(polygon_id)
            adjacent_graph[i].remove(i)
        return adjacent_graph

    def split_mesh(self,graph):
        visited = [False] * len(self._faces)
        groups = []
        # bfs
        for polygon_id in range(len(self._faces)):
            if visited[polygon_id]:
                continue

            polygon_label = self._labels[polygon_id]
            if polygon_label == enum_label.in_wall.value or polygon_label == enum_label.out_wall.value:
                continue
            q = queue.Queue()
            split_polygons = {polygon_id}
            visited[polygon_id] = True
            q.put(polygon_id)
            while not q.empty():
                cur_polygon_id = q.get()
                for neighbor_id in graph[cur_polygon_id]:
                    if not visited[neighbor_id]:
                        if self._labels[neighbor_id] == polygon_label:
                            split_polygons.add(neighbor_id)
                            q.put(neighbor_id)
                            visited[neighbor_id] = True
            if len(split_polygons) == 1:
                continue
            groups.append(split_polygons)
        return groups

    def merge_mesh(self,groups):
        merged_polygons = []
        for group in groups:
            self._merged_polygons.append(self.reconstruct_mesh(group))

    def merge_mesh_with_shapely(self,groups):
        for group in groups:
            self._merged_polygons.append(self.reconstruct_mesh_depend_shapely(group))

    def sort_neib_clockwise(self, center_id, neib_ids):
        center_point = self._points[center_id]
        points_with_angles = []
        for neib_id in neib_ids:
            neib_point = self._points[neib_id]
            angle = math.atan2((neib_point[1] - center_point[1]),
                               (neib_point[0] - center_point[0]))
            points_with_angles.append((neib_id, angle))

        points_with_angles.sort(key=lambda x: x[1])
        sorted_neib_ids = [x[0] for x in points_with_angles]
        neib_ids = sorted_neib_ids
        return neib_ids

    def reconstruct_mesh_depend_shapely(self,group):
        polygons = []
        for face_id in group:
            face = self._faces[face_id]
            polygon = Polygon([self._points[point_id] for point_id in face])
            polygons.append(polygon)
        union = unary_union(polygons)
        # 提取外轮廓
        if isinstance(union, MultiPolygon):
            # 如果有多个独立的外轮廓
            exterior_coords = [list(poly.exterior.coords) for poly in union.geoms]
        else:
            # 如果只有一个外轮廓
            exterior_coords = [list(union.exterior.coords)]
        union_areas = [Polygon(coords).area for coords in exterior_coords]
        max_area_id = union_areas.index(max(union_areas))
        exterior_coords = exterior_coords[max_area_id]
        exterior_coords.pop()
        return exterior_coords

    def reconstruct_mesh(self,group):
        edge_count = defaultdict(int)

        def insert(p1_id, p2_id, edge_count):
            v1 = self._points[p1_id]
            v2 = self._points[p2_id]
            if v1[0] > v2[0] or (v1[0] == v2[0] and v1[1] > v2[1]):
                p1_id, p2_id = p2_id, p1_id
            edge_count[(p1_id, p2_id)] += 1

        for face_id in group:
            face = self._faces[face_id]
            insert(face[0], face[1], edge_count)
            insert(face[1], face[2], edge_count)
            insert(face[2], face[0], edge_count)

        bound_points_neib = defaultdict(list)
        bound_edges = []
        for edge, count in edge_count.items():
            if count == 1:
                bound_points_neib[edge[0]].append(edge[1])
                bound_points_neib[edge[1]].append(edge[0])
                bound_edges.append(edge)

        to_process_infos = []

        for point_id, neib_ids in bound_points_neib.items():
            if len(neib_ids) != 2 and len(neib_ids) != 4:
                raise ValueError("The mesh is not a 2-manifold")
            if len(neib_ids) == 4:
                raise ValueError("The mesh is not a 2-manifold")
                # degenerate_point = point_id
                # sorted_neib_ids = self.sort_neib_clockwise(degenerate_point, neib_ids)
                # degenerate_neib = []
                #
                # for edge, count in edge_count.items():
                #     if edge[0] == degenerate_point or edge[1] == degenerate_point:
                #         if edge[0] != degenerate_point:
                #             degenerate_neib.append(edge[1])
                #         else:
                #             degenerate_neib.append(edge[0])
                #
                # degenerate_neib_num = len(degenerate_neib)
                # degenerate_neib_index = defaultdict(int)
                # for i in range(degenerate_neib_num):
                #     degenerate_neib_index[degenerate_neib[i]] = i
                #
                # neib_graph_without_target_point = [[0 for _ in range(degenerate_neib_num)]
                #                                    for _ in range(degenerate_neib_num)]
                # for edge, count in edge_count.items():
                #     if edge[0] in degenerate_neib_index and edge[1] in degenerate_neib_index:
                #         neib_graph_without_target_point[degenerate_neib_index[edge[0]]][degenerate_neib_index[edge[1]]] = 1
                #         neib_graph_without_target_point[degenerate_neib_index[edge[1]]][degenerate_neib_index[edge[0]]] = 1
                #
                # v0 = degenerate_neib_index[neib_ids[0]]
                # v1 = degenerate_neib_index[neib_ids[1]]
                # v2 = degenerate_neib_index[neib_ids[2]]
                # v3 = degenerate_neib_index[neib_ids[3]]
                #
                # to_process_info = []
                # to_process_info.append(degenerate_point)
                # if is_connected(v0, v1, neib_graph_without_target_point) and\
                #    is_connected(v1, v2, neib_graph_without_target_point):
                #     to_process_info.append(neib_ids[0])
                #     to_process_info.append(neib_ids[3])
                #     to_process_info.append(neib_ids[1])
                #     to_process_info.append(neib_ids[2])
                # elif is_connected(v0, v3, neib_graph_without_target_point) and\
                #      is_connected(v1, v2, neib_graph_without_target_point):
                #     to_process_info.append(neib_ids[0])
                #     to_process_info.append(neib_ids[1])
                #     to_process_info.append(neib_ids[2])
                #     to_process_info.append(neib_ids[3])
                #
                # to_process_infos.append(to_process_info)

        for to_process_info in to_process_infos:
            target_point = to_process_info[0]
            edge0_point0 = to_process_info[1]
            edge0_point1 = to_process_info[2]
            edge1_point0 = to_process_info[3]
            edge1_point1 = to_process_info[4]

            # bound_edges 操作：去除和degen_point相关的4条边，增加两条边;
            # bound_points_neib 操作：去除degen_point的ele,修改4个顶点的相关的ele
            bound_edges = [edge for edge in bound_edges if edge[0] != target_point and edge[1] != target_point]
            bound_edges.append((edge0_point0, edge0_point1))
            bound_edges.append((edge1_point0, edge1_point1))
            bound_points_neib.pop(target_point)
            replace_target_point(bound_points_neib[edge0_point0], replace_one=edge0_point1, target_point=target_point)
            replace_target_point(bound_points_neib[edge0_point1], replace_one=edge0_point0, target_point=target_point)
            replace_target_point(bound_points_neib[edge1_point0], replace_one=edge0_point1, target_point=target_point)
            replace_target_point(bound_points_neib[edge1_point1], replace_one=edge0_point1, target_point=target_point)

        # 3.reconstruct triangles
        polygons = []
        bound_points_visited = set()
        for edge in bound_edges:
            start_point = edge[0]
            if start_point in bound_points_visited:
                continue
            current_point = start_point
            pre_point = edge[1]
            polygon = []
            while True:
                polygon.append(current_point)
                tmp = current_point
                bound_points_visited.add(current_point)

                if bound_points_neib[current_point][0] == pre_point:
                    current_point = bound_points_neib[current_point][1]
                else:
                    current_point = bound_points_neib[current_point][0]
                if current_point == start_point:
                    break
                pre_point = tmp
            polygons.append(polygon)

        chosen_polygon = -1
        chosen_polygon_size = 0
        for i in range(len(polygons)):
            if len(polygons[i]) > chosen_polygon_size:
                chosen_polygon = i
                chosen_polygon_size = len(polygons[i])

        return polygons[chosen_polygon]

    def simplify_mesh(self):
        def calculate_vector_angle(p1, p2):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            return math.atan2(dy, dx)

        def is_colinear(pre, check_one, post):
            pre_point = self._points[pre]
            check_point = self._points[check_one]
            post_point = self._points[post]
            slope1 = calculate_vector_angle(pre_point, check_point)
            slope2 = calculate_vector_angle(check_point, post_point)
            return abs(slope1 - slope2) < 0.0001

        for polygon_id, polygon in enumerate(self._merged_polygons):
            simplified_polygon = []
            for i in range(len(polygon)):
                if i == 0:
                    pre = len(polygon) - 1
                else:
                    pre = i - 1
                post = (i + 1) % len(polygon)
                if not is_colinear(polygon[pre], polygon[i], polygon[post]):
                    simplified_polygon.append(polygon[i])

            self._merged_polygons[polygon_id] = simplified_polygon

    def simplify_mesh_2(self):
        def calculate_vector_angle(p1, p2):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            return math.atan2(dy, dx)

        def is_colinear(pre, check_one, post):
            slope1 = calculate_vector_angle(pre, check_one)
            slope2 = calculate_vector_angle(check_one, post)
            return abs(slope1 - slope2) < 0.0001

        for polygon_id, polygon in enumerate(self._merged_polygons):
            simplified_polygon = []
            for i in range(len(polygon)):
                if i == 0:
                    pre = len(polygon) - 1
                else:
                    pre = i - 1
                post = (i + 1) % len(polygon)
                if not is_colinear(polygon[pre], polygon[i], polygon[post]):
                    simplified_polygon.append(polygon[i])

            self._merged_polygons[polygon_id] = simplified_polygon


def test_merge_polygon_with_mock_data():
    points = [[0, 0], [2, 2], [0, 2], [-2, 2], [-2, 0],[0,-2],[2,-2],[2,0]]
    faces = [[0, 3, 4], [0, 2, 3], [0, 1, 2], [0, 1, 7],[0,6,7],[0,5,6]]
    labels = [enum_label.balcony.value] * 6
    solution = MergePolygonSolution()
    solution.load_data(points, faces, labels)
    solution.start_work()
    rooms = solution.get_merged_polygons_with_coords()

def test_merge_polygon():
    start_time = datetime.now()
    shp_file_root = r'G:\workspace_plane2DDL\testData\10_percent_box\scene_00020'
    solution = MergePolygonSolution()
    solution.load_data_from_shp_file(shp_file_root)
    solution.start_work()
    coco_dict = solution.get_merged_polygons_as_coco_format(20)

    # density_folder = r'G:\workspace_plane2DDL\augment_point_cloud_density'
    # img_folder = os.path.join(density_folder, 'train')
    # annotation_json_path = os.path.join(density_folder, 'annotations', 'train.json')
    #
    # visualization_seg_with_custom_anno(20,img_path=img_folder,json_path=annotation_json_path,coco_anno_dict=coco_dict)

if __name__ == '__main__':
    test_merge_polygon()