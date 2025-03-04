import math
import queue
from collections import defaultdict

import trimesh

from util.s3d_data_load import enum_label

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

    def start_work(self):
        ajacent_graph = self.generate_adjacent_graph()
        pass

    def generate_adjacent_graph(self):
        polygon_id_around_point_table = defaultdict(set)
        for i in range(len(self._faces)):
            triangle = self._faces[i]
            polygon_id_around_point_table[triangle[0]].add(i)
            polygon_id_around_point_table[triangle[1]].add(i)
            polygon_id_around_point_table[triangle[2]].add(i)

        face_neighborhood = trimesh.Trimesh(vertices=self._points, faces=self._faces, process=False).face_neighborhood

        adjacent_graph = [set() for _ in range(len(self._faces))]
        for i in range(len(self._faces)):
            for j in face_neighborhood[i]:
                adjacent_graph[i].add(j)
                adjacent_graph[j].add(i)
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
            if polygon_label == enum_label.in_wall.value or polygon_label == enum_label.ceiling.value:
                continue
            q = queue.Queue()
            split_polygons = {polygon_id}
            while not q.empty():
                cur_polygon_id = q.get()
                visited[cur_polygon_id] = True
                for neighbor_id in graph[cur_polygon_id]:
                    if not visited[neighbor_id]:
                        if self._labels[neighbor_id] == polygon_label:
                            split_polygons.add(neighbor_id)
                            q.put(neighbor_id)
            if len(split_polygons) > 1:
                continue
            groups.append(split_polygons)
        return groups

    def merge_mesh(self,groups):
        merged_polygons = []
        for group in groups:
            pass

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
                return []
            if len(neib_ids) == 4:
                degenerate_point = point_id
                sorted_neib_ids = self.sort_neib_clockwise(degenerate_point, neib_ids)
                degenerate_neib = []

                for edge, count in edge_count.items():
                    if edge[0] == degenerate_point or edge[1] == degenerate_point:
                        if edge[0] != degenerate_point:
                            degenerate_neib.append(edge[1])
                        else:
                            degenerate_neib.append(edge[0])

                degenerate_neib_num = len(degenerate_neib)
                degenerate_neib_index = defaultdict(int)
                for i in range(degenerate_neib_num):
                    degenerate_neib_index[degenerate_neib[i]] = i

                neib_graph_without_target_point = [[0 for _ in range(degenerate_neib_num)]
                                                   for _ in range(degenerate_neib_num)]
                for edge, count in edge_count.items():
                    if edge[0] in degenerate_neib_index and edge[1] in degenerate_neib_index:
                        neib_graph_without_target_point[degenerate_neib_index[edge[0]]][degenerate_neib_index[edge[1]]] = 1
                        neib_graph_without_target_point[degenerate_neib_index[edge[1]]][degenerate_neib_index[edge[0]]] = 1

                v0 = degenerate_neib_index[neib_ids[0]]
                v1 = degenerate_neib_index[neib_ids[1]]
                v2 = degenerate_neib_index[neib_ids[2]]
                v3 = degenerate_neib_index[neib_ids[3]]

                to_process_info = []
                to_process_info.append(degenerate_point)
                if is_connected(v0, v1, neib_graph_without_target_point) and\
                   is_connected(v1, v2, neib_graph_without_target_point):
                    to_process_info.append(neib_ids[0])
                    to_process_info.append(neib_ids[3])
                    to_process_info.append(neib_ids[1])
                    to_process_info.append(neib_ids[2])
                elif is_connected(v0, v3, neib_graph_without_target_point) and\
                     is_connected(v1, v2, neib_graph_without_target_point):
                    to_process_info.append(neib_ids[0])
                    to_process_info.append(neib_ids[1])
                    to_process_info.append(neib_ids[2])
                    to_process_info.append(neib_ids[3])

                to_process_infos.append(to_process_info)

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
                pre_point = tmp
                if current_point == pre_point:
                    break
            polygons.append(polygon)

        chosen_polygon = -1
        chosen_polygon_size = 0
        for i in range(len(polygons)):
            if len(polygons[i]) == 3:





