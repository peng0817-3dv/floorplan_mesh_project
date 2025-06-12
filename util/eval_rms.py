import os

import numpy as np
import laspy
from objloader import Obj
from tqdm import tqdm

'''
this file is used to evaluate the rms eval value of the ground truth las file with the predict floor plan data.
'''

class RMS_Evaluator:
    def __init__(self,
                 las_path = None,
                 floor_plan_path = None,
                 floor_plan_data = None,
                 output_las_result_path = None):
        self.las = None
        self.walls = None
        self.wall_segments = None
        self.las_save_path = None

        if las_path is not None:
            # 确定输出文件名
            if output_las_result_path is None:
                base_name, ext = os.path.splitext(las_path)
                self.las_save_path = f"{base_name}_wall{ext}"
            else:
                self.las_save_path = output_las_result_path

            in_file = laspy.read(las_path)

            if not all(hasattr(in_file.points, dim) for dim in ["red", "green", "blue"]):

                header = laspy.LasHeader(point_format=3, version="1.2")
                self.las = laspy.LasData(header=header)
                self.las.x = in_file.x
                self.las.y = in_file.y
                self.las.z = in_file.z
            else:
                # 如果原始点格式已支持RGB，则直接使用
                self.las = in_file

        method_type = '3d_walls_points'
        if floor_plan_data is not None:
            if method_type == '3d_walls':
                self.walls = self._convert_2d_floor_plan_data_to_3d_walls(floor_plan_data)
            elif method_type == '3d_walls_points':
                self.walls = self._convert_2d_floor_plan_data_to_3d_walls_points(floor_plan_data)
            else:
                self.walls = None
            self.wall_segments = self._convert_2d_floor_plan_data_to_2d_wall_segment_lines(floor_plan_data)
        elif floor_plan_path is not None:
            self.walls = self._convert_2d_floor_plan_obj_to_data_by_method_type(floor_plan_path,method_type)
            self.wall_segments = self._convert_2d_floor_plan_obj_to_data_by_method_type(floor_plan_path,'2d_wall_segment_lines')



    def _convert_2d_floor_plan_data_to_2d_wall_segment_lines(self,data):
        rooms_vertices = data
        lines = []
        for room_vertices in rooms_vertices:
            for i in range(len(room_vertices)):
                v1 = room_vertices[i]
                v2 = room_vertices[(i+1) % len(room_vertices)]
                lines.append({'start':[v1[0],v1[1]],'end':[v2[0],v2[1]]})
        return lines


    def _convert_2d_floor_plan_obj_to_data_by_method_type(self,floor_plan_path,method_type):
        data = read_obj_file(floor_plan_path)
        rooms_vertices = [[data['vertices'][v] for v in face] for face in data['faces']]
        if method_type == '2d_wall_segment_lines':
            return self._convert_2d_floor_plan_data_to_2d_wall_segment_lines(rooms_vertices)
        elif method_type == '3d_walls':
            return self._convert_2d_floor_plan_data_to_3d_walls(rooms_vertices)
        elif method_type == '3d_walls_points':
            return self._convert_2d_floor_plan_data_to_3d_walls_points(rooms_vertices)
        else:
            return None

    def _convert_2d_floor_plan_data_to_3d_walls(self,data):
        rooms_vertices = data
        lines = []
        for room_vertices in rooms_vertices:
            for i in range(len(room_vertices)):
                v1 = room_vertices[i]
                v2 = room_vertices[(i+1) % len(room_vertices)]
                # calculate line a b c param
                a = v2[1] - v1[1]
                b = v1[0] - v2[0]
                c = v2[0]*v1[1] - v1[0]*v2[1]

                lines.append((a, b, c))
        walls = []
        for line in lines:
            # 垂直于xy平面的平面 a b c d
            walls.append((line[0], line[1], 0, line[2]))
        return walls

    def _convert_2d_floor_plan_data_to_3d_walls_points(self,data):
        rooms_vertices = data
        points = np.vstack((self.las.x, self.las.y, self.las.z)).transpose()
        z_min = np.min(points[:,2])
        z_max = np.max(points[:,2])

        walls = []
        for room_vertices in rooms_vertices:
            for i in range(len(room_vertices)):
                v1_level_plane = room_vertices[i]
                v2_level_plane = room_vertices[(i+1) % len(room_vertices)]
                p1 = [v1_level_plane[0],v1_level_plane[1],z_min]
                p2 = [v1_level_plane[0],v1_level_plane[1],z_max]
                p3 = [v2_level_plane[0],v2_level_plane[1],z_max]
                p4 = [v2_level_plane[0],v2_level_plane[1],z_min]
                walls.append([p1,p2,p3,p4])
        return walls


    def point_to_plane_distance(self, point, wall):
        start = np.array(wall['start'])
        end = np.array(wall['end'])
        p = np.array([point[0],point[1]])

        line_vector = end - start
        point_vector = p - start

        line_len_sq = np.sum(line_vector ** 2)

        # 防止零值
        epsilon = 1e-6
        if line_len_sq < epsilon:
            line_len_sq = epsilon

        t = np.clip(np.dot(point_vector, line_vector) / line_len_sq, 0, 1)
        projection = start + t * line_vector
        true_dist = np.linalg.norm(p - projection)

        vis_dist = min(0.2,true_dist)
        return true_dist

    def calculate_rms(self,
                      z_range:dict = None,
                      need_save_las = True,
                      auto_clip_z = False,
                      mode = 'slow'):
        # mode: now mode is all fast, slow is deprecated


        if auto_clip_z:
            z_min = np.min(self.las.z)
            z_max = np.max(self.las.z)
            z_max_to_min = z_max - z_min
            z_range = {
                'min':z_min + 0.1*z_max_to_min,
                'max':z_max - 0.1*z_max_to_min
            }

        calculate_point_count = 0

        points_num = len(self.las.x)
        slice_len = 500000
        slices = [[i, min(i+slice_len,points_num)-1] for i in range(0, points_num, slice_len)]
        points = np.vstack((self.las.x, self.las.y, self.las.z)).transpose()
        wall_distances = np.zeros(len(points), dtype=np.float32)
        z_mask = np.ones(len(points), dtype=bool)

        for slice in slices:
            points_slice = points[slice[0]:slice[1]]
            distances_slice,z_mask_slice = self.point_cloud_to_plane_distance(points_slice,z_range)
            calculate_point_count += z_mask_slice.sum()
            wall_distances[slice[0]:slice[1]] = distances_slice
            z_mask[slice[0]:slice[1]] = z_mask_slice


        if calculate_point_count == 0:
            avg_rms = 0
        else:
            avg_rms = np.sum(wall_distances[z_mask]) / calculate_point_count

        if need_save_las:
            rms_dimention = laspy.ExtraBytesParams(
                name="rms_value",  # 字段名
                type="float32",  # 数据类型
                description="rms value"  # 描述
            )
            self.las.add_extra_dim(rms_dimention)
            self.las.rms_value = wall_distances
            self.project_values_in_pc_color(wall_distances)

            self.las.write(self.las_save_path)
        avg_rms_txt_path = os.path.splitext(self.las_save_path)[0] + '_avg_rms.txt'
        with open(avg_rms_txt_path, 'w') as f:
            f.write(str(avg_rms))
        return avg_rms

    def delete_invalid_walls(self,walls):
        num_walls = walls.shape[0]
        # 提取墙面的四个顶点
        p1 = walls[:, 0, :]  # (M, 3)
        p2 = walls[:, 1, :]  # (M, 3)
        p3 = walls[:, 2, :]  # (M, 3)
        p4 = walls[:, 3, :]  # (M, 3)

        para_a = p3[:, 1] - p1[:, 1]
        para_b = p1[:, 0] - p3[:, 0]
        para_c = np.zeros(num_walls)
        para_d = p3[:, 0] * p1[:, 1] - p1[:, 0] * p3[:, 1]

        planes = np.vstack((para_a, para_b, para_c, para_d)).transpose()  # (M, 4)
        normals_2 = planes[:, :3]  # (M, 3)

        # normal_length = np.linalg.norm(normals_2, axis=1)  # (M,)
        epsilon = 1e-6  # 定义极小值防止除零
        # 原始计算
        normal_length = np.linalg.norm(normals_2, axis=1)  # 形状为 (M,)
        # 检测并修正为零的值
        mask = normal_length < epsilon  # 找出接近零的值
        valid_walls = walls[~mask]
        if valid_walls.shape[0] != num_walls:
            print(f"origin wall num:{num_walls},delete invalid walls:{num_walls - valid_walls.shape[0]}")
        return valid_walls

    def point_cloud_to_plane_distance(self,points,z_range = None):
        walls = np.array(self.walls)
        walls = self.delete_invalid_walls(walls)
        num_points = points.shape[0]
        num_walls = walls.shape[0]

        # 提取墙面的四个顶点
        p1 = walls[:, 0, :]  # (M, 3)
        p2 = walls[:, 1, :]  # (M, 3)
        p3 = walls[:, 2, :]  # (M, 3)
        p4 = walls[:, 3, :]  # (M, 3)

        para_a = p3[:,1] - p1[:,1]
        para_b = p1[:,0] - p3[:,0]
        para_c = np.zeros(num_walls)
        para_d = p3[:,0] * p1[:,1] - p1[:,0] * p3[:,1]

        planes = np.vstack((para_a, para_b, para_c, para_d)).transpose()  # (M, 4)
        normals_2 = planes[:, :3]  # (M, 3)

        # normal_length = np.linalg.norm(normals_2, axis=1)  # (M,)
        epsilon = 1e-6  # 定义极小值防止除零
        # 原始计算
        normal_length = np.linalg.norm(normals_2, axis=1)  # 形状为 (M,)
        # 检测并修正为零的值
        mask = normal_length < epsilon  # 找出接近零的值
        normal_length[mask] = epsilon  # 将接近零的值替换为epsilon

        unit_normal = normals_2 / normal_length[:, np.newaxis]  # (M, 3)
        offsets = planes[:, 3]  # (M,)
        d = offsets

        numerator = np.abs(np.dot(points, normals_2.T) + offsets)  # 形状为 (N, M)
        denominator = normal_length
        plane_distances = numerator / denominator[np.newaxis, :]  # 形状为 (N, M)


        # 扩展维度以实现广播
        points_expanded = points[:, np.newaxis, :]  # (N, 1, 3)
        unit_normal_expanded = unit_normal[np.newaxis, :, :]  # (1, M, 3)
        d_expanded = d[np.newaxis, :]  # (1, M)
        # 计算点在平面上的投影
        projection = points_expanded - (np.sum(points_expanded * unit_normal_expanded, axis=2) + d_expanded)[:, :,
                                       np.newaxis] * unit_normal_expanded  # (N, M, 3)

        # 构建墙面的局部坐标系
        u = p2 - p1  # (M, 3)
        u_length = np.linalg.norm(u, axis=1)  # (M,)
        unit_u = u / u_length[:, np.newaxis]  # (M, 3)

        v = p4 - p1  # (M, 3)
        v = v - np.sum(v * unit_u, axis=1)[:, np.newaxis] * unit_u  # 正交化 (M, 3)

        v_length = np.linalg.norm(v, axis=1)  # (M,)
        # 检测并修正为零的值
        mask = v_length < epsilon  # 找出接近零的值
        v_length[mask] = epsilon  # 将接近零的值替换为epsilon

        unit_v = v / v_length[:, np.newaxis]  # (M, 3)

        # 计算投影点在局部坐标系中的坐标
        w = projection - p1[np.newaxis, :, :]  # (N, M, 3)
        u_coords = np.sum(w * unit_u[np.newaxis, :, :], axis=2)  # (N, M)
        v_coords = np.sum(w * unit_v[np.newaxis, :, :], axis=2)  # (N, M)

        # 判断投影点是否在墙面内
        in_wall = (u_coords >= 0) & (u_coords <= u_length[np.newaxis, :]) & \
                  (v_coords >= 0) & (v_coords <= v_length[np.newaxis, :])  # (N, M)


        # 对于不在墙面内的点，计算到四条边的距离
        edge_distances = np.zeros((num_points, num_walls, 4))

        # 边1: p1-p2
        edge_distances[:, :, 0] = self.vectorized_point_to_line_distance(points, p1, p2)

        # 边2: p2-p3
        edge_distances[:, :, 1] = self.vectorized_point_to_line_distance(points, p2, p3)

        # 边3: p3-p4
        edge_distances[:, :, 2] = self.vectorized_point_to_line_distance(points, p3, p4)

        # 边4: p4-p1
        edge_distances[:, :, 3] = self.vectorized_point_to_line_distance(points, p4, p1)

        # 取最小边距离
        min_edge_distances = np.min(edge_distances, axis=2)  # (N, M)

        # 合并结果：在墙面内的点使用平面距离，否则使用最小边距离
        distances = np.where(in_wall, plane_distances, min_edge_distances)
        count = len(points)
        distances = np.min(distances, axis=1)
        if z_range is not None:
            z_mask = (points[:, 2] >= z_range['min']) & (points[:, 2] <= z_range['max'])
            distances[~z_mask] = 1.0
            count = np.sum(z_mask)
        return distances,z_mask

    def vectorized_point_to_line_distance(self,points, line_start, line_end):
        """
        向量化计算点到线段的距离

        参数:
        points: 点云数组，形状为 (N, 3)
        line_start: 线段起点，形状为 (M, 3)
        line_end: 线段终点，形状为 (M, 3)

        返回:
        distances: 距离数组，形状为 (N, M)
        """
        # 扩展维度以实现广播
        points_expanded = points[:, np.newaxis, :]  # (N, 1, 3)
        line_start_expanded = line_start[np.newaxis, :, :]  # (1, M, 3)
        line_end_expanded = line_end[np.newaxis, :, :]  # (1, M, 3)

        # 计算线段向量和点到起点的向量
        line_vec = line_end_expanded - line_start_expanded  # (1, M, 3)
        point_vec = points_expanded - line_start_expanded  # (N, M, 3)

        # 计算线段长度和单位向量
        line_len = np.linalg.norm(line_vec, axis=2)  # (1, M)

        # 检测并修正为零的值
        epsilon = 1e-6  # 定义极小值防止除零
        mask = line_len < epsilon  # 找出接近零的值
        line_len[mask] = epsilon  # 将接近零的值替换为epsilon


        line_unitvec = line_vec / line_len[:, :, np.newaxis]  # (1, M, 3)

        # 计算投影长度
        projection_length = np.sum(point_vec * line_unitvec, axis=2)  # (N, M)

        # 处理三种情况
        # 1. 投影在起点之前
        before_start = projection_length < 0
        dist_before_start = np.linalg.norm(point_vec, axis=2)  # (N, M)

        # 2. 投影在终点之后
        after_end = projection_length > line_len
        dist_after_end = np.linalg.norm(points_expanded - line_end_expanded, axis=2)  # (N, M)

        # 3. 投影在线段上
        on_segment = ~(before_start | after_end)
        projection = line_start_expanded + projection_length[:, :, np.newaxis] * line_unitvec  # (N, M, 3)
        dist_on_segment = np.linalg.norm(points_expanded - projection, axis=2)  # (N, M)

        # 合并结果
        distances = np.zeros_like(projection_length)
        distances[before_start] = dist_before_start[before_start]
        distances[after_end] = dist_after_end[after_end]
        distances[on_segment] = dist_on_segment[on_segment]

        return distances

    def project_values_in_pc_color(self,values):

        colors_map = []
        # color1:068306
        colors_map.append([6, 131, 6])
        # color2:90ed90
        colors_map.append([144, 237, 144])
        # color3:ffff00
        colors_map.append([255, 255, 0])
        # color4:ffa400
        colors_map.append([255, 164, 0])
        # color5:ff0000
        colors_map.append([255, 0, 0])
        # color6:8a0000
        colors_map.append([138, 0, 0])
        # color_gray:
        colors_map.append([230, 230, 230])

        for i in range(len(colors_map)):
            colors_map[i] = [colors_map[i][0] * 257, colors_map[i][1] * 257, colors_map[i][2] * 257]

        r = np.zeros(len(values))
        g = np.zeros(len(values))
        b = np.zeros(len(values))

        discrete_range = [0.020, 0.040, 0.060, 0.080, 0.100,0.150]

        r[values < discrete_range[0]] = colors_map[0][0]
        g[values < discrete_range[0]] = colors_map[0][1]
        b[values < discrete_range[0]] = colors_map[0][2]

        r[np.logical_and(values >= discrete_range[0], values < discrete_range[1])] = colors_map[1][0]
        g[np.logical_and(values >= discrete_range[0], values < discrete_range[1])] = colors_map[1][1]
        b[np.logical_and(values >= discrete_range[0], values < discrete_range[1])] = colors_map[1][2]

        r[np.logical_and(values >= discrete_range[1], values < discrete_range[2])] = colors_map[2][0]
        g[np.logical_and(values >= discrete_range[1], values < discrete_range[2])] = colors_map[2][1]
        b[np.logical_and(values >= discrete_range[1], values < discrete_range[2])] = colors_map[2][2]

        r[np.logical_and(values >= discrete_range[2], values < discrete_range[3])] = colors_map[3][0]
        g[np.logical_and(values >= discrete_range[2], values < discrete_range[3])] = colors_map[3][1]
        b[np.logical_and(values >= discrete_range[2], values < discrete_range[3])] = colors_map[3][2]

        r[np.logical_and(values >= discrete_range[3], values < discrete_range[4])] = colors_map[4][0]
        g[np.logical_and(values >= discrete_range[3], values < discrete_range[4])] = colors_map[4][1]
        b[np.logical_and(values >= discrete_range[3], values < discrete_range[4])] = colors_map[4][2]

        r[np.logical_and(values >= discrete_range[4], values < discrete_range[5])] = colors_map[5][0]
        g[np.logical_and(values >= discrete_range[4], values < discrete_range[5])] = colors_map[5][1]
        b[np.logical_and(values >= discrete_range[4], values < discrete_range[5])] = colors_map[5][2]


        r[values >= discrete_range[5]] = colors_map[6][0]
        g[values >= discrete_range[5]] = colors_map[6][1]
        b[values >= discrete_range[5]] = colors_map[6][2]

        self.las.red = r.astype(np.uint16)
        self.las.green = g.astype(np.uint16)
        self.las.blue = b.astype(np.uint16)


    def calculate_nearest_wall_number(self):
        points = np.vstack((self.las.x, self.las.y, self.las.z)).transpose()
        wall_number = np.zeros(len(points))

        bar = tqdm(total = len(points),desc='Calculating wall number')
        for i, point in enumerate(points):
            distances = [self.point_to_plane_distance(point, wall) for wall in self.walls]
            # 保存最近墙面的距离
            min_dist = min(distances)
            min_dist_idx = distances.index(min_dist)
            if min_dist > 0.5:
                min_dist_idx = -1
            wall_number[i] = min_dist_idx
            bar.update(1)
        bar.close()

        # 创建新的点云特征维度
        if 'wall_num' not in self.las.point_format.extra_dimension_names:
            self.las.add_extra_dim(laspy.ExtraBytesParams(
                name="wall_num",
                type="int",
                description="Distance to nearest wall"
            ))

        # 赋值墙面距离
        self.las.wall_num = wall_number.astype(np.int32)

        self.las.write(self.las_save_path)


def read_obj_file(file_path):
    """
    读取 OBJ 文件并解析其内容
    """
    data = {
        'vertices': [],
        'faces': []
    }

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                prefix = parts[0]

                if prefix == 'v':  # 顶点坐标
                    vertex = tuple(map(float, parts[1:4]))
                    data['vertices'].append(vertex)
                elif prefix == 'f':  # 面信息
                    face = []
                    for vertex_data in parts[1:]:
                        indices = vertex_data.split('/')
                        # OBJ 索引从 1 开始，转换为 Python 从 0 开始的索引
                        v_idx = int(indices[0]) - 1 if indices[0] else None
                        face.append(v_idx)
                    data['faces'].append(face)

    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'")
        return None
    except Exception as e:
        print(f"错误：读取文件时发生异常: {e}")
        return None

    return data


if __name__ == '__main__':
    # # Test the RMS_Evaluator class
    floor_path = r''
    las_path = r''
    output_path = r''
    evaluator = RMS_Evaluator(las_path,floor_path,output_las_result_path=output_path)
    evaluator.calculate_rms(auto_clip_z=True,mode = 'fast')
