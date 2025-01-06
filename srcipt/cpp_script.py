import os
from multiprocessing import Pool, Manager
from tqdm import tqdm

from util.multi_thread_process import show_progress, Param, task_function


def mesh_generate(root_dir, label_dir, result_dir, point_cloud_name, label_name):

    total_tasks = len(os.listdir(root_dir))
    # 使用 Manager 提供的队列进行进程间通信
    with Manager() as manager:
        progress_queue = manager.Queue()
        # 启动进度显示线程
        from threading import Thread
        progress_thread = Thread(target=show_progress, args=(total_tasks, progress_queue))
        progress_thread.start()

        # 使用 Pool 创建进程池并执行任务
        pool = Pool(3)

        for scene in os.listdir(root_dir):
            if scene >= "scene_00250":
                break
            input_file = os.path.join(root_dir, scene, f"{point_cloud_name}")
            output_file = os.path.join(result_dir, scene)
            label_file = os.path.join(label_dir, scene, f"{label_name}")
            if not os.path.exists(input_file) or not os.path.exists(label_file):
                continue
            if os.path.exists(output_file):
                continue
            param = Param(input_file, output_file, label_file, scene)
            pool.apply_async(task_function, (param, progress_queue))

        pool.close()  # 关闭进程池，不再接受新的进程
        pool.join()  # 主进程阻塞等待子进程的退出

        progress_queue.put(("None", "None"))  # 通知进度显示线程所有任务已完成
        progress_thread.join()

    print("所有任务处理完成！")

if __name__ == '__main__':
    root_dir = r"G:\workspace_plane2DDL\real_point_cloud_dataset\stru3d_pointcloud_scale_0.001"
    label_dir = r"G:\workspace_plane2DDL\real_point_cloud_dataset\stru3d_anno_scale_0.001"
    result_dir = r"G:\workspace_plane2DDL\real_point_cloud_dataset\stru3d_featured_shp"
    point_cloud_name = "point_cloud.las"
    label_name = "GT_room_poly.shp"
    mesh_generate(root_dir, label_dir, result_dir, point_cloud_name, label_name)