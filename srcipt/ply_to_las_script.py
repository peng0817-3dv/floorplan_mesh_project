import os
from multiprocessing import Manager, Pool

from util.multi_thread_process import show_progress

def from_ply_to_las(input_file):
    ply_path = os.path.join(input_file, "point_cloud.ply")
    if not os.path.exists(ply_path):
        return
    cmd = "CloudCompare -VERBOSITY 4 -SILENT -O {0} -C_EXPORT_FMT LAS -NO_TIMESTAMP -SAVE_CLOUDS".format(
        ply_path
    )
    os.system(cmd)
    os.remove(ply_path)


class Param:
    def __init__(self, scene_name, input_file):
        self.scene_name = scene_name
        self.input_file = input_file


def task_function(params:Param, progress_queue):
    scene_name = params.scene_name
    # print(f"Processing file:{scene_name}")
    progress_queue.put((scene_name, "processing"))
    from_ply_to_las(params.input_file)
    progress_queue.put((scene_name, "done"))
    # print(f"Processed file:{scene_name}")

def main():
    root_dir = r"G:\workspace_plane2DDL\real_point_cloud_dataset\stru3d_pointcloud_scale_0.001"
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
            input_file = os.path.join(root_dir, scene)
            if not os.path.exists(input_file):
                continue
            params = Param(scene, input_file)
            pool.apply_async(task_function, (params, progress_queue,))

        pool.close()  # 关闭进程池，不再接受新的进程
        pool.join()  # 主进程阻塞等待子进程的退出

        progress_queue.put(("None", "None"))  # 通知进度显示线程所有任务已完成
        progress_thread.join()

    print("所有任务处理完成！")

def test():
    input_file = r"G:\workspace_plane2DDL\real_point_cloud_dataset\stru3d_pointcloud_scale_0.001\scene_00000"
    from_ply_to_las(input_file)

if __name__ == '__main__':
    # test()
    main()