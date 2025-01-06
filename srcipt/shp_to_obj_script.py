import os
from multiprocessing import Manager, Pool

from util.analyse_dataset import shp_to_obj
from util.multi_thread_process import show_progress


class Param:
    def __init__(self, scene_name, shp_path, obj_path):
        self.scene_name = scene_name
        self.shp_path = shp_path
        self.obj_path = obj_path


def task_function(params:Param, progress_queue):
    scene_name = params.scene_name
    # print(f"Processing file:{scene_name}")
    progress_queue.put((scene_name, "processing"))
    shp_to_obj(params.shp_path, params.obj_path)
    progress_queue.put((scene_name, "done"))
    # print(f"Processed file:{scene_name}")


def main():
    root_dir = r"G:\workspace_plane2DDL\real_point_cloud_dataset\stru3d_anno_scale_0.001"
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
            shp_path = os.path.join(root_dir, scene, "GT_room_poly.shp")
            if not os.path.exists(shp_path):
                continue
            obj_path = os.path.join(root_dir, scene, f"{scene}.obj")
            params = Param(scene, shp_path, obj_path)
            pool.apply_async(task_function, (params, progress_queue,))

        pool.close()  # 关闭进程池，不再接受新的进程
        pool.join()  # 主进程阻塞等待子进程的退出

        progress_queue.put(("None", "None"))  # 通知进度显示线程所有任务已完成
        progress_thread.join()

    print("所有任务处理完成！")


if __name__ == '__main__':
    # test()
    main()
