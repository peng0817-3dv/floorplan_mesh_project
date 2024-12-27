import os
import threading
import time
from multiprocessing import Pool, Manager
from queue import Queue as normalQueue

from tqdm import tqdm

CPP_EXE_PATH = r"F:\DIP\DipTools_Indoor\bin\Release\ExportMeshGptFeature.exe"


class Param:
    def __init__(self, input_file, output_file, label_file,scene_name):
        self.input_file = input_file
        self.output_file = output_file
        self.label_file = label_file
        self.scene_name = scene_name


def from_ply_to_las(input_file):
    ply_path = os.path.join(input_file, "point_cloud.ply")
    if not os.path.exists(ply_path):
        return
    cmd = "CloudCompare -VERBOSITY 4 -SILENT -O {0} -C_EXPORT_FMT LAS -NO_TIMESTAMP -SAVE_CLOUDS".format(
        ply_path
    )
    os.system(cmd)
    os.remove(ply_path)


def cpp_exe_process(input_file, label_file, output_file):
    cmd = "{0} -i {1} --label_file {2} -o {3} --mode las".format(CPP_EXE_PATH,input_file,label_file,output_file)
    # print(cmd)
    os.system(cmd)


def task_function(params:Param, progress_queue):
    scene_name = params.scene_name
    # print(f"Processing file:{scene_name}")
    progress_queue.put((scene_name, "processing"))
    cpp_exe_process(params.input_file,params.label_file,params.output_file)
    progress_queue.put((scene_name, "done"))
    # print(f"Processed file:{scene_name}")

def show_progress(total_tasks,progress_queue):
    completed_tasks = 0
    last_completed_tasks = 0
    task_progress = {}
    tqdm_bar = tqdm(total=total_tasks)

    while completed_tasks < total_tasks:
        diff_tasks = completed_tasks - last_completed_tasks
        last_completed_tasks = completed_tasks

        scene_name, progress_status = progress_queue.get()
        if scene_name == "None":
            print("All tasks completed")
            break
        task_progress[scene_name] = progress_status
        completed_tasks = sum(1 for p in task_progress.values() if p == "done")

        processing_scenes = [scene_name for scene_name, progress_status in task_progress.items() if progress_status == "processing"]
        tqdm_bar.update(diff_tasks)
        tqdm_bar.set_description(f"Processing scenes: {processing_scenes}")
        # print("==Processing scenes: ", processing_scenes)
        # print(f"==Processed : {completed_tasks}/{total_tasks}")

    diff_tasks = completed_tasks - last_completed_tasks
    tqdm_bar.update(diff_tasks)
    tqdm_bar.close()



def main():
    root_dir = r"G:\workspace_plane2DDL\real_point_cloud_dataset\augment_stru3d_pointcloud"
    label_dir = r"G:\workspace_plane2DDL\real_point_cloud_dataset\augment_stru3d_anno"
    result_dir = r"G:\workspace_plane2DDL\real_point_cloud_dataset\augment_stru3d_featured_shp"
    point_cloud_name = "scale.laz"
    label_name = "GT_room_poly.shp"
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


def test_cpp_exe():
    root_dir = r"G:\workspace_plane2DDL\real_point_cloud_dataset\augment_stru3d_pointcloud"
    label_dir = r"G:\workspace_plane2DDL\real_point_cloud_dataset\augment_stru3d_anno"
    result_dir = r"G:\workspace_plane2DDL\real_point_cloud_dataset\augment_stru3d_featured_shp"
    scene = "scene_00001"
    cpp_exe_process(os.path.join(root_dir, scene, "scale.laz"), os.path.join(label_dir, scene, "GT_room_poly.shp")\
                    , os.path.join(result_dir, scene))


if __name__ == '__main__':
    # from_ply_to_las(r"H:\0-400\scene_00004")
    main()
    # test_cpp_exe()