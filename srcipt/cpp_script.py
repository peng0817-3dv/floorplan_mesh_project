import os
import shutil
from multiprocessing import Pool, Manager
from tqdm import tqdm

from util.multi_thread_process import show_progress, Param, task_function
from batch_copy_script import batch_copy_files

'''
由于cpp程序执行单场景点云的mesh生成耗时较长，我们在python环境下使用多进程的方式执行cpp程序，以加速整体数据集的mesh生成
'''

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


def mesh_generate_specific(root_dir, label_dir, result_dir, point_cloud_name, label_name, specific_scenes):

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
            if scene not in specific_scenes:
                continue
            input_file = os.path.join(root_dir, scene, f"{point_cloud_name}")
            output_file = os.path.join(result_dir, scene)
            label_file = os.path.join(label_dir, scene, f"{label_name}")
            if not os.path.exists(input_file) or not os.path.exists(label_file):
                continue
            # if os.path.exists(output_file):
            #     continue
            param = Param(input_file, output_file, label_file, scene)
            pool.apply_async(task_function, (param, progress_queue))

        pool.close()  # 关闭进程池，不再接受新的进程
        pool.join()  # 主进程阻塞等待子进程的退出

        progress_queue.put(("None", "None"))  # 通知进度显示线程所有任务已完成
        progress_thread.join()

    print("所有任务处理完成！")


def check_failure_scene(result_dir):
    bar = tqdm(total=len(os.listdir(result_dir)))
    top_file = "raster_cover.top"
    failure_scene = []
    for scene in os.listdir(result_dir):
        bar.update(1)
        if not os.path.exists(os.path.join(result_dir, scene, top_file)):
            failure_scene.append(scene)
            continue
        with open(os.path.join(result_dir, scene, top_file), "r") as f:
            lines = f.readlines()
            line_number = lines[2].split(':')[1].strip()
            line_number = int(line_number)
            if line_number == 0:
                failure_scene.append(scene)
    bar.close()
    return failure_scene


def remove_failure_scene(result_dir, failure_scene):
    for scene in failure_scene:
        if os.path.exists(os.path.join(result_dir, scene)):
            shutil.rmtree(os.path.join(result_dir, scene))
    print(f"失败场景数:{len(failure_scene)}, 已删除. f{failure_scene}")


def copy_GT_room_poly(anno_root, feature_root):
    bar = tqdm(total=len(os.listdir(feature_root)))
    for scene in os.listdir(feature_root):
        anno_file_1 = os.path.join(anno_root, scene, 'GT_room_poly.shp')
        anno_file_2 = os.path.join(anno_root, scene, 'GT_room_poly.dbf')
        anno_file_3 = os.path.join(anno_root, scene, 'GT_room_poly.shx')
        target_dir = os.path.join(feature_root, scene)
        shutil.copy(anno_file_1, target_dir)
        shutil.copy(anno_file_2, target_dir)
        shutil.copy(anno_file_3, target_dir)
        bar.update(1)
    bar.close()


def main_1():
    root_dir =   r"...\structured3d_scale_0.001_scene_3000_3499"
    label_dir = r"...\real_point_cloud_dataset\stru3d_anno_scale_0.001"
    result_dir = r"...\real_point_cloud_dataset\stru3d_featured_shp_bbox_10_percent"
    point_cloud_name = "point_cloud.las"
    label_name = "GT_room_poly.shp"
    specific_scenes = ['scene_00001','scene_00011','scene_00020','scene_00030','scene_00040','scene_00050','scene_00060',
                       'scene_00230','scene_00240','scene_00250']
    # mesh_generate_specific(root_dir, label_dir, result_dir, point_cloud_name, label_name, specific_scenes=specific_scenes)
    mesh_generate(root_dir, label_dir, result_dir, point_cloud_name, label_name)
    failure_scene = check_failure_scene(result_dir)
    remove_failure_scene(result_dir, failure_scene)
    #
    # copy_GT_room_poly(label_dir, result_dir)
    # mxd_file = r"G:\workspace_plane2DDL\confidence_and_GT.mxd"
    # batch_copy_files(mxd_file, result_dir)


def main_2():
    root_dir = r"H:\3400_3410"
    label_dir = r"...\stru3d_anno_scale_0.001"
    result_dir = r"...\full_s3d_2"
    point_cloud_name = "point_cloud.las"
    label_name = "GT_room_poly.shp"
    specific_scenes = ['scene_00001','scene_00011','scene_00020','scene_00030','scene_00040','scene_00050','scene_00060',
                       'scene_00230','scene_00240','scene_00250']
    # mesh_generate_specific(root_dir, label_dir, result_dir, point_cloud_name, label_name, specific_scenes=specific_scenes)
    mesh_generate(root_dir, label_dir, result_dir, point_cloud_name, label_name)
    failure_scene = check_failure_scene(result_dir)
    remove_failure_scene(result_dir, failure_scene)
    #
    copy_GT_room_poly(label_dir, result_dir)
    mxd_file = r"...\confidence_and_GT.mxd"
    batch_copy_files(mxd_file, result_dir)



def cpp_exe_process_without_label(input_file, output_file):
    CPP_EXE_PATH = r"...\ExportMeshGptFeature.exe"
    cmd = "{0} -i {1} -o {2} --mode las".format(CPP_EXE_PATH,input_file,output_file)
    os.system(cmd)


def main_3():
    input_dir = r"...\other_point_cloud\pointcloud\1.las"
    output_dir = r"...\other_point_cloud\shp_root\SCD_001"
    cpp_exe_process_without_label(input_dir, output_dir)

if __name__ == '__main__':
    main_2()
