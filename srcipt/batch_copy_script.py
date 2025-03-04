import os
import shutil
import tqdm


def copy_files(src, tgt):
    dir_name = os.path.dirname(tgt)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)  # 如果目标文件夹不存在，则创建

    dest_file = os.path.join(dir_name, os.path.basename(src))
    shutil.copy(src, tgt)  # 使用 shutil.copy2 保留文件元数据（如修改时间）
    return dest_file


def batch_copy_files(src, folder_root):
    bar = tqdm.tqdm(total=len(os.listdir(folder_root)))
    src_basename = os.path.basename(src).split('.')[0]
    suffix = os.path.basename(src).split('.')[1]

    for scene_folder in os.listdir(folder_root):
        if not scene_folder.startswith('scene'):
            continue
        scene_folder_path = os.path.join(folder_root, scene_folder)
        scene_tgt_name = os.path.join(scene_folder_path, src_basename + f"_{scene_folder}.{suffix}")
        copy_files(src, scene_tgt_name)
        bar.update(1)
    bar.close()


if __name__ == '__main__':
    src_file = r"G:\workspace_plane2DDL\confidence_and_GT.mxd"
    dest_folder = r"G:\workspace_plane2DDL\real_point_cloud_dataset\stru3d_featured_shp"
    batch_copy_files(src_file, dest_folder)
