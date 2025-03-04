import os
import shutil
import tqdm

CPP_EXE_PATH = r"F:\DIP\DipTools_Indoor\bin\Release\ExportMeshGptFeature.exe"
def batch_reconstruct_files(folder_root):
    bar = tqdm.tqdm(total=len(os.listdir(folder_root)) / 2)
    # src_basename = os.path.basename(src).split('.')[0]
    for scene_folder in os.listdir(folder_root):
        if not scene_folder.startswith('scene') and not os.path.isdir(os.path.join(folder_root, scene_folder)):
            continue
        scene_folder_path = os.path.join(folder_root, scene_folder)
        cmd = "{0} -i {1} --label_file {2} -o {3} --mode reconstruct".format(CPP_EXE_PATH, scene_folder_path,
                                                                             "./tmp", "./tmp")
        # print(cmd)
        os.system(cmd)
        bar.update(1)
    bar.close()


if __name__ == '__main__':
    folder_root = r"G:\workspace_plane2DDL\to_construct\ori\inference_from_98-0_for_test_ori"
    batch_reconstruct_files(folder_root)