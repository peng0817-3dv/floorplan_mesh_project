import os


def generate_split_scene_txt(start, end, txt_file):
    # 打开文件准备写入
    with open(txt_file, "w") as file:
        for i in range(start, end + 1):
            # 格式化 scene 编号为 5 位数字
            scene_name = f"scene_{i:05d}"
            # 写入文件
            file.write(scene_name + "\n")

    print(f"已生成从 scene_{start:05d} 到 scene_{end:05d} 的文本，并保存到 {txt_file} 文件中。")


def generate_train_val_test_txt(txt_file_root):
    train_txt_file = os.path.join(txt_file_root, "train.txt")
    val_txt_file = os.path.join(txt_file_root, "val.txt")
    test_txt_file = os.path.join(txt_file_root, "test.txt")
    generate_split_scene_txt(0,2999,train_txt_file)
    generate_split_scene_txt(3000,3249,val_txt_file)
    generate_split_scene_txt(3250,3499,test_txt_file)

if __name__ == '__main__':
    generate_train_val_test_txt(r"G:\workspace_plane2DDL\real_point_cloud_dataset\augment_stru3d_bbox_10_percent_shp")