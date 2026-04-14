import os
import shutil

source_dir = "/home/UserData/les/Awesome-Backbones-main/不同朝向四腔心及四类非四腔心/心尖背向探头vs全部非四腔心/train"
target_dir = "/home/UserData/les/Awesome-Backbones-main/不同朝向四腔心及四类非四腔心/心尖背向探头vs全部非四腔心/test"

os.makedirs(target_dir, exist_ok=True)

for class_name in os.listdir(source_dir):
    class_source_path = os.path.join(source_dir, class_name)

    if os.path.isdir(class_source_path):
        class_target_path = os.path.join(target_dir, class_name)
        os.makedirs(class_target_path, exist_ok=True)

        for filename in os.listdir(class_source_path):
            file_source_path = os.path.join(class_source_path, filename)
            file_target_path = os.path.join(class_target_path, filename)

            # �����ļ�
            shutil.copy2(file_source_path, file_target_path)
            print(f"Copied: {file_source_path} -> {file_target_path}")

print("All files have been copied successfully!")