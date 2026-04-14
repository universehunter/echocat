import cv2
import os


def detect_and_delete_corrupt_images(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for filename in files:
        if filename.lower().endswith(valid_extensions):
            file_path = os.path.join(folder_path, filename)
            try:
                # 尝试读取图像
                img = cv2.imread(file_path)

                # 如果图像读取失败（img 是 None），则删除该图像
                if img is None:
                    print(f"Failed to read (NoneType): {filename}. Deleting...")
                    os.remove(file_path)
                else:
                    # 进一步检测图像是否完整
                    try:
                        img.shape  # 尝试访问图像的属性来确保图像未损坏
                    except Exception as shape_err:
                        print(f"Corrupt image detected (shape error): {filename}. Deleting...")
                        os.remove(file_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}. Deleting...")
                os.remove(file_path)
        else:
            print(f"Skipped non-image file: {filename}")


# 设置要检测和删除破损图像的文件夹路径
folder_path = './datasets/test/class_0'
detect_and_delete_corrupt_images(folder_path)
