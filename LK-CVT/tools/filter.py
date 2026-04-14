import csv
import os
import shutil

# ¶¨ÒåÎÄ¼þÂ·¾¶
csv_file_path = '/home/UserData/les/Awesome-Backbones-main/eval_results/TinyViT/2025-01-25-22-11-10/prediction_results.csv'  # CSVÎÄ¼þÂ·¾¶£¬¸ù¾ÝÊµ¼ÊÇé¿öÐÞ¸Ä
wrong_dir = '/home/UserData/les/Awesome-Backbones-main/test/wrong'  # ´íÎóÍ¼ÏñµÄÄ¿±êÎÄ¼þ¼Ð

# È·±£Ä¿±êÎÄ¼þ¼Ð´æÔÚ
os.makedirs(os.path.join(wrong_dir, '0'), exist_ok=True)
os.makedirs(os.path.join(wrong_dir, '1'), exist_ok=True)

# ¶ÁÈ¡CSVÎÄ¼þ²¢´¦Àí
with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        file_path = row['File']
        pre_label = row['Pre_label']
        true_label = row['True_label']
        success = row['Success']

        # ¼ì²éÊÇ·ñÔ¤²â´íÎó
        if success == 'False':
            # ¸ù¾ÝÕæÊµ±êÇ©ºÍÔ¤²â±êÇ©È·¶¨Ä¿±êÎÄ¼þ¼Ð
            if true_label == 'class_0' and pre_label == 'class_1':
                target_folder = os.path.join(wrong_dir, '0')
            elif true_label == 'class_1' and pre_label == 'class_0':
                target_folder = os.path.join(wrong_dir, '1')
            else:
                continue  # Èç¹û²»·ûºÏÌõ¼þ£¬Ìø¹ý

            # ¸´ÖÆÎÄ¼þµ½Ä¿±êÎÄ¼þ¼Ð
            shutil.copy(file_path, target_folder)
            print(f"Moved: {file_path} -> {target_folder}")