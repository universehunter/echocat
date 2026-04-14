import os
import sys

# 将当前目录添加到路径，确保能导入项目模块
sys.path.insert(0, os.getcwd())
import copy
import argparse
import shutil
import time
import numpy as np
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

# 导入项目自定义模块
from utils.history import History
from utils.dataloader import Mydataset, collate
from utils.train_utils import train, validation, print_info, file2dict, init_random_seed, set_random_seed, resume_model
from utils.inference import init_model
from core.optimizers import *
from models.build import BuildNet

import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')


def main():
    # ========================= 用户配置区域 =========================

    # 1. 配置文件路径
    config_file_path = '/home/UserData/les/Awesome-Backbones-main/datas/train_output/测试集1-增强_诊断二分类/tinyvit_21m.py'

    # 2. 标签总文件路径 (只需这一个，脚本会自动分割)
    total_annotations_path = '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类11.29/华西诊断训练集增强-labels.txt'

    # 3. 自定义输出保存目录
    custom_save_dir = '/home/UserData/les/Awesome-Backbones-main/datas/诊断分类11.29/华西诊断训练集增强-重新训练'

    # 4. 验证集比例 (默认0.2，即 8:2 分割)
    val_ratio = 0.2

    # 5. 显卡设置
    gpu_id = 0

    # 6. 随机种子 (固定种子保证每次划分一致)
    seed_value = 42

    # 7. 断点续训 (如有需要填入 .pth 路径，否则为 None)
    resume_checkpoint = None

    # ===============================================================

    # 读取配置
    model_cfg, train_pipeline, val_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(config_file_path)
    print_info(model_cfg)

    # 初始化 Meta 信息与保存目录
    meta = dict()
    save_dir = custom_save_dir
    meta['save_dir'] = save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建输出目录: {save_dir}")

    # 设置随机数种子
    seed = init_random_seed(seed_value)
    set_random_seed(seed, deterministic=True)
    meta['seed'] = seed

    # ================= 数据读取与严格清洗逻辑 =================
    print(f"正在读取标签文件: {total_annotations_path}")
    with open(total_annotations_path, encoding='utf-8') as f:
        raw_lines = f.readlines()

    print(f"原始行数: {len(raw_lines)}")

    total_datas = []
    skipped_count = 0
    missing_file_count = 0

    for i, line in enumerate(raw_lines):
        line = line.strip()
        if not line:
            skipped_count += 1
            continue

        # 【核心修复】：使用 rsplit 从右边切分一次
        # 这样可以正确处理路径中包含空格的情况 (例如: "/data/张三  李四/1.jpg 0")
        try:
            img_path, label_str = line.rsplit(' ', 1)
        except ValueError:
            # 如果这一行切分不出两部分（比如没有空格），说明格式严重错误
            # print(f"[格式错误] 第 {i+1} 行无法解析: {line}")
            skipped_count += 1
            continue

        # 检查文件是否存在
        if not os.path.exists(img_path):
            # 调试时可以打开下面这行 print，查看具体缺失的文件
            # print(f"[文件缺失] 跳过: {img_path}")
            missing_file_count += 1
            continue

        # 重新组合标准格式，确保没有多余字符
        clean_line = f"{img_path} {label_str}\n"
        total_datas.append(clean_line)

    total_nums = len(total_datas)
    print("-" * 40)
    print(f"格式错误跳过: {skipped_count}")
    print(f"文件缺失跳过: {missing_file_count}")
    print(f"最终有效数据: {total_nums}")
    print("-" * 40)

    if total_nums == 0:
        print("【严重错误】有效数据为0，请检查标签文件路径或图片路径是否正确！")
        return

    # 打乱数据
    rng = np.random.default_rng(seed)
    total_datas_shuffled = copy.deepcopy(total_datas)
    rng.shuffle(total_datas_shuffled)

    # 8:2 切分
    val_nums = int(total_nums * val_ratio)
    train_nums = total_nums - val_nums

    train_datas = total_datas_shuffled[:train_nums]
    val_datas = total_datas_shuffled[train_nums:]

    print(f"训练集: {len(train_datas)} | 验证集: {len(val_datas)}")

    # 保存切分后的标签文件
    train_txt_save_path = os.path.join(save_dir, 'train_split.txt')
    val_txt_save_path = os.path.join(save_dir, 'val_split.txt')

    with open(train_txt_save_path, 'w', encoding='utf-8') as f:
        f.writelines(train_datas)
    with open(val_txt_save_path, 'w', encoding='utf-8') as f:
        f.writelines(val_datas)

    print(f"已保存切分名单至: {save_dir}")
    # ============================================================

    # 初始化模型
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    model = BuildNet(model_cfg)

    if not data_cfg.get('train').get('pretrained_flag'):
        model.init_weights()

    if data_cfg.get('train').get('freeze_flag') and data_cfg.get('train').get('freeze_layers'):
        freeze_layers = ' '.join(list(data_cfg.get('train').get('freeze_layers')))
        print('Freeze layers : ' + freeze_layers)
        model.freeze_layers(data_cfg.get('train').get('freeze_layers'))

    if device.type != 'cpu':
        model = DataParallel(model, device_ids=[gpu_id])
        model.to(device)

    # 优化器与学习率
    optimizer = eval('optim.' + optimizer_cfg.pop('type'))(params=model.parameters(), **optimizer_cfg)
    lr_update_func = eval(lr_config.pop('type'))(**lr_config)

    # 构建 DataLoader
    train_dataset = Mydataset(train_datas, train_pipeline)
    # 如果没有定义验证集 pipeline，则复用训练集的
    if val_pipeline is None:
        val_pipeline = copy.deepcopy(train_pipeline)
    val_dataset = Mydataset(val_datas, val_pipeline)

    train_loader = DataLoader(
        train_dataset, shuffle=True,
        batch_size=data_cfg.get('batch_size'), num_workers=data_cfg.get('num_workers'),
        pin_memory=True, drop_last=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False,
        batch_size=data_cfg.get('batch_size'), num_workers=data_cfg.get('num_workers'),
        pin_memory=True, drop_last=True, collate_fn=collate
    )

    # Runner 状态管理
    runner = dict(
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        iter=0,
        epoch=0,
        max_epochs=data_cfg.get('train').get('epoches'),
        max_iters=data_cfg.get('train').get('epoches') * len(train_loader),
        best_train_loss=float('INF'),
        best_val_acc=float(0),
        best_train_weight='',
        best_val_weight='',
        last_weight=''
    )
    meta['train_info'] = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[])

    # 恢复或初始化
    if resume_checkpoint:
        print(f"Resuming form {resume_checkpoint} ...")
        model, runner, meta = resume_model(model, runner, resume_checkpoint, meta)
    else:
        # 备份配置文件
        shutil.copyfile(config_file_path, os.path.join(save_dir, os.path.split(config_file_path)[1]))
        model = init_model(model, data_cfg, device=device, mode='train')

    train_history = History(meta['save_dir'])
    lr_update_func.before_run(runner)

    print("Start Training...")
    # 训练循环
    for epoch in range(runner.get('epoch'), runner.get('max_epochs')):
        lr_update_func.before_train_epoch(runner)

        train(model, runner, lr_update_func, device, epoch, data_cfg.get('train').get('epoches'), data_cfg.get('test'),
              meta)
        validation(model, runner, data_cfg.get('test'), device, epoch, data_cfg.get('train').get('epoches'), meta)

        train_history.after_epoch(meta)


if __name__ == "__main__":
    main()