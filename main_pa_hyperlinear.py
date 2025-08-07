# coding=utf-8
import argparse
import os
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # main root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
from tqdm import tqdm
from copy import deepcopy
import time

from utils.general import set_seed
from utils.dataloader import CustomDataLoader
from models.pa_hyperlinear import PAHyperLinear


def main(args):
    # 选择设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 设置线程数
    torch.set_num_threads(4)
    # 设置随机种子
    set_seed(args.seed)

    # 在模型创建前添加
    print("=== 数据加载调试 ===")
    data_loader = CustomDataLoader(
        args.data,
        args.batch_size,
        args.seq_len,
        args.pred_len,
        args.feature_type,
        args.target,
    )

    print(f"数据路径: {args.data}")
    print(f"特征数量: {data_loader.n_feature}")

    # 检查实际数据形状
    train_data = data_loader.get_train()
    for batch_x, batch_y in train_data:
        print(f"原始输入形状: {batch_x.shape}")
        print(f"原始输出形状: {batch_y.shape}")
        break

    print("=== 模型创建调试 ===")
    model = PAHyperLinear(
        input_shape=(args.seq_len, data_loader.n_feature),
        pred_len=args.pred_len,
        k_periods=args.k_periods,
        d_model=args.d_model,
        dropout=args.dropout,
        gdd_reduction=args.gdd_reduction,
        target_slice=data_loader.target_slice,
    ).to(device)

    print(f"模型期望输入: {model.seq_len} x {model.n_features}")



    # 加载数据集
    data_loader = CustomDataLoader(
        args.data,
        args.batch_size,
        args.seq_len,
        args.pred_len,
        args.feature_type,
        args.target,
    )

    train_data = data_loader.get_train()
    val_data = data_loader.get_val()
    test_data = data_loader.get_test()

    # 创建PA-HyperLinear模型
    model = PAHyperLinear(
        input_shape=(args.seq_len, data_loader.n_feature),
        pred_len=args.pred_len,
        k_periods=args.k_periods,
        d_model=args.d_model,
        dropout=args.dropout,
        gdd_reduction=args.gdd_reduction,
        target_slice=data_loader.target_slice,
    ).to(device)

    # 打印模型参数量
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 设置损失函数和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-9)

    best_loss = torch.tensor(float('inf'))

    # 创建检查点目录
    save_directory = os.path.join(args.checkpoint_dir, args.name)
    if os.path.exists(save_directory):
        import glob
        import re

        path = Path(save_directory)
        dirs = glob.glob(f"{path}*")  # 相似路径
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # 索引
        n = max(i) + 1 if i else 2  # 递增数字
        save_directory = f"{path}{n}"  # 更新路径

    os.makedirs(save_directory)

    # 开始训练
    for epoch in range(args.train_epochs):
        model.train()
        train_mloss = torch.zeros(1, device=device)
        iter_time = 0
        print(f"轮次: {epoch + 1}")
        print("训练中...")
        pbar = tqdm(enumerate(train_data), total=len(train_data))

        for i, (batch_x, batch_y) in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            start_time = time.time()

            # 前向传播
            outputs, aux_loss = model(batch_x)

            # 计算总损失
            main_loss = criterion(outputs, batch_y)
            total_loss = main_loss + aux_loss

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            end_time = time.time()
            train_mloss = (train_mloss * i + total_loss.detach()) / (i + 1)
            pbar.set_description(('%-10s' * 1 + '%-10.8g ' * 1) % (f'{epoch + 1}/{args.train_epochs}', train_mloss))
            iteration_time = (end_time - start_time) * 1000
            iter_time = (iter_time * i + iteration_time) / (i + 1)

        print(f"训练损失: {train_mloss.item()}, 迭代时间: {iter_time:.4f}ms")

        # 验证阶段
        model.eval()
        val_mloss = torch.zeros(1, device=device)
        val_mae = torch.zeros(1, device=device)
        val_mse = torch.zeros(1, device=device)
        print("验证中...")
        pbar = tqdm(enumerate(val_data), total=len(val_data))

        with torch.no_grad():
            for i, (batch_x, batch_y) in pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs, aux_loss = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_mloss = (val_mloss * i + loss.detach()) / (i + 1)
                mae = torch.abs(outputs - batch_y).mean()
                val_mae = (val_mae * i + mae.detach()) / (i + 1)
                mse = ((outputs - batch_y) ** 2).mean()
                val_mse = (val_mse * i + mse.detach()) / (i + 1)
                pbar.set_description(('%-10s' * 1 + '%-10.8g' * 1) % ('', val_mloss))

            # 保存最佳模型
            if val_mloss < best_loss or epoch == args.train_epochs - 1:
                best_loss = val_mloss
                best_model = deepcopy(model.state_dict())
                torch.save(best_model, os.path.join(save_directory, "best.pt"))

        print(f"验证损失: {val_mloss.item()}, 验证MSE: {val_mse.item()}, 验证MAE: {val_mae.item()}")

        # 打印模型信息 (仅第一轮)
        if epoch == 0:
            model_info = model.get_model_info()
            print(f"检测到的周期: {model_info.get('detected_periods', 'N/A')}")

    # 加载最佳模型
    model.load_state_dict(best_model)

    # 开始测试
    model.eval()
    test_mloss = torch.zeros(1, device=device)
    test_mae = torch.zeros(1, device=device)
    test_mse = torch.zeros(1, device=device)

    print("最终测试中...")
    pbar = tqdm(enumerate(test_data), total=len(test_data))

    with torch.no_grad():
        for i, (batch_x, batch_y) in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs, aux_loss = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_mloss = (test_mloss * i + loss.detach()) / (i + 1)
            mae = torch.abs(outputs - batch_y).mean()
            test_mae = (test_mae * i + mae.detach()) / (i + 1)
            mse = ((outputs - batch_y) ** 2).mean()
            test_mse = (test_mse * i + mse.detach()) / (i + 1)
            pbar.set_description(('%-10.8g' * 1) % (test_mloss))

    print(f"测试损失: {test_mloss.item()}, 测试MSE: {test_mse.item()}, 测试MAE: {test_mae.item()}")

    # 保存测试结果
    save_test_results(args, test_mse.item(), test_mae.item())

    # 保存模型信息
    save_model_info(model, save_directory)


def save_test_results(args, test_mse, test_mae):
    """
    保存测试结果到txt文件
    """
    # 从数据路径中提取数据集名称
    data_path = Path(args.data)
    dataset_name = data_path.stem

    # 创建results目录
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # 构建结果文件名
    result_filename = f"{dataset_name}_pa_hyperlinear_results.txt"
    result_filepath = results_dir / result_filename

    # 构建模型配置信息字符串
    model_config = (
        f"PAHyperLinear_{dataset_name}_M_ft{args.seq_len}_sl{args.seq_len}_ll{args.pred_len}_"
        f"k{args.k_periods}_dm{args.d_model}_dr{args.dropout}_"
        f"gdd{args.gdd_reduction}_fc{args.feature_type}"
    )

    # 准备结果内容
    result_content = (
        f"{dataset_name}_{args.seq_len}_{args.pred_len}_{model_config}\n"
        f"mse:{test_mse:.8f}, mae:{test_mae:.8f}\n\n"
    )

    # 追加写入文件
    with open(result_filepath, 'a', encoding='utf-8') as f:
        f.write(result_content)

    print(f"测试结果已保存到: {result_filepath}")


def save_model_info(model, save_directory):
    """
    保存模型详细信息
    """
    model_info = model.get_model_info()
    info_path = os.path.join(save_directory, "model_info.txt")

    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("PA-HyperLinear模型信息\n")
        f.write("=" * 50 + "\n")
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n")

    print(f"模型信息已保存到: {info_path}")


def infer_extension(dataset_name):
    """推断数据文件扩展名"""
    if dataset_name.startswith('solar'):
        extension = 'txt'
    elif dataset_name.startswith('PEMS'):
        extension = 'npz'
    else:
        extension = 'csv'
    return extension


def parse_args():
    dataset = "ETTh1"
    parser = argparse.ArgumentParser(description='PA-HyperLinear时间序列预测模型')

    # 基础配置
    parser.add_argument('--seed', type=int, default=2025, help='随机种子')

    # 数据加载器
    parser.add_argument('--data', type=str,
                        default=ROOT / f'./data/{dataset}.{infer_extension(dataset)}',
                        help='数据集路径')
    parser.add_argument('--feature_type', type=str, default='M',
                        choices=['S', 'M', 'MS'], help='预测任务类型')
    parser.add_argument('--target', type=str, default='OT', help='目标特征')
    parser.add_argument('--checkpoint_dir', type=str, default=ROOT / 'checkpoints',
                        help='模型检查点位置')
    parser.add_argument('--name', type=str, default=f'{dataset}',
                        help='保存最佳模型到 checkpoints/name')

    # 预测任务
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=96, help='预测序列长度')

    # PA-HyperLinear 特有参数
    parser.add_argument('--k_periods', type=int, default=3, help='周期数量')
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--gdd_reduction', type=int, default=2,
                        help='GDD-MLP中间层压缩比例')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout率')

    # 优化
    parser.add_argument('--train_epochs', type=int, default=10, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='学习率')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)