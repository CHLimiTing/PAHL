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

    # 在模型创建前添加 (保留您之前的调试代码)
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

    # 重新获取数据加载器实例，以供训练使用
    val_data = data_loader.get_val()
    test_data = data_loader.get_test()

    # 打印模型参数量
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 设置损失函数和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-9)

    # --- 早停机制初始化 ---
    best_loss = float('inf')
    patience_counter = 0
    # -----------------------

    # 创建检查点目录
    save_directory = os.path.join(args.checkpoint_dir, args.name)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    best_model_path = os.path.join(save_directory, "best.pt")

    # 开始训练
    for epoch in range(args.train_epochs):
        model.train()
        train_mloss = torch.zeros(1, device=device)
        iter_time = 0
        print(f"轮次: {epoch + 1}/{args.train_epochs}")
        print("训练中...")
        pbar = tqdm(enumerate(train_data), total=len(train_data))

        for i, (batch_x, batch_y) in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            start_time = time.time()

            outputs, aux_loss = model(batch_x)
            main_loss = criterion(outputs, batch_y)
            total_loss = main_loss + aux_loss

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
        pbar_val = tqdm(enumerate(val_data), total=len(val_data))

        with torch.no_grad():
            for i, (batch_x, batch_y) in pbar_val:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs, aux_loss = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_mloss = (val_mloss * i + loss.detach()) / (i + 1)
                mae = torch.abs(outputs - batch_y).mean()
                val_mae = (val_mae * i + mae.detach()) / (i + 1)
                mse = ((outputs - batch_y) ** 2).mean()
                val_mse = (val_mse * i + mse.detach()) / (i + 1)
                pbar_val.set_description(('%-10s' * 1 + '%-10.8g' * 1) % ('', val_mloss))

        print(f"验证损失: {val_mloss.item()}, 验证MSE: {val_mse.item()}, 验证MAE: {val_mae.item()}")

        # --- 早停逻辑判断 ---
        current_loss = val_mloss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            print(f"发现更好的模型，在轮次 {epoch + 1} 保存。")
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            print(f"验证损失未改善。早停计数: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"早停触发！在轮次 {epoch + 1} 停止训练。")
            break
        # ---------------------

        if epoch == 0:
            model_info = model.get_model_info()
            print(f"检测到的周期: {model_info.get('detected_periods', 'N/A')}")

    print("\n训练结束。")
    # 加载最佳模型
    print(f"加载在验证集上表现最佳的模型进行测试: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))

    # 开始测试
    model.eval()
    test_mloss = torch.zeros(1, device=device)
    test_mae = torch.zeros(1, device=device)
    test_mse = torch.zeros(1, device=device)

    print("最终测试中...")
    pbar_test = tqdm(enumerate(test_data), total=len(test_data))

    with torch.no_grad():
        for i, (batch_x, batch_y) in pbar_test:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs, aux_loss = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_mloss = (test_mloss * i + loss.detach()) / (i + 1)
            mae = torch.abs(outputs - batch_y).mean()
            test_mae = (test_mae * i + mae.detach()) / (i + 1)
            mse = ((outputs - batch_y) ** 2).mean()
            test_mse = (test_mse * i + mse.detach()) / (i + 1)
            pbar_test.set_description(('%-10.8g' * 1) % (test_mloss))

    print(f"测试损失: {test_mloss.item()}, 测试MSE: {test_mse.item()}, 测试MAE: {test_mae.item()}")

    save_test_results(args, test_mse.item(), test_mae.item())
    save_model_info(model, save_directory)


def save_test_results(args, test_mse, test_mae):
    data_path = Path(args.data)
    dataset_name = data_path.stem
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    result_filename = f"{dataset_name}_pa_hyperlinear_results.txt"
    result_filepath = results_dir / result_filename
    model_config = (
        f"PAHyperLinear_{dataset_name}_M_ft{args.seq_len}_sl{args.seq_len}_ll{args.pred_len}_"
        f"k{args.k_periods}_dm{args.d_model}_dr{args.dropout}_"
        f"gdd{args.gdd_reduction}_fc{args.feature_type}"
    )
    result_content = (
        f"{dataset_name}_{args.seq_len}_{args.pred_len}_{model_config}\n"
        f"mse:{test_mse}, mae:{test_mae}\n\n"
    )
    with open(result_filepath, 'a', encoding='utf-8') as f:
        f.write(result_content)
    print(f"测试结果已保存到: {result_filepath}")


def save_model_info(model, save_directory):
    model_info = model.get_model_info()
    info_path = os.path.join(save_directory, "model_info.txt")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("PA-HyperLinear模型信息\n")
        f.write("=" * 50 + "\n")
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n")
    print(f"模型信息已保存到: {info_path}")


def infer_extension(dataset_name):
    if dataset_name.startswith('solar'):
        return 'txt'
    elif dataset_name.startswith('PEMS'):
        return 'npz'
    else:
        return 'csv'


def parse_args():
    dataset = "ETTh1"
    parser = argparse.ArgumentParser(description='PA-HyperLinear时间序列预测模型')

    parser.add_argument('--seed', type=int, default=2025, help='随机种子')
    parser.add_argument('--data', type=str, default=ROOT / f'./data/{dataset}.{infer_extension(dataset)}',
                        help='数据集路径')
    parser.add_argument('--feature_type', type=str, default='M', choices=['S', 'M', 'MS'], help='预测任务类型')
    parser.add_argument('--target', type=str, default='OT', help='目标特征')
    parser.add_argument('--checkpoint_dir', type=str, default=ROOT / 'checkpoints', help='模型检查点位置')
    parser.add_argument('--name', type=str, default=f'{dataset}', help='保存最佳模型到 checkpoints/name')
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=96, help='预测序列长度')
    parser.add_argument('--k_periods', type=int, default=3, help='周期数量')
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--gdd_reduction', type=int, default=2, help='GDD-MLP中间层压缩比例')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout率')
    parser.add_argument('--train_epochs', type=int, default=10, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='学习率')

    # --- 新增早停参数 ---
    parser.add_argument('--patience', type=int, default=5, help='早停的耐心轮次')
    # --------------------

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
