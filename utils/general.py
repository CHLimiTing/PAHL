import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
# # import random
# # import numpy as np
# # import torch
# # import torch.backends.cudnn as cudnn
#
# # general.py (Updated Version)
#
# import random
# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import matplotlib.pyplot as plt
#
#
# def set_seed(seed=0):
#     """
#     设置随机种子以确保实验可复现。
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     cudnn.benchmark, cudnn.deterministic = (False, True)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
#
#
# def get_metrics(pred, true):
#     """
#     计算预测结果的MAE和MSE。
#
#     Args:
#         pred (np.array): 预测值
#         true (np.array): 真实值
#
#     Returns:
#         tuple: (mae, mse)
#     """
#     mae = np.mean(np.abs(pred - true))
#     mse = np.mean((pred - true) ** 2)
#     return mae, mse
#
#
# def plot_results(true, pred, path):
#     """
#     将真实值和预测值进行可视化，并保存为图片。
#
#     Args:
#         true (np.array): 真实值
#         pred (np.array): 预测值
#         path (str): 图片保存路径
#     """
#     # 为了清晰，只绘制第一个通道（变量）的对比图
#     plt.figure(figsize=(15, 5))
#     plt.plot(true[:, 0], label='GroundTruth')
#     plt.plot(pred[:, 0], label='Prediction')
#     plt.legend()
#     plt.title('Prediction vs. GroundTruth (First Channel)')
#     plt.savefig(path)
#     plt.close()
#     print(f"Prediction plot saved to {path}")