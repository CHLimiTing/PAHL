import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


def FFT_for_Period(x: torch.Tensor, k: int = 3) -> Tuple[np.ndarray, torch.Tensor]:
    """
    使用FFT检测时间序列中最强的k个周期

    Args:
        x: 输入张量 [Batch, Seq_len, Channels]
        k: 检测的主周期数量

    Returns:
        periods: 检测到的k个周期长度
        amplitudes: 对应的振幅信息
    """
    # 执行FFT变换
    xf = torch.fft.rfft(x, dim=1)

    # 计算多通道融合的振幅谱 (平均所有通道)
    frequency_list = abs(xf).mean(0).mean(-1)  # [Freq_bins]
    frequency_list[0] = 0  # 去除直流分量

    # 找到最强的k个频率
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()

    # 将频率转换为周期长度
    periods = x.shape[1] // top_list

    # 返回振幅信息用于后续分析
    amplitudes = abs(xf).mean(-1)[:, top_list]  # [Batch, k]

    return periods, amplitudes


def create_patches_from_periods(x: torch.Tensor, periods: np.ndarray) -> List[torch.Tensor]:
    """
    基于检测到的周期创建多尺度patch

    Args:
        x: 输入张量 [Batch, Seq_len, Channels]
        periods: 周期长度数组

    Returns:
        patch_list: k个不同尺度的patch张量列表
                   每个元素形状为 [Batch, Num_patches_i, Patch_len_i, Channels]
    """
    batch_size, seq_len, n_channels = x.shape
    patch_list = []

    for period in periods:
        patch_len = int(period)

        # 确保patch_len不超过序列长度
        patch_len = min(patch_len, seq_len)

        # 计算可以创建的patch数量
        num_patches = seq_len // patch_len

        # 如果无法整除，截断多余部分
        if num_patches == 0:
            num_patches = 1
            patch_len = seq_len

        # 截断输入序列到可整除的长度
        effective_len = num_patches * patch_len
        x_truncated = x[:, :effective_len, :]  # [Batch, Effective_len, Channels]

        # 重塑为patch格式
        patches = x_truncated.view(batch_size, num_patches, patch_len, n_channels)

        patch_list.append(patches)

    return patch_list


def validate_periods(periods: np.ndarray, seq_len: int, min_patch_len: int = 4) -> np.ndarray:
    """
    验证和调整检测到的周期，确保其合理性

    Args:
        periods: 原始周期数组
        seq_len: 输入序列长度
        min_patch_len: 最小patch长度

    Returns:
        validated_periods: 验证后的周期数组
    """
    validated_periods = []

    for period in periods:
        # 确保周期长度合理
        period = max(min_patch_len, min(period, seq_len // 2))
        validated_periods.append(period)

    # 去除重复的周期并排序
    validated_periods = sorted(list(set(validated_periods)))

    return np.array(validated_periods)