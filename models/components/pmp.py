# import torch
# import torch.nn as nn
# from typing import List
# from ..layers.fft_utils import FFT_for_Period, create_patches_from_periods, validate_periods
#
#
# class PMPModule(nn.Module):
#     """
#     PMP (Periodic Multi-scale Patching) 模块
#     基于FFT检测的周期进行多尺度数据分块
#     """
#
#     def __init__(self, k_periods: int = 3, min_patch_len: int = 4):
#         """
#         Args:
#             k_periods: 检测的主周期数量
#             min_patch_len: 最小patch长度限制
#         """
#         super(PMPModule, self).__init__()
#         self.k_periods = k_periods
#         self.min_patch_len = min_patch_len
#
#         # 存储检测到的周期信息
#         self.detected_periods = None
#         self.amplitudes = None
#
#     def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
#         """
#         前向传播：从完整时间序列到多尺度patch列表
#
#         Args:
#             x: 输入张量 [Batch, Seq_len, Channels]
#
#         Returns:
#             patch_list: k个不同尺度的patch张量列表
#                        每个元素形状为 [Batch, Num_patches_i, Patch_len_i, Channels]
#         """
#         batch_size, seq_len, n_channels = x.shape
#
#         # 1. 使用FFT检测最强的k个周期
#         periods, amplitudes = FFT_for_Period(x, k=self.k_periods)
#
#         # 2. 验证和调整周期的合理性
#         periods = validate_periods(periods, seq_len, self.min_patch_len)
#
#         # 3. 如果验证后周期数量不足k，用合理的默认值补充
#         if len(periods) < self.k_periods:
#             default_periods = self._generate_default_periods(seq_len, len(periods))
#             periods = list(periods) + default_periods
#             periods = periods[:self.k_periods]
#
#         # 4. 存储周期信息供后续分析使用
#         self.detected_periods = periods
#         self.amplitudes = amplitudes
#
#         # 5. 基于检测到的周期创建多尺度patch
#         patch_list = create_patches_from_periods(x, periods)
#
#         return patch_list
#
#     def _generate_default_periods(self, seq_len: int, existing_count: int) -> List[int]:
#         """
#         当检测到的周期数量不足时，生成合理的默认周期
#
#         Args:
#             seq_len: 输入序列长度
#             existing_count: 已有周期数量
#
#         Returns:
#             default_periods: 默认周期列表
#         """
#         needed_count = self.k_periods - existing_count
#         default_periods = []
#
#         # 生成一些标准的分割方式
#         candidates = [
#             seq_len // 8,  # 8个patch
#             seq_len // 16,  # 16个patch
#             seq_len // 32,  # 32个patch
#             seq_len // 4,  # 4个patch
#             seq_len // 64,  # 64个patch
#         ]
#
#         # 过滤掉太小的patch
#         candidates = [p for p in candidates if p >= self.min_patch_len]
#
#         # 选择前needed_count个作为默认周期
#         for i in range(min(needed_count, len(candidates))):
#             if candidates[i] not in self.detected_periods:
#                 default_periods.append(candidates[i])
#
#         print('周期尺度：',default_periods)
#
#         return default_periods
#
#     def get_period_info(self) -> dict:
#         """
#         获取检测到的周期信息，用于分析和调试
#
#         Returns:
#             period_info: 包含周期长度和振幅信息的字典
#         """
#         if self.detected_periods is None:
#             return {"periods": None, "amplitudes": None}
#
#         return {
#             "periods": self.detected_periods,
#             "amplitudes": self.amplitudes.detach().cpu().numpy() if self.amplitudes is not None else None
#         }
#
#     def extra_repr(self) -> str:
#         """模型信息字符串"""
#         return f'k_periods={self.k_periods}, min_patch_len={self.min_patch_len}'
# =================================================================================
# 文件: chlimiting/pahl/PAHL-542ce48f58f098e68cbbfca009fa68450a9c8bea/models/components/pmp.py
# (这是完整的、可以直接替换的最终代码)
# =================================================================================

import torch
import torch.nn as nn
from typing import List
from ..layers.fft_utils import FFT_for_Period, create_patches_from_periods, validate_periods


class PMPModule(nn.Module):
    """
    PMP (Periodic Multi-scale Patching) 模块
    基于FFT检测的周期进行多尺度数据分块
    """

    def __init__(self, k_periods: int = 3, min_patch_len: int = 4):
        """
        Args:
            k_periods: 检测的主周期数量
            min_patch_len: 最小patch长度限制
        """
        super(PMPModule, self).__init__()
        self.k_periods = k_periods
        self.min_patch_len = min_patch_len

        # 存储检测到的周期信息
        self.detected_periods = None
        self.amplitudes = None

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播：从完整时间序列到多尺度patch列表

        Args:
            x: 输入张量 [Batch, Seq_len, Channels]

        Returns:
            patch_list: k个不同尺度的patch张量列表
                       每个元素形状为 [Batch, Num_patches_i, Patch_len_i, Channels]
        """
        batch_size, seq_len, n_channels = x.shape

        # 1. 使用FFT检测最强的k个周期
        periods, amplitudes = FFT_for_Period(x, k=self.k_periods)

        # 2. 验证和调整周期的合理性
        periods = validate_periods(periods, seq_len, self.min_patch_len)

        # 3. 如果验证后周期数量不足k，用合理的默认值补充
        if len(periods) < self.k_periods:
            # 【修改点1】将已有的 periods 作为参数传入
            default_periods = self._generate_default_periods(seq_len, periods)
            periods = list(periods) + default_periods
            periods = periods[:self.k_periods]

        # 4. 存储周期信息供后续分析使用
        self.detected_periods = periods
        self.amplitudes = amplitudes

        # 5. 基于检测到的周期创建多尺度patch
        patch_list = create_patches_from_periods(x, periods)

        return patch_list

    # 【修改点2】修改函数签名，接收 existing_periods
    def _generate_default_periods(self, seq_len: int, existing_periods: list) -> List[int]:
        """
        当检测到的周期数量不足时，生成合理的默认周期

        Args:
            seq_len: 输入序列长度
            existing_periods: 已有周期列表

        Returns:
            default_periods: 默认周期列表
        """
        needed_count = self.k_periods - len(existing_periods)
        default_periods = []

        # 生成一些标准的分割方式
        candidates = [
            seq_len // 8,  # 8个patch
            seq_len // 16, # 16个patch
            seq_len // 32, # 32个patch
            seq_len // 4,  # 4个patch
            seq_len // 64, # 64个patch
        ]

        # 过滤掉太小的patch
        candidates = [p for p in candidates if p >= self.min_patch_len]

        # 选择前needed_count个作为默认周期
        for candidate in candidates:
            if len(default_periods) >= needed_count:
                break
            # 【修改点3】使用传入的 existing_periods 进行判断
            if candidate not in existing_periods and candidate not in default_periods:
                default_periods.append(candidate)

        print('补充的周期尺度：', default_periods)

        return default_periods

    def get_period_info(self) -> dict:
        """
        获取检测到的周期信息，用于分析和调试

        Returns:
            period_info: 包含周期长度和振幅信息的字典
        """
        if self.detected_periods is None:
            return {"periods": None, "amplitudes": None}

        return {
            "periods": self.detected_periods,
            "amplitudes": self.amplitudes.detach().cpu().numpy() if self.amplitudes is not None else None
        }

    def extra_repr(self) -> str:
        """模型信息字符串"""
        return f'k_periods={self.k_periods}, min_patch_len={self.min_patch_len}'
