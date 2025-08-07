import torch
import torch.nn as nn
from ..layers.revin import PatchRevIN


class RLinearBlock(nn.Module):
    """
    RLinear主干网络模块
    实现RevIN norm -> Linear -> RevIN denorm的核心流程
    """

    def __init__(self, patch_len: int, d_model: int, n_vars: int):
        """
        Args:
            patch_len: 输入patch的长度
            d_model: 输出的隐藏维度
            n_vars: 通道数量
        """
        super(RLinearBlock, self).__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.n_vars = n_vars

        # 可逆实例归一化
        self.revin = PatchRevIN(n_vars)

        # 核心线性映射层 (通道独立)
        self.linear = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：时序特征提取

        Args:
            x: 输入张量 [Batch, Num_patches, Patch_len, Channels]

        Returns:
            output: 输出张量 [Batch, Num_patches, d_model, Channels]
        """
        # 1. RevIN 归一化
        x_norm = self.revin(x, mode='norm')

        # 2. 线性映射 patch_len -> d_model
        # [B, N, P, C] -> [B, N, d_model, C]
        x_mapped = self.linear(x_norm)

        # 3. RevIN 反归一化
        output = self.revin(x_mapped, mode='denorm')

        return output

    def extra_repr(self) -> str:
        """模型信息字符串"""
        return f'patch_len={self.patch_len}, d_model={self.d_model}, n_vars={self.n_vars}'


class MultiScaleRLinear(nn.Module):
    """
    多尺度RLinear模块集合
    为k个不同尺度分别创建独立的RLinear实例，并包含全局平均池化
    """

    def __init__(self, patch_lengths: list, d_model: int, n_vars: int):
        """
        Args:
            patch_lengths: k个不同尺度的patch长度列表
            d_model: 统一的隐藏维度
            n_vars: 通道数量
        """
        super(MultiScaleRLinear, self).__init__()
        self.d_model = d_model
        self.n_vars = n_vars
        self.k_scales = len(patch_lengths)

        # 为每个尺度创建独立的RLinear块
        self.rlinear_blocks = nn.ModuleList([
            RLinearBlock(patch_len, d_model, n_vars)
            for patch_len in patch_lengths
        ])

        # 全局平均池化层
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, F_channel_aware: list) -> tuple:
        """
        前向传播：多尺度时序特征提取与汇总

        Args:
            F_channel_aware: k个尺度的张量列表
                           每个元素: [Batch, Num_patches_i, Patch_len_i, Channels]

        Returns:
            F_temporal: k个时序特征张量列表 [Batch, Num_patches_i, d_model, Channels]
            F_summarized: k个概要特征张量列表 [Batch, d_model, Channels]
        """
        F_temporal = []
        F_summarized = []

        for i, feature in enumerate(F_channel_aware):
            # 1. 通过RLinear块提取时序特征
            temporal_feature = self.rlinear_blocks[i](feature)
            F_temporal.append(temporal_feature)

            # 2. 全局平均池化：在Num_patches维度上汇总
            # [B, N, d_model, C] -> [B, d_model, C]
            batch_size, num_patches, d_model, channels = temporal_feature.shape

            # 重塑以便进行池化: [B, N, d_model, C] -> [B*C, d_model, N]
            temporal_reshaped = temporal_feature.permute(0, 3, 2, 1).contiguous()
            temporal_reshaped = temporal_reshaped.view(batch_size * channels, d_model, num_patches)

            # 执行全局平均池化: [B*C, d_model, N] -> [B*C, d_model, 1]
            pooled = self.global_pool(temporal_reshaped).squeeze(-1)

            # 重塑回原始维度: [B*C, d_model] -> [B, d_model, C]
            summarized = pooled.view(batch_size, channels, d_model).transpose(1, 2)

            F_summarized.append(summarized)

        return F_temporal, F_summarized

    def extra_repr(self) -> str:
        """模型信息字符串"""
        return f'k_scales={self.k_scales}, d_model={self.d_model}, n_vars={self.n_vars}'


def create_multiscale_rlinear(periods: list, d_model: int, n_vars: int) -> MultiScaleRLinear:
    """
    工厂函数：基于检测到的周期创建多尺度RLinear模块

    Args:
        periods: 检测到的周期长度列表
        d_model: 隐藏维度
        n_vars: 通道数量

    Returns:
        MultiScaleRLinear实例
    """
    return MultiScaleRLinear(periods, d_model, n_vars)