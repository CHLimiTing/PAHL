import torch
import torch.nn as nn


class GDDMLPModule(nn.Module):
    """
    GDD-MLP (Global Dynamic Dimension MLP) 模块
    在每个尺度内学习通道间的交互关系并生成修正项
    """

    def __init__(self, n_vars: int, reduction: int = 2, dropout: float = 0.1,
                 avg_flag: bool = True, max_flag: bool = True):
        """
        Args:
            n_vars: 通道数量 (变量数量)
            reduction: MLP中间层的压缩比例
            dropout: dropout比例
            avg_flag: 是否使用平均池化
            max_flag: 是否使用最大池化
        """
        super(GDDMLPModule, self).__init__()
        self.n_vars = n_vars
        self.avg_flag = avg_flag
        self.max_flag = max_flag

        # 自适应池化层
        self.avg_pool = nn.AdaptiveAvgPool1d(1) if avg_flag else None
        self.max_pool = nn.AdaptiveMaxPool1d(1) if max_flag else None

        # 用于生成scale修正项的MLP
        self.fc_scale = nn.Sequential(
            nn.Linear(n_vars, n_vars // reduction, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_vars // reduction, n_vars, bias=False)
        )

        # 用于生成shift修正项的MLP
        self.fc_shift = nn.Sequential(
            nn.Linear(n_vars, n_vars // reduction, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_vars // reduction, n_vars, bias=False)
        )

        # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     前向传播：通道交互与特征优化
    #
    #     Args:
    #         x: 输入张量 [Batch, Num_patches, Patch_len, Channels]
    #
    #     Returns:
    #         output: 经过通道交互优化的张量 [Batch, Num_patches, Patch_len, Channels]
    #     """
    #     b, n, p, d = x.shape  # batch, num_patches, patch_len, channels
    #
    #     # 初始化scale和shift修正项
    #     scale = torch.zeros_like(x)
    #     shift = torch.zeros_like(x)
    #
    #     # 平均池化分支
    #     if self.avg_flag and self.avg_pool is not None:
    #         # 重塑为 [b*n, p, d]
    #         x_reshaped = x.reshape(b * n, p, d)
    #         # 池化得到 [b*n, 1, d]
    #         pooled_avg = self.avg_pool(x_reshaped)
    #         # 重塑为 [b, n, d]
    #         pooled_avg = pooled_avg.reshape(b, n, d)
    #         # 转置为 [b, d, n] 用于MLP处理
    #         pooled_avg = pooled_avg.permute(0, 2, 1)
    #
    #         # MLP处理: [b, d, n]
    #         scale_contrib = self.fc_scale(pooled_avg)
    #         shift_contrib = self.fc_shift(pooled_avg)
    #
    #         # 转回原始维度: [b, d, n] -> [b, n, 1, d]
    #         scale_contrib = scale_contrib.permute(0, 2, 1).unsqueeze(2)
    #         shift_contrib = shift_contrib.permute(0, 2, 1).unsqueeze(2)
    #
    #         scale += scale_contrib
    #         shift += shift_contrib
    #
    #     # 最大池化分支
    #     if self.max_flag and self.max_pool is not None:
    #         # 与平均池化类似的处理流程
    #         x_reshaped = x.reshape(b * n, p, d)
    #         pooled_max = self.max_pool(x_reshaped)
    #         pooled_max = pooled_max.reshape(b, n, d)
    #         pooled_max = pooled_max.permute(0, 2, 1)
    #
    #         scale_contrib = self.fc_scale(pooled_max)
    #         shift_contrib = self.fc_shift(pooled_max)
    #
    #         scale_contrib = scale_contrib.permute(0, 2, 1).unsqueeze(2)
    #         shift_contrib = shift_contrib.permute(0, 2, 1).unsqueeze(2)
    #
    #         scale += scale_contrib
    #         shift += shift_contrib
    #
    #     # 应用sigmoid激活并进行残差连接
    #     output = self.sigmoid(scale) * x + self.sigmoid(shift)
    #
    #     return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：通道交互与特征优化

        Args:
            x: 输入张量 [Batch, Num_patches, Patch_len, Channels]

        Returns:
            output: 经过通道交互优化的张量 [Batch, Num_patches, Patch_len, Channels]
        """
        b, n, p, d = x.shape  # batch, num_patches, patch_len, channels

        # 初始化scale和shift修正项
        scale = torch.zeros((b, n, 1, d), device=x.device)  # 修正初始化形状以匹配广播
        shift = torch.zeros((b, n, 1, d), device=x.device)  # 修正初始化形状以匹配广播

        pooled_features = []

        # 统一处理输入以适应Pool1d
        # 原始: [b, n, p, d] -> permute -> [b, n, d, p] -> reshape -> [b*n, d, p]
        x_for_pool = x.permute(0, 1, 3, 2).reshape(b * n, d, p)

        # 平均池化分支
        if self.avg_flag and self.avg_pool is not None:
            # 在 patch_len (p) 维度上池化: [b*n, d, p] -> [b*n, d, 1]
            pooled_avg = self.avg_pool(x_for_pool).squeeze(-1)  # -> [b*n, d]
            pooled_features.append(pooled_avg)

        # 最大池化分支
        if self.max_flag and self.max_pool is not None:
            # 在 patch_len (p) 维度上池化: [b*n, d, p] -> [b*n, d, 1]
            pooled_max = self.max_pool(x_for_pool).squeeze(-1)  # -> [b*n, d]
            pooled_features.append(pooled_max)

        for pooled in pooled_features:
            # 重塑以适应MLP: [b*n, d] -> [b, n, d]
            pooled = pooled.reshape(b, n, d)

            # MLP处理，现在fc层直接在最后一个维度(d)上操作
            scale_contrib = self.fc_scale(pooled)  # -> [b, n, d]
            shift_contrib = self.fc_shift(pooled)  # -> [b, n, d]

            # 增加一个维度以进行广播: [b, n, d] -> [b, n, 1, d]
            scale += scale_contrib.unsqueeze(2)
            shift += shift_contrib.unsqueeze(2)

        # 应用sigmoid激活并进行残差连接
        # (sigmoid(scale)的[b, n, 1, d])与x的[b, n, p, d]进行广播相乘
        output = self.sigmoid(scale) * x + self.sigmoid(shift)

        return output

    def extra_repr(self) -> str:
        """模型信息字符串"""
        return (f'n_vars={self.n_vars}, avg_flag={self.avg_flag}, '
                f'max_flag={self.max_flag}')


class MultiScaleGDDMLP(nn.Module):
    """
    多尺度GDD-MLP模块集合
    为k个不同尺度分别创建独立的GDD-MLP实例
    """

    def __init__(self, n_vars: int, k_scales: int = 3, reduction: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            n_vars: 通道数量
            k_scales: 尺度数量
            reduction: MLP压缩比例
            dropout: dropout比例
        """
        super(MultiScaleGDDMLP, self).__init__()
        self.k_scales = k_scales

        # 为每个尺度创建独立的GDD-MLP模块
        self.gdd_mlp_layers = nn.ModuleList([
            GDDMLPModule(n_vars, reduction, dropout)
            for _ in range(k_scales)
        ])

    def forward(self, patch_list: list) -> list:
        """
        前向传播：处理多个尺度的patch

        Args:
            patch_list: k个尺度的patch张量列表

        Returns:
            F_channel_aware: k个经过通道交互的张量列表
        """
        F_channel_aware = []

        for i, patches in enumerate(patch_list):
            # 每个尺度使用独立的GDD-MLP模块
            processed_patches = self.gdd_mlp_layers[i](patches)
            F_channel_aware.append(processed_patches)

        return F_channel_aware

    def extra_repr(self) -> str:
        """模型信息字符串"""
        return f'k_scales={self.k_scales}'
# import torch
# import torch.nn as nn
#
#
# class GDDMLPModule(nn.Module):
#     """
#     改进的GDD-MLP模块
#     基于原始设计理念，修复维度处理问题，增强通道间关系学习
#     """
#
#     def __init__(self, n_vars: int, reduction: int = 2, dropout: float = 0.1,
#                  avg_flag: bool = True, max_flag: bool = True):
#         super(GDDMLPModule, self).__init__()
#         self.n_vars = n_vars
#         self.avg_flag = avg_flag
#         self.max_flag = max_flag
#
#         # 自适应池化层
#         self.avg_pool = nn.AdaptiveAvgPool1d(1) if avg_flag else None
#         self.max_pool = nn.AdaptiveMaxPool1d(1) if max_flag else None
#
#         # 【改进1】：分别为scale和shift设计独立的MLP，增强表达能力
#         self.fc_scale = nn.Sequential(
#             nn.Linear(n_vars, n_vars // reduction, bias=False),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(n_vars // reduction, n_vars, bias=False)
#         )
#
#         self.fc_shift = nn.Sequential(
#             nn.Linear(n_vars, n_vars // reduction, bias=False),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(n_vars // reduction, n_vars, bias=False)
#         )
#
#         # 【改进2】：使用更合适的激活函数组合
#         self.sigmoid = nn.Sigmoid()
#
#         # 【改进3】：添加可学习的权重平衡参数
#         self.alpha = nn.Parameter(torch.ones(1))
#         self.beta = nn.Parameter(torch.ones(1))
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         改进的前向传播：修复原始维度处理问题
#
#         Args:
#             x: [Batch, Num_patches, Patch_len, Channels]
#         Returns:
#             output: [Batch, Num_patches, Patch_len, Channels]
#         """
#         b, n, p, d = x.shape
#
#         # 初始化累积器
#         scale_total = torch.zeros(b, n, d, device=x.device, dtype=x.dtype)
#         shift_total = torch.zeros(b, n, d, device=x.device, dtype=x.dtype)
#
#         # 【修复关键问题】：正确处理维度，保持通道信息
#         # Step 1: 重塑为池化格式 [b, n, p, d] -> [b*n, p, d] -> [b*n, d, p]
#         x_reshaped = x.reshape(b * n, p, d).transpose(1, 2)  # [b*n, d, p]
#
#         # 平均池化分支
#         if self.avg_flag and self.avg_pool is not None:
#             # Step 2: 池化 [b*n, d, p] -> [b*n, d, 1] -> [b*n, d]
#             pooled_avg = self.avg_pool(x_reshaped).squeeze(-1)  # [b*n, d]
#
#             # Step 3: 恢复batch和patch维度 [b*n, d] -> [b, n, d]
#             pooled_avg = pooled_avg.reshape(b, n, d)  # [b, n, d]
#
#             # Step 4: 转置以让MLP作用在通道维度 [b, n, d] -> [b, d, n]
#             pooled_avg = pooled_avg.transpose(1, 2)  # [b, d, n]
#
#             # Step 5: MLP处理 - 学习不同patch对每个通道的重要性
#             scale_contrib = self.fc_scale(pooled_avg)  # [b, d, n]
#             shift_contrib = self.fc_shift(pooled_avg)  # [b, d, n]
#
#             # Step 6: 转回原维度 [b, d, n] -> [b, n, d]
#             scale_total += scale_contrib.transpose(1, 2)
#             shift_total += shift_contrib.transpose(1, 2)
#
#         # 最大池化分支
#         if self.max_flag and self.max_pool is not None:
#             pooled_max = self.max_pool(x_reshaped).squeeze(-1)
#             pooled_max = pooled_max.reshape(b, n, d)
#             pooled_max = pooled_max.transpose(1, 2)
#
#             scale_contrib = self.fc_scale(pooled_max)
#             shift_contrib = self.fc_shift(pooled_max)
#
#             scale_total += scale_contrib.transpose(1, 2)
#             shift_total += shift_contrib.transpose(1, 2)
#
#         # 【改进4】：更精细的特征调制
#         # 扩展维度以进行广播 [b, n, d] -> [b, n, 1, d]
#         scale_expanded = scale_total.unsqueeze(2)  # [b, n, 1, d]
#         shift_expanded = shift_total.unsqueeze(2)  # [b, n, 1, d]
#
#         # 应用可学习权重和sigmoid激活
#         scale_activated = self.sigmoid(self.alpha * scale_expanded)
#         shift_activated = self.sigmoid(self.beta * shift_expanded)
#
#         # 【改进5】：增强的残差连接
#         output = scale_activated * x + shift_activated
#
#         return output
#
#     def extra_repr(self) -> str:
#         return (f'n_vars={self.n_vars}, avg_flag={self.avg_flag}, '
#                 f'max_flag={self.max_flag}')
#
#
# class EnhancedGDDMLPModule(nn.Module):
#     """
#     进一步增强的GDD-MLP模块
#     增加更多设计改进，提升性能
#     """
#
#     def __init__(self, n_vars: int, reduction: int = 2, dropout: float = 0.1,
#                  avg_flag: bool = True, max_flag: bool = True,
#                  use_layer_norm: bool = True):
#         super(EnhancedGDDMLPModule, self).__init__()
#         self.n_vars = n_vars
#         self.avg_flag = avg_flag
#         self.max_flag = max_flag
#         self.use_layer_norm = use_layer_norm
#
#         # 池化层
#         self.avg_pool = nn.AdaptiveAvgPool1d(1) if avg_flag else None
#         self.max_pool = nn.AdaptiveMaxPool1d(1) if max_flag else None
#
#         # 【增强1】：更深的MLP网络
#         hidden_dim = max(n_vars // reduction, 8)  # 确保最小维度
#
#         self.fc_scale = nn.Sequential(
#             nn.Linear(n_vars, hidden_dim, bias=False),
#             nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, n_vars, bias=False)
#         )
#
#         self.fc_shift = nn.Sequential(
#             nn.Linear(n_vars, hidden_dim, bias=False),
#             nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, n_vars, bias=False)
#         )
#
#         # 【增强2】：多种激活函数组合
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()
#
#         # 【增强3】：可学习的融合权重
#         self.fusion_weight = nn.Parameter(torch.tensor(0.5))
#         self.scale_weight = nn.Parameter(torch.ones(1))
#         self.shift_weight = nn.Parameter(torch.ones(1))
#
#         # 【增强4】：通道注意力机制
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Conv2d(n_vars, n_vars // reduction, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(n_vars // reduction, n_vars, 1, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         b, n, p, d = x.shape
#
#         # 基础的GDD-MLP处理（与上面相同）
#         scale_total = torch.zeros(b, n, d, device=x.device, dtype=x.dtype)
#         shift_total = torch.zeros(b, n, d, device=x.device, dtype=x.dtype)
#
#         x_reshaped = x.reshape(b * n, p, d).transpose(1, 2)
#
#         # 池化分支处理
#         if self.avg_flag and self.avg_pool is not None:
#             pooled_avg = self.avg_pool(x_reshaped).squeeze(-1).reshape(b, n, d)
#             pooled_avg = pooled_avg.transpose(1, 2)
#
#             scale_contrib = self.fc_scale(pooled_avg)
#             shift_contrib = self.fc_shift(pooled_avg)
#
#             scale_total += scale_contrib.transpose(1, 2)
#             shift_total += shift_contrib.transpose(1, 2)
#
#         if self.max_flag and self.max_pool is not None:
#             pooled_max = self.max_pool(x_reshaped).squeeze(-1).reshape(b, n, d)
#             pooled_max = pooled_max.transpose(1, 2)
#
#             scale_contrib = self.fc_scale(pooled_max)
#             shift_contrib = self.fc_shift(pooled_max)
#
#             scale_total += scale_contrib.transpose(1, 2)
#             shift_total += shift_contrib.transpose(1, 2)
#
#         # 【增强处理】
#         scale_expanded = scale_total.unsqueeze(2)
#         shift_expanded = shift_total.unsqueeze(2)
#
#         # 混合激活函数
#         scale_sig = self.sigmoid(self.scale_weight * scale_expanded)
#         shift_tanh = self.tanh(self.shift_weight * shift_expanded)
#
#         # 通道注意力
#         # [b, n, p, d] -> [b, d, n, p] 以符合Conv2d要求
#         x_for_attn = x.permute(0, 3, 1, 2)
#         channel_weights = self.channel_attention(x_for_attn)  # [b, d, 1, 1]
#         channel_weights = channel_weights.permute(0, 2, 3, 1)  # [b, 1, 1, d]
#
#         # 最终输出融合
#         gdd_output = scale_sig * x + shift_tanh
#         attended_output = channel_weights * gdd_output
#
#         # 加权融合原始输入和处理后的输出
#         output = self.fusion_weight * attended_output + (1 - self.fusion_weight) * x
#
#         return output
#
#
# class MultiScaleGDDMLP(nn.Module):
#     """
#     多尺度GDD-MLP模块 - 保持接口兼容
#     """
#
#     def __init__(self, n_vars: int, k_scales: int = 3, reduction: int = 2,
#                  dropout: float = 0.1, use_enhanced: bool = False):
#         super(MultiScaleGDDMLP, self).__init__()
#         self.k_scales = k_scales
#         self.use_enhanced = use_enhanced
#
#         # 可选择使用基础版本或增强版本
#         if use_enhanced:
#             self.gdd_mlp_layers = nn.ModuleList([
#                 EnhancedGDDMLPModule(n_vars, reduction, dropout)
#                 for _ in range(k_scales)
#             ])
#         else:
#             self.gdd_mlp_layers = nn.ModuleList([
#                 GDDMLPModule(n_vars, reduction, dropout)
#                 for _ in range(k_scales)
#             ])
#
#     def forward(self, patch_list: list) -> list:
#         """保持原始接口"""
#         F_channel_aware = []
#         for i, patches in enumerate(patch_list):
#             processed_patches = self.gdd_mlp_layers[i](patches)
#             F_channel_aware.append(processed_patches)
#         return F_channel_aware
#
#     def extra_repr(self) -> str:
#         return f'k_scales={self.k_scales}, use_enhanced={self.use_enhanced}'
