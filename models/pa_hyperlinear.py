# import torch
# import torch.nn as nn
# from typing import List, Tuple
#
# from .components.pmp import PMPModule
# from .components.gdd_mlp import MultiScaleGDDMLP
# from .components.rlinear_block import MultiScaleRLinear, create_multiscale_rlinear
# from .components.dual_ams import DualAMSModule
#
#
# class PAHyperLinear(nn.Module):
#     """
#     PA-HyperLinear: 基于周期感知的混合线性时间序列预测模型
#
#     架构流程:
#     输入 -> PMP(周期检测与分块) -> GDD-MLP(通道交互) -> RLinear(时序建模) -> 双输入AMS(专家融合) -> 输出
#     """
#
#     def __init__(self, input_shape: Tuple[int, int], pred_len: int,
#                  k_periods: int = 3, d_model: int = 128, dropout: float = 0.1,
#                  gdd_reduction: int = 2, target_slice: slice = None):
#         """
#         Args:
#             input_shape: 输入形状 (seq_len, n_features)
#             pred_len: 预测长度
#             k_periods: 检测的主周期数量
#             d_model: RLinear的隐藏维度 (推荐128/256/512)
#             dropout: dropout比例
#             gdd_reduction: GDD-MLP的压缩比例
#             target_slice: 目标特征切片 (用于多变量预测单变量任务)
#         """
#         super(PAHyperLinear, self).__init__()
#
#         self.seq_len, self.n_features = input_shape
#         self.pred_len = pred_len
#         self.k_periods = k_periods
#         self.d_model = d_model
#         self.target_slice = target_slice
#
#         # 1. PMP模块：周期检测与多尺度分块
#         self.pmp_module = PMPModule(k_periods=k_periods, min_patch_len=4)
#
#         # 2. GDD-MLP模块：多尺度通道交互 (k个独立实例)
#         self.gdd_mlp_module = MultiScaleGDDMLP(
#             n_vars=self.n_features,
#             k_scales=k_periods,
#             reduction=gdd_reduction,
#             dropout=dropout
#         )
#
#         # 3. RLinear模块：延迟初始化 (需要知道具体的patch长度)
#         self.rlinear_module = None
#
#         # 4. 双输入AMS模块：自适应专家混合
#         self.dual_ams_module = DualAMSModule(
#             d_model=d_model,
#             pred_len=pred_len,
#             n_vars=self.n_features,
#             num_experts=k_periods,
#             dropout=dropout
#         )
#
#         # 存储检测到的周期信息
#         self.detected_periods = None
#
#     def _initialize_rlinear(self, periods: List[int]):
#         """
#         基于检测到的周期初始化RLinear模块
#
#         Args:
#             periods: 检测到的周期长度列表
#         """
#         if self.rlinear_module is None:
#             self.rlinear_module = create_multiscale_rlinear(
#                 periods=periods,
#                 d_model=self.d_model,
#                 n_vars=self.n_features
#             )
#             # 将模块移动到正确的设备
#             if next(self.parameters()).is_cuda:
#                 self.rlinear_module = self.rlinear_module.cuda()
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#
#         print(f"模型输入形状: {x.shape}")
#
#         # 1. PMP模块：周期检测与多尺度分块
#         patch_list = self.pmp_module(x)
#         print(f"PMP输出patch数量: {len(patch_list)}")
#         for i, patch in enumerate(patch_list):
#             print(f"  Patch {i} 形状: {patch.shape}")
#
#         # 2. GDD-MLP模块前
#         F_channel_aware = self.gdd_mlp_module(patch_list)
#         print(f"GDD-MLP输出数量: {len(F_channel_aware)}")
#         for i, feature in enumerate(F_channel_aware):
#             print(f"  Feature {i} 形状: {feature.shape}")
#
#
#         """
#         前向传播
#
#         Args:
#             x: 输入张量 [Batch, Seq_len, Features]
#
#         Returns:
#             predictions: 预测结果 [Batch, pred_len, Features]
#             aux_loss: 辅助损失 (负载均衡损失)
#         """
#         # 1. PMP模块：周期检测与多尺度分块
#         patch_list = self.pmp_module(x)
#
#         # 获取检测到的周期信息并初始化RLinear模块
#         period_info = self.pmp_module.get_period_info()
#         self.detected_periods = period_info["periods"]
#
#         if self.rlinear_module is None:
#             self._initialize_rlinear(self.detected_periods)
#
#         # 2. GDD-MLP模块：k个尺度的通道交互
#         F_channel_aware = self.gdd_mlp_module(patch_list)
#
#         # 3. RLinear模块：多尺度时序特征提取
#         F_temporal, F_summarized = self.rlinear_module(F_channel_aware)
#
#         # 4. 双输入AMS模块：自适应专家混合
#         predictions = self.dual_ams_module(F_channel_aware, F_summarized)
#
#         # 5. 目标切片处理 (如果是MS任务)
#         if self.target_slice is not None:
#             predictions = predictions[:, :, self.target_slice]
#
#         # 计算辅助损失 (这里简化为0，可以根据需要添加负载均衡损失等)
#         aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
#
#         return predictions, aux_loss
#
#     def get_model_info(self) -> dict:
#         """
#         获取模型信息，用于调试和分析
#
#         Returns:
#             model_info: 包含模型各组件信息的字典
#         """
#         info = {
#             "architecture": "PA-HyperLinear",
#             "input_shape": (self.seq_len, self.n_features),
#             "pred_len": self.pred_len,
#             "k_periods": self.k_periods,
#             "d_model": self.d_model,
#             "detected_periods": self.detected_periods,
#             "total_parameters": sum(p.numel() for p in self.parameters()),
#             "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
#         }
#
#         # 添加周期检测信息
#         if hasattr(self.pmp_module, 'get_period_info'):
#             period_info = self.pmp_module.get_period_info()
#             info.update(period_info)
#
#         return info
#
#     def extra_repr(self) -> str:
#         """模型信息字符串"""
#         return (f'input_shape=({self.seq_len}, {self.n_features}), '
#                 f'pred_len={self.pred_len}, k_periods={self.k_periods}, '
#                 f'd_model={self.d_model}')
#
#
# # 为了兼容原有的AMD接口，提供一个包装函数
# def create_pa_hyperlinear_model(input_shape: Tuple[int, int], pred_len: int,
#                                 dropout: float = 0.1, target_slice: slice = None,
#                                 **kwargs) -> PAHyperLinear:
#     """
#     工厂函数：创建PA-HyperLinear模型实例
#     兼容原AMD项目的参数接口
#
#     Args:
#         input_shape: 输入形状 (seq_len, n_features)
#         pred_len: 预测长度
#         dropout: dropout比例
#         target_slice: 目标特征切片
#         **kwargs: 其他参数
#
#     Returns:
#         PA-HyperLinear模型实例
#     """
#     # 从kwargs中提取PA-HyperLinear特有的参数
#     k_periods = kwargs.get('k_periods', 3)
#     d_model = kwargs.get('d_model', 128)
#     gdd_reduction = kwargs.get('gdd_reduction', 2)
#
#     return PAHyperLinear(
#         input_shape=input_shape,
#         pred_len=pred_len,
#         k_periods=k_periods,
#         d_model=d_model,
#         dropout=dropout,
#         gdd_reduction=gdd_reduction,
#         target_slice=target_slice
#     )
# =================================================================================
# 文件: chlimiting/pahl/PAHL-542ce48f58f098e68cbbfca009fa68450a9c8bea/models/pa_hyperlinear.py
# (这是完整的、可以直接替换的最终代码)
# =================================================================================

import torch
import torch.nn as nn
from typing import List, Tuple

from .components.pmp import PMPModule
from .components.gdd_mlp import MultiScaleGDDMLP
from .components.rlinear_block import MultiScaleRLinear, create_multiscale_rlinear
from .components.dual_ams import DualAMSModule
# 确保这个导入存在
from .layers.fft_utils import create_patches_from_periods


class PAHyperLinear(nn.Module):
    """
    PA-HyperLinear: 基于周期感知的混合线性时间序列预测模型
    架构流程:
    输入 -> PMP(周期检测与分块) -> GDD-MLP(通道交互) -> RLinear(时序建模) -> 双输入AMS(专家融合) -> 输出
    """

    def __init__(self, input_shape: Tuple[int, int], pred_len: int,
                 k_periods: int = 3, d_model: int = 128, dropout: float = 0.1,
                 gdd_reduction: int = 2, target_slice: slice = None):
        """
        Args:
            input_shape: 输入形状 (seq_len, n_features)
            pred_len: 预测长度
            k_periods: 检测的主周期数量
            d_model: RLinear的隐藏维度 (推荐128/256/512)
            dropout: dropout比例
            gdd_reduction: GDD-MLP的压缩比例
            target_slice: 目标特征切片 (用于多变量预测单变量任务)
        """
        super(PAHyperLinear, self).__init__()

        self.seq_len, self.n_features = input_shape
        self.pred_len = pred_len
        self.k_periods = k_periods
        self.d_model = d_model
        self.target_slice = target_slice

        # PMP模块保持不变
        self.pmp_module = PMPModule(k_periods=k_periods, min_patch_len=4)

        # GDD-MLP模块保持不变
        self.gdd_mlp_module = MultiScaleGDDMLP(
            n_vars=self.n_features,
            k_scales=k_periods,
            reduction=gdd_reduction,
            dropout=dropout
        )

        # RLinear模块仍然延迟初始化
        self.rlinear_module = None

        # AMS模块保持不变
        self.dual_ams_module = DualAMSModule(
            d_model=d_model,
            pred_len=pred_len,
            n_vars=self.n_features,
            num_experts=k_periods,
            dropout=dropout
        )

        # 【修改点1】: 使用 register_buffer 替代普通属性
        # 这不是一个新模块，而是PyTorch的一个标准功能，用于让模型“记住”一个状态
        # 初始化为0，作为“尚未确定周期”的标志
        self.register_buffer('fixed_periods', torch.tensor([0], dtype=torch.long))

    def _initialize_rlinear(self, periods: List[int]):
        """ 这个辅助函数保持不变 """
        if self.rlinear_module is None:
            self.rlinear_module = create_multiscale_rlinear(
                periods=periods,
                d_model=self.d_model,
                n_vars=self.n_features
            )
            if next(self.parameters()).is_cuda:
                self.rlinear_module = self.rlinear_module.cuda()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        【修改点2】: 修改 forward 方法的内部逻辑
        """
        # 判断是否需要进行“一次性动态识别”
        # self.training 确保在评估/推理模式下，如果遇到不同形状的数据也能重新确定周期
        if self.fixed_periods[0] == 0 and self.training:
            # --- 这部分代码只在训练的第一个批次执行 ---
            # 1a. 运行PMP模块以执行FFT并获取周期
            with torch.no_grad():  # 在此阶段不计算梯度
                _ = self.pmp_module(x)

                # 1b. 从PMP模块中获取检测到的周期
            period_info = self.pmp_module.get_period_info()
            detected_p = period_info["periods"]

            # 1c. 将这组周期“锁定”并存入模型的缓冲区
            self.fixed_periods = torch.tensor(detected_p, dtype=torch.long, device=x.device)

            # 1d. 使用这组固定的周期来初始化RLinear模块
            self._initialize_rlinear(detected_p)
            # --- 初始化结束 ---

        # 检查rlinear_module是否已初始化 (处理推理阶段的第一次运行)
        if self.rlinear_module is None:
            self._initialize_rlinear(self.fixed_periods.cpu().tolist())

        # 对于所有批次（包括第一次），都使用已“锁定”的周期进行分块
        patch_list = create_patches_from_periods(x, self.fixed_periods.cpu().numpy())

        # --- 后续流程完全不变 ---
        F_channel_aware = self.gdd_mlp_module(patch_list)
        F_temporal, F_summarized = self.rlinear_module(F_channel_aware)
        predictions = self.dual_ams_module(F_channel_aware, F_summarized)

        if self.target_slice is not None:
            predictions = predictions[:, :, self.target_slice]

        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        return predictions, aux_loss

    def get_model_info(self) -> dict:
        info = {
            "architecture": "PA-HyperLinear",
            "input_shape": (self.seq_len, self.n_features),
            "pred_len": self.pred_len,
            "k_periods": self.k_periods,
            "d_model": self.d_model,
            "detected_periods": self.fixed_periods.cpu().numpy().tolist() if self.fixed_periods[0] != 0 else "N/A",
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        return info

    def extra_repr(self) -> str:
        return (f'input_shape=({self.seq_len}, {self.n_features}), '
                f'pred_len={self.pred_len}, k_periods={self.k_periods}, '
                f'd_model={self.d_model}')
