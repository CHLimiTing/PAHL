import torch
import torch.nn as nn
from typing import List


class GatingNetwork(nn.Module):
    """
    门控网络 - 决策流
    基于丰富的通道感知特征生成融合权重
    """

    def __init__(self, total_features: int, num_experts: int):
        """
        Args:
            total_features: 展平拼接后的总特征维度
            num_experts: 专家网络数量 (等于k)
        """
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts

        # 单层门控网络
        self.gate = nn.Sequential(
            nn.Linear(total_features, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 展平拼接的特征 [Batch, Total_features]

        Returns:
            weights: 融合权重 [Batch, num_experts]
        """
        return self.gate(x)


class ExpertNetwork(nn.Module):
    """
    专家网络 - 执行流中的单个专家
    将概要特征映射到预测输出
    """

    def __init__(self, d_model: int, pred_len: int, n_vars: int, dropout: float = 0.1):
        """
        Args:
            d_model: 输入特征维度
            pred_len: 预测长度
            n_vars: 通道数量
            dropout: dropout比例
        """
        super(ExpertNetwork, self).__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.n_vars = n_vars

        # 专家网络：d_model -> pred_len，包含dropout
        self.expert = nn.Sequential(
            nn.Linear(d_model, pred_len),
            nn.Dropout(dropout)  # 在线性层后添加dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 概要特征 [Batch, d_model, Channels]

        Returns:
            output: 专家预测输出 [Batch, pred_len, Channels]
        """
        # 对每个通道独立进行预测
        # [B, d_model, C] -> [B, C, d_model] -> [B, C, pred_len] -> [B, pred_len, C]
        x_transposed = x.transpose(1, 2)  # [B, C, d_model]
        predictions = self.expert(x_transposed)  # [B, C, pred_len]
        output = predictions.transpose(1, 2)  # [B, pred_len, C]

        return output


class DualAMSModule(nn.Module):
    """
    双输入自适应混合专家系统 (Dual-input Adaptive Mixture of Specialists)
    结合决策流和执行流，实现智能的多尺度预测融合
    """

    def __init__(self, d_model: int, pred_len: int, n_vars: int,
                 num_experts: int, dropout: float = 0.1):
        """
        Args:
            d_model: 隐藏维度
            pred_len: 预测长度
            n_vars: 通道数量
            num_experts: 专家网络数量 (等于k个尺度)
            dropout: dropout比例
        """
        super(DualAMSModule, self).__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.num_experts = num_experts

        # 门控网络 (决策流) - 延迟初始化，因为需要知道展平后的特征维度
        self.gating_network = None

        # 专家网络集合 (执行流)
        self.experts = nn.ModuleList([
            ExpertNetwork(d_model, pred_len, n_vars, dropout)
            for _ in range(num_experts)
        ])

    def _compute_total_features(self, F_channel_aware: List[torch.Tensor]) -> int:
        """
        计算展平拼接后的总特征维度

        Args:
            F_channel_aware: k个尺度的通道感知特征列表

        Returns:
            total_features: 总特征维度
        """
        total_features = 0
        for features in F_channel_aware:
            # [B, N, P, C] -> B * (N * P * C)
            batch_size, num_patches, patch_len, channels = features.shape
            total_features += num_patches * patch_len * channels

        return total_features

    def _flatten_and_concat(self, F_channel_aware: List[torch.Tensor]) -> torch.Tensor:
        """
        展平并拼接所有尺度的通道感知特征

        Args:
            F_channel_aware: k个尺度的特征列表

        Returns:
            flattened_features: 展平拼接后的特征 [Batch, Total_features]
        """
        batch_size = F_channel_aware[0].shape[0]
        flattened_list = []

        for features in F_channel_aware:
            # [B, N, P, C] -> [B, N*P*C]
            flattened = features.view(batch_size, -1)
            flattened_list.append(flattened)

        # 沿特征维度拼接
        concatenated = torch.cat(flattened_list, dim=1)  # [B, Total_features]

        return concatenated

    def forward(self, F_channel_aware: List[torch.Tensor],
                F_summarized: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播：双输入自适应专家混合

        Args:
            F_channel_aware: k个尺度的通道感知特征 (决策流输入)
            F_summarized: k个尺度的概要特征 (执行流输入)

        Returns:
            final_output: 最终融合预测结果 [Batch, pred_len, Channels]
        """
        batch_size = F_channel_aware[0].shape[0]

        # 1. 决策流：展平拼接特征并生成融合权重
        flattened_features = self._flatten_and_concat(F_channel_aware)

        # 延迟初始化门控网络
        if self.gating_network is None:
            total_features = flattened_features.shape[1]
            self.gating_network = GatingNetwork(total_features, self.num_experts).to(
                flattened_features.device
            )

        # 获取融合权重 [B, k]
        fusion_weights = self.gating_network(flattened_features)

        # 2. 执行流：每个专家网络生成预测
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # 每个专家处理对应尺度的概要特征
            expert_pred = expert(F_summarized[i])  # [B, pred_len, C]
            expert_outputs.append(expert_pred)

        # 3. 加权融合所有专家的输出
        # 将专家输出堆叠: k个[B, pred_len, C] -> [B, pred_len, C, k]
        stacked_outputs = torch.stack(expert_outputs, dim=-1)

        # 扩展权重维度进行广播: [B, k] -> [B, 1, 1, k]
        expanded_weights = fusion_weights.unsqueeze(1).unsqueeze(2)

        # 加权求和: [B, pred_len, C, k] * [B, 1, 1, k] -> [B, pred_len, C]
        final_output = (stacked_outputs * expanded_weights).sum(dim=-1)

        return final_output

    def compute_load_balancing_loss(self, fusion_weights: torch.Tensor) -> torch.Tensor:
        """
        计算负载均衡损失，鼓励专家网络的均衡使用

        Args:
            fusion_weights: 融合权重 [Batch, num_experts]

        Returns:
            load_loss: 负载均衡损失
        """
        # 计算每个专家的平均使用率
        expert_usage = fusion_weights.mean(dim=0)  # [num_experts]

        # 使用变异系数作为负载均衡损失
        mean_usage = expert_usage.mean()
        variance = ((expert_usage - mean_usage) ** 2).mean()
        cv_loss = variance / (mean_usage ** 2 + 1e-8)

        return cv_loss

    def extra_repr(self) -> str:
        """模型信息字符串"""
        return (f'd_model={self.d_model}, pred_len={self.pred_len}, '
                f'n_vars={self.n_vars}, num_experts={self.num_experts}')