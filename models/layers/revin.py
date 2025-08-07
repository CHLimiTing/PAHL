import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    可逆实例归一化 (Reversible Instance Normalization)
    专门为RLinear模块设计的归一化层
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        """
        Args:
            num_features: 特征维度数量 (通道数)
            eps: 数值稳定性常数
            affine: 是否使用可学习的仿射变换参数
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量
            mode: 'norm' 进行归一化, 'denorm' 进行反归一化

        Returns:
            处理后的张量
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")

        return x

    def _init_params(self):
        """初始化可学习参数"""
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: torch.Tensor):
        """计算统计量"""
        # 计算除了最后一个维度(通道维度)之外所有维度的均值和标准差
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """执行归一化"""
        x = x - self.mean
        x = x / self.stdev

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """执行反归一化"""
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)

        x = x * self.stdev
        x = x + self.mean

        return x


class PatchRevIN(nn.Module):
    """
    专门为patch级别设计的RevIN
    在patch维度上进行归一化，保持通道独立性
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super(PatchRevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [Batch, Num_patches, Patch_len, Channels] 或
               [Batch, Num_patches, d_model, Channels]
            mode: 'norm' 或 'denorm'
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")

        return x

    def _get_statistics(self, x: torch.Tensor):
        """在patch维度上计算统计量"""
        # 在patch_len维度上计算均值和标准差，保持其他维度
        self.mean = torch.mean(x, dim=2, keepdim=True).detach()  # [B, N, 1, C]
        self.stdev = torch.sqrt(
            torch.var(x, dim=2, keepdim=True, unbiased=False) + self.eps
        ).detach()  # [B, N, 1, C]

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """执行归一化"""
        x = (x - self.mean) / self.stdev
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """执行反归一化"""
        x = x * self.stdev + self.mean
        return x