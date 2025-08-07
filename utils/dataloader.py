import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from pathlib import Path

class CustomDataLoader:
    """Generate data loader from raw data."""

    def __init__(
            self, data, batch_size, seq_len, pred_len, feature_type, target='OT'
    ):
        self.data = Path(data)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_type = feature_type
        self.target = target
        self.target_slice = slice(0, None)

        self._read_data()

    def _read_data(self):
        """Load raw data and split datasets."""
        df_raw = None
        if self.data.stem.startswith('PEMS'):
            data_raw = np.load(self.data)
            data_raw = data_raw['data'][:,:,0]
            df_raw = pd.DataFrame(data_raw)
        else:
            df_raw = pd.read_csv(self.data)

        # S: univariate-univariate, M: multivariate-multivariate, MS:
        # multivariate-univariate
        if self.data.stem.startswith('solar'):
            df = df_raw
        elif self.data.stem.startswith('PEMS'):
            df = df_raw
        else:
            df = df_raw.set_index('date')
        if self.feature_type == 'S':
            df = df[[self.target]]
        elif self.feature_type == 'MS':
            target_idx = df.columns.get_loc(self.target)
            self.target_slice = slice(target_idx, target_idx + 1)

        # split train/valid/test
        n = len(df)
        if self.data.stem.startswith('ETTm'):
            train_end = 12 * 30 * 24 * 4
            val_end = train_end + 4 * 30 * 24 * 4
            test_end = val_end + 4 * 30 * 24 * 4
        elif self.data.stem.startswith('ETTh'):
            train_end = 12 * 30 * 24
            val_end = train_end + 4 * 30 * 24
            test_end = val_end + 4 * 30 * 24
        elif self.data.stem.startswith('PEMS'):
            train_end = int(n * 0.6)
            val_end = n - int(n * 0.2)
            test_end = n
        else:
            train_end = int(n * 0.7)
            val_end = n - int(n * 0.2)
            test_end = n
        train_df = df[:train_end]
        val_df = df[train_end - self.seq_len: val_end]
        test_df = df[val_end - self.seq_len: test_end]

        READ_SHAPE = 1
        if READ_SHAPE:
            print("train : ", len(train_df) - self.seq_len + 1)
            print("valid : ", len(val_df) - self.seq_len + 1)
            print("test  : ", len(test_df) - self.seq_len + 1)

        # standardize by training set
        self.scaler = StandardScaler()
        self.scaler.fit(train_df.values)

        def scale_df(df, scaler):
            data = scaler.transform(df.values)
            return pd.DataFrame(data, index=df.index, columns=df.columns)

        self.train_df = scale_df(train_df, self.scaler)
        self.val_df = scale_df(val_df, self.scaler)
        self.test_df = scale_df(test_df, self.scaler)
        self.n_feature = self.train_df.shape[-1]

    def _make_dataset(self, data, shuffle=True, drop_last=True):
        data = np.array(data, dtype=np.float32)

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(data[:, self.target_slice], dtype=torch.float32)

        return DataLoader(
            torch.utils.data.Subset(
                CustomDataset(data_x, data_y, self.seq_len, self.pred_len),
                range(len(data_x) - self.seq_len - self.pred_len + 1)
            ),
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_train(self, shuffle=True):
        return self._make_dataset(self.train_df, shuffle=shuffle)

    def get_val(self, shuffle=True):
        return self._make_dataset(self.val_df, shuffle=shuffle)

    def get_test(self):
        return self._make_dataset(self.test_df, shuffle=False, drop_last=False)


class CustomDataset(Dataset):
    def __init__(self, data_x, data_y, seq_len, pred_len):
        self.data_x = data_x
        self.data_y = data_y

        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, idx):
        return self.data_x[idx: idx + self.seq_len], self.data_y[idx + self.seq_len: idx + self.seq_len + self.pred_len]


# def _read_data(self):
#     # 在读取数据后添加调试
#     df_raw = pd.read_csv(self.data)
#     print(f"原始数据形状: {df_raw.shape}")
#     print(f"原始数据列名: {df_raw.columns.tolist()}")
#
#     # 设置索引后
#     df = df_raw.set_index('date')
#     print(f"设置索引后形状: {df.shape}")
#     print(f"设置索引后列名: {df.columns.tolist()}")
#
#     # 特征类型处理后
#     if self.feature_type == 'MS':
#         target_idx = df.columns.get_loc(self.target)
#         self.target_slice = slice(target_idx, target_idx + 1)
#         print(f"MS模式 - 目标列索引: {target_idx}")
#         print(f"目标切片: {self.target_slice}")
#
#     # 最终特征数
#     self.n_feature = df.shape[-1]  # 这里应该是7
#     print(f"最终特征数: {self.n_feature}")

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
#
# import torch
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
#
# from pathlib import Path
#
#
# class CustomDataLoader:
#     """从原始数据生成数据加载器。"""
#
#     def __init__(
#             self, data_path, batch_size, seq_len, pred_len, feature_type, target='OT'
#     ):
#         # 修正了参数名称以匹配主程序
#         self.data_path = Path(data_path)
#         self.batch_size = batch_size
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#         self.feature_type = feature_type
#         self.target = target
#         self.target_slice = slice(0, None)
#
#         self._read_data()
#
#     def _read_data(self):
#         """加载原始数据并分割数据集。"""
#         df_raw = None
#         if self.data_path.stem.startswith('PEMS'):
#             data_raw = np.load(self.data_path)
#             data_raw = data_raw['data'][:, :, 0]
#             df_raw = pd.DataFrame(data_raw)
#         else:
#             df_raw = pd.read_csv(self.data_path)
#
#         if self.data_path.stem.startswith('solar'):
#             df = df_raw
#         elif self.data_path.stem.startswith('PEMS'):
#             df = df_raw
#         else:
#             df = df_raw.set_index('date')
#
#         if self.feature_type == 'S':
#             df = df[[self.target]]
#         elif self.feature_type == 'MS':
#             target_idx = df.columns.get_loc(self.target)
#             self.target_slice = slice(target_idx, target_idx + 1)
#
#         n = len(df)
#         if self.data_path.stem.startswith('ETTm'):
#             train_end = 12 * 30 * 24 * 4
#             val_end = train_end + 4 * 30 * 24 * 4
#             test_end = val_end + 4 * 30 * 24 * 4
#         elif self.data_path.stem.startswith('ETTh'):
#             train_end = 12 * 30 * 24
#             val_end = train_end + 4 * 30 * 24
#             test_end = val_end + 4 * 30 * 24
#         elif self.data_path.stem.startswith('PEMS'):
#             train_end = int(n * 0.6)
#             val_end = n - int(n * 0.2)
#             test_end = n
#         else:
#             train_end = int(n * 0.7)
#             val_end = n - int(n * 0.2)
#             test_end = n
#
#         train_df = df[:train_end]
#         val_df = df[train_end - self.seq_len: val_end]
#         test_df = df[val_end - self.seq_len: test_end]
#
#         self.scaler = StandardScaler()
#         self.scaler.fit(train_df.values)
#
#         def scale_df(df, scaler):
#             data = scaler.transform(df.values)
#             return pd.DataFrame(data, index=df.index, columns=df.columns)
#
#         self.train_df = scale_df(train_df, self.scaler)
#         self.val_df = scale_df(val_df, self.scaler)
#         self.test_df = scale_df(test_df, self.scaler)
#
#         # ### --- 【新】添加了这两行代码 --- ###
#         # 将完整的训练集和变量数作为类的属性暴露出来
#         self.train_set = self.train_df.values
#         self.var_num = self.train_df.shape[1]
#         # ### --- 添加结束 --- ###
#
#     def _make_dataset(self, data, shuffle=True, drop_last=True):
#         # 补全这部分逻辑
#         dataset = CustomDataset(data, self.target_slice, self.seq_len, self.pred_len)
#         data_loader = DataLoader(
#             dataset,
#             batch_size=self.batch_size,
#             shuffle=shuffle,
#             drop_last=drop_last,
#             num_workers=0
#         )
#         return data_loader
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
#
#     def get_train(self, shuffle=True):
#         return self._make_dataset(self.train_df.values, shuffle=shuffle)
#
#     def get_val(self, shuffle=False):
#         return self._make_dataset(self.val_df.values, shuffle=shuffle)
#
#     def get_test(self):
#         return self._make_dataset(self.test_df.values, shuffle=False, drop_last=False)
#
#
# class CustomDataset(Dataset):
#     def __init__(self, data, target_slice, seq_len, pred_len):
#         self.data = data
#         self.target_slice = target_slice
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#
#     def __len__(self):
#         return len(self.data) - self.seq_len - self.pred_len + 1
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end
#         r_end = r_begin + self.pred_len
#
#         seq_x = self.data[s_begin:s_end]
#         seq_y = self.data[r_begin:r_end, self.target_slice]
#         return torch.from_numpy(seq_x).float(), torch.from_numpy(seq_y).float()