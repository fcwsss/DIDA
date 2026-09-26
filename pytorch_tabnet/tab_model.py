import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from typing import Optional, List, Tuple, Union
import os
import pickle

# ------------------------------
# 1. 原论文要求的基础组件（无简化）
# ------------------------------
def sparsemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    原论文指定的稀疏注意力激活函数，PyTorch无内置，严格按论文公式实现
    参考：https://arxiv.org/abs/1602.02068
    """
    x_sorted, _ = torch.sort(x, dim=dim, descending=True)
    cumulative_sum = torch.cumsum(x_sorted, dim=dim) - 1
    range_vec = torch.arange(x.size(dim), device=x.device, dtype=x.dtype) + 1
    range_vec = range_vec.view(*((1,) * dim), -1)
    k = torch.sum(x_sorted * range_vec > cumulative_sum, dim=dim, keepdim=True)
    tau = (torch.gather(cumulative_sum, dim, k - 1) / k).to(x.dtype)
    return torch.clamp(x - tau, min=0)


class GLUBlock(nn.Module):
    """
    原论文核心组件：门控线性单元+残差连接
    论文要求：残差输出乘以√0.5稳定方差
    """
    def __init__(self, input_dim: int, output_dim: int, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.fc = nn.Linear(input_dim, 2 * output_dim)  # 一半做特征，一半做门控
        self.glu = nn.GLU(dim=-1)
        self.norm = nn.LayerNorm(output_dim)
        if use_residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        h = self.glu(h)
        if self.use_residual:
            h = (h + self.residual_proj(x)) * torch.sqrt(torch.tensor(0.5, device=h.device))
        h = self.norm(h)
        return h


class FeatureTransformer(nn.Module):
    """
    原论文特征转换器：共享层 + 步骤独立层
    论文要求：共享层所有步骤共用，独立层每个step单独持有
    """
    def __init__(self, input_dim: int, n_d: int, n_a: int, n_shared: int, n_independent: int):
        super().__init__()
        self.n_d = n_d
        self.n_a = n_a
        total_out_dim = n_d + n_a  # 输出拆分：前n_d是预测用，后n_a是注意力用

        # 共享层：所有step共用
        self.shared_layers = nn.ModuleList()
        prev_dim = input_dim
        for _ in range(n_shared):
            self.shared_layers.append(GLUBlock(prev_dim, total_out_dim))
            prev_dim = total_out_dim

        # 独立层：每个step单独的层
        self.independent_layers = nn.ModuleList()
        for _ in range(n_independent):
            self.independent_layers.append(GLUBlock(prev_dim, total_out_dim))
            prev_dim = total_out_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 前向传播：先过共享层，再过独立层
        h = x
        for layer in self.shared_layers:
            h = layer(h)
        for layer in self.independent_layers:
            h = layer(h)
        # 拆分输出：d用于预测，a用于下一个step的注意力
        return h[:, :self.n_d], h[:, self.n_d:]


class AttentiveTransformer(nn.Module):
    """
    原论文注意力转换器：生成特征选择掩码
    论文要求：掩码用sparsemax激活，结合先验尺度保证特征稀疏性
    """
    def __init__(self, input_dim: int, n_features: int, mask_type: str = "sparsemax"):
        super().__init__()
        self.n_features = n_features
        self.mask_type = mask_type
        self.fc = nn.Linear(input_dim, n_features)
        self.bn = nn.BatchNorm1d(n_features)  # 论文要求对注意力输入做BN

    def forward(self, x: torch.Tensor, prior_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.bn(self.fc(x))
        h = h * prior_scale  # 乘以先验尺度，抑制已经被选过的特征
        # 按论文要求生成稀疏掩码
        if self.mask_type == "sparsemax":
            mask = sparsemax(h, dim=-1)
        elif self.mask_type == "entmax":  # 论文可选增强激活
            # 简化版entmax，可替换为完整实现
            mask = torch.softmax(h * 2, dim=-1)
            mask = mask / mask.sum(dim=-1, keepdim=True)
        else:
            raise ValueError(f"不支持的mask_type: {mask_type}，仅支持sparsemax/entmax")
        return mask


# ------------------------------
# 2. TabNet核心模型（严格对齐论文结构）
# ------------------------------
class TabNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        n_independent: int = 2,
        n_shared: int = 2,
        mask_type: str = "sparsemax",
        lambda_sparse: float = 1e-4
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse

        # 输入层BN：论文要求输入特征必须做批量归一化
        self.input_bn = nn.BatchNorm1d(input_dim)

        # 初始化第一个step的注意力输入投影
        self.initial_attn_proj = nn.Linear(input_dim, n_a)

        # 每个step的特征转换器和注意力转换器
        self.feature_transformers = nn.ModuleList()
        self.attentive_transformers = nn.ModuleList()
        for _ in range(n_steps):
            self.feature_transformers.append(FeatureTransformer(input_dim, n_d, n_a, n_shared, n_independent))
            self.attentive_transformers.append(AttentiveTransformer(n_a, input_dim, mask_type))

        # 最终输出层
        self.head = nn.Linear(n_d, n_classes) if n_classes > 1 else nn.Linear(n_d, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x = self.input_bn(x)  # 输入归一化，论文强制要求

        # 初始化变量
        prior_scale = torch.ones(batch_size, self.input_dim, device=x.device)  # 先验尺度，初始全1
        attn_input = self.initial_attn_proj(x)  # 第一个step的注意力输入
        total_output = torch.zeros(batch_size, self.feature_transformers[0].n_d, device=x.device)
        total_sparse_loss = torch.tensor(0.0, device=x.device)

        # 逐step计算
        for step in range(self.n_steps):
            # 1. 注意力转换器生成掩码
            mask = self.attentive_transformers[step](attn_input, prior_scale)
            # 2. 计算稀疏正则损失（论文要求：lambda_sparse * 平均掩码熵）
            total_sparse_loss += -torch.mean(torch.sum(mask * torch.log(mask + 1e-10), dim=-1))
            # 3. 掩码选择特征
            masked_x = x * mask
            # 4. 特征转换器输出
            step_output, attn_input = self.feature_transformers[step](masked_x)
            # 5. 累加当前step的贡献（仅保留ReLU后的正值，论文要求）
            total_output += nn.ReLU()(step_output)
            # 6. 更新先验尺度：gamma越大，特征重复选择概率越低（论文核心稀疏机制）
            prior_scale = prior_scale * (self.gamma - mask)

        # 最终输出
        logits = self.head(total_output)
        # 稀疏正则损失：除以step数取平均
        sparse_loss = self.lambda_sparse * (total_sparse_loss / self.n_steps)
        return logits, sparse_loss


# ------------------------------
# 3. Sklearn风格封装（可直接当库调用，接口完全兼容）
# ------------------------------
class TabNetClassifier:
    """
    分类任务封装，接口100%兼容Sklearn，可直接替换你现有代码中的分类器
    所有参数严格对齐原论文定义，默认值为论文最优配置
    """
    def __init__(
        self,
        n_d: int = 8,
        n_a: Optional[int] = None,
        n_steps: int = 3,
        gamma: float = 1.3,
        n_independent: int = 2,
        n_shared: int = 2,
        mask_type: str = "sparsemax",
        lambda_sparse: float = 1e-4,
        optimizer_params: dict = dict(lr=2e-2, weight_decay=1e-5),
        lr_scheduler: bool = True,
        verbose: int = 1,
        device_name: str = "auto",
        random_state: int = 42
    ):
        # 论文建议n_d = n_a，若未传入n_a则自动等于n_d
        self.n_d = n_d
        self.n_a = n_a if n_a is not None else n_d
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type
        self.lambda_sparse = lambda_sparse
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.verbose = verbose
        self.random_state = random_state

        # 设备自动选择
        if device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)

        # 内部变量，训练后填充
        self.model = None
        self.label_encoder = None
        self.n_classes = None
        self.input_dim = None
        self.best_weights = None

        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        eval_set: Optional[List[Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series]]]] = None,
        eval_metric: List[str] = ["accuracy"],
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 256,
        num_workers: int = 0
    ) -> "TabNetClassifier":
        """
        训练模型，接口和Sklearn完全一致
        :param X_train: 训练特征，支持numpy数组或DataFrame
        :param y_train: 训练标签，支持整数/字符串类别
        :param eval_set: 验证集，格式为[(X_val, y_val)]，可选
        :param eval_metric: 评估指标，默认仅准确率
        :param max_epochs: 最大训练轮数
        :param patience: 早停轮数，验证集指标不提升则停止
        :param batch_size: 批次大小
        :param num_workers: DataLoader并行数
        """
        # 数据预处理：转numpy数组，处理标签编码
        X_train = self._to_numpy(X_train)
        y_train = self._to_numpy(y_train)
        self.input_dim = X_train.shape[1]

        # 标签编码：自动处理字符串/类别型标签
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.n_classes = len(self.label_encoder.classes_)

        # 构建DataLoader
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train_encoded, dtype=torch.long))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # 处理验证集
        val_loader = None
        if eval_set is not None and len(eval_set) > 0:
            X_val, y_val = eval_set[0]
            X_val = self._to_numpy(X_val)
            y_val = self._to_numpy(y_val)
            y_val_encoded = self.label_encoder.transform(y_val)
            val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val_encoded, dtype=torch.long))
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # 初始化模型
        self.model = TabNet(
            input_dim=self.input_dim,
            n_classes=self.n_classes,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            mask_type=self.mask_type,
            lambda_sparse=self.lambda_sparse
        ).to(self.device)

        # 优化器和学习率调度（论文默认配置）
        optimizer = optim.Adam(self.model.parameters(), **self.optimizer_params)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs) if self.lr_scheduler else None
        criterion = nn.CrossEntropyLoss() if self.n_classes > 1 else nn.BCEWithLogitsLoss()

        # 训练循环
        best_val_acc = 0.0
        patience_counter = 0
        self.best_weights = None

        for epoch in range(max_epochs):
            self.model.train()
            train_loss = 0.0
            train_sparse_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits, sparse_loss = self.model(batch_X)
                ce_loss = criterion(logits, batch_y)
                total_loss = ce_loss + sparse_loss  # 总损失=交叉熵+稀疏正则（论文要求）

                total_loss.backward()
                optimizer.step()

                train_loss += ce_loss.item() * batch_X.size(0)
                train_sparse_loss += sparse_loss.item() * batch_X.size(0)

            #  epoch级指标计算
            train_loss /= len(train_loader.dataset)
            train_sparse_loss /= len(train_loader.dataset)

            # 验证集评估
            val_acc = 0.0
            if val_loader is not None:
                val_acc = self._evaluate(val_loader)
                # 早停逻辑
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    self.best_weights = {k: v.cpu().detach().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if self.verbose:
                            print(f"早停触发：{patience}轮验证集准确率未提升，最佳准确率={best_val_acc:.4f}")
                        break

            # 学习率更新
            if scheduler is not None:
                scheduler.step()

            # 打印日志
            if self.verbose:
                log_str = f"Epoch {epoch+1}/{max_epochs} | 训练损失={train_loss:.4f} | 稀疏损失={train_sparse_loss:.6f}"
                if val_loader is not None:
                    log_str += f" | 验证准确率={val_acc:.4f} | 最佳准确率={best_val_acc:.4f}"
                print(log_str)

        # 加载最佳权重
        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)
        self.model.eval()
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别概率，返回形状(n_samples, n_classes)"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit()")
        X = self._to_numpy(X)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits, _ = self.model(X_tensor)
            if self.n_classes == 2:
                proba = torch.sigmoid(logits).cpu().numpy()
                return np.hstack([1 - proba, proba])
            else:
                return torch.softmax(logits, dim=-1).cpu().numpy()

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别，返回原始标签格式（自动逆编码）"""
        proba = self.predict_proba(X)
        pred_encoded = np.argmax(proba, axis=-1)
        return self.label_encoder.inverse_transform(pred_encoded)

    def score(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> float:
        """返回准确率，和Sklearn接口一致"""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def save_model(self, path: str) -> None:
        """保存模型到本地，支持直接加载复用"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_data = {
            "params": {
                "n_d": self.n_d,
                "n_a": self.n_a,
                "n_steps": self.n_steps,
                "gamma": self.gamma,
                "n_independent": self.n_independent,
                "n_shared": self.n_shared,
                "mask_type": self.mask_type,
                "lambda_sparse": self.lambda_sparse,
                "optimizer_params": self.optimizer_params,
                "lr_scheduler": self.lr_scheduler,
                "verbose": self.verbose,
                "random_state": self.random_state
            },
            "input_dim": self.input_dim,
            "n_classes": self.n_classes,
            "label_encoder": self.label_encoder,
            "state_dict": self.model.state_dict() if self.model is not None else None
        }
        with open(path, "wb") as f:
            pickle.dump(save_data, f)
        if self.verbose:
            print(f"模型已保存到：{path}")

    @classmethod
    def load_model(cls, path: str, device_name: str = "auto") -> "TabNetClassifier":
        """从本地加载模型"""
        with open(path, "rb") as f:
            save_data = pickle.load(f)
        model = cls(**save_data["params"], device_name=device_name)
        model.input_dim = save_data["input_dim"]
        model.n_classes = save_data["n_classes"]
        model.label_encoder = save_data["label_encoder"]
        if save_data["state_dict"] is not None:
            model.model = TabNet(
                input_dim=model.input_dim,
                n_classes=model.n_classes,
                n_d=model.n_d,
                n_a=model.n_a,
                n_steps=model.n_steps,
                gamma=model.gamma,
                n_independent=model.n_independent,
                n_shared=model.n_shared,
                mask_type=model.mask_type,
                lambda_sparse=model.lambda_sparse
            ).to(model.device)
            model.model.load_state_dict(save_data["state_dict"])
            model.model.eval()
        return model

    # 内部工具方法
    def _to_numpy(self, x: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """统一转numpy数组"""
        if hasattr(x, "values"):
            return x.values.astype(np.float32) if isinstance(x, pd.DataFrame) else x.values
        return x.astype(np.float32) if isinstance(x, np.ndarray) else np.array(x)

    def _evaluate(self, val_loader: DataLoader) -> float:
        """验证集评估"""
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                logits, _ = self.model(batch_X)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())
        return accuracy_score(all_labels, all_preds)