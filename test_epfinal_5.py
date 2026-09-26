import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score # 新增f1_score导入
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
from shap import TreeExplainer
import warnings
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# 基础数据处理库
from itertools import product

# 修正：拆分导入（区分集成模型和单棵决策树）
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.tree import DecisionTreeClassifier  # 决策树单独从tree模块导入
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import torch
# 深拷贝（避免模型参数污染）

seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
plt.switch_backend('Agg')
warnings.filterwarnings('ignore')

# -------------------------- 1. 数据加载与预处理 --------------------------

# 加载数据（相对路径：确保csv文件和代码在同一目录）
try:
    data = pd.read_csv("heart.csv", encoding='utf-8')
except UnicodeDecodeError:
    # 兼容中文编码问题（Windows常见）
    data = pd.read_csv("heart.csv", encoding='gbk')

# 跳过第一列（通常是ID或名称）
if data.shape[1] > 1:
    data = data.iloc[:, 1:]  # 从第二列开始保留
    print(f"已跳过第一列，数据形状变为：{data.shape}（样本数：{data.shape[0]}, 特征数：{data.shape[1] - 1}）")
else:
    print("数据只有一列，无法跳过第一列")

# 定义目标列
target_col = "HeartDisease"
if target_col not in data.columns:
    raise ValueError(
        f"目标列'{target_col}'不存在！请修改target_col为数据中的真实标签列名（当前数据列名：{data.columns.tolist()}）")

# 分离特征X和目标y
X = data.drop(columns=[target_col])
y = data[target_col]

# -------------------------- 关键修改1：保存目标变量编码器（y_encoder）- 解决测试集y编码问题 + 类型兼容 --------------------------
y_encoder = LabelEncoder()  # 全局变量：保存目标变量编码映射（供测试集使用）
n_classes = len(y.unique())
print(f"\n目标列识别到 {n_classes} 个类别（类别值：{sorted(y.unique())}）")

# 目标变量编码（统一使用LabelEncoder确保类别从0开始，转为Series保留索引，避免reset_index报错）
y_encoded = y_encoder.fit_transform(y)  # 先得到numpy数组
y = pd.Series(y_encoded, index=y.index)  # 转为pandas Series，保留原始索引
print(f"目标列已通过y_encoder编码（原始类别：{y_encoder.classes_} → 编码后：{sorted(np.unique(y))}）")

# 识别数值/类别特征
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"\n数值型特征：{numeric_cols}")
print(f"类别型特征：{categorical_cols}")

# -------------------------- 关键修改2：类别特征处理（去重+统一命名）- 避免重复编码 --------------------------
label_encoders = {}  # 全局变量：存储每个类别特征的编码映射（供反向转换+测试集使用）
cat_imputer = None  # 全局变量：类别特征填充器（供测试集使用）
if categorical_cols:
    # 类别特征缺失值填充（用独立变量名cat_imputer，避免与数值特征冲突）
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
    print("类别型特征缺失值填充完成（众数填充）")

    # 类别特征编码 + 保存编码器（仅执行一次，删除后续重复代码）
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))  # 转为字符串避免类型错误
        label_encoders[col] = le  # 保存每个特征的编码器
    print(f"类别型特征编码完成，已保存 {len(label_encoders)} 个特征的编码映射")
else:
    print("无类别型特征，跳过缺失值填充和编码")

# -------------------------- 关键修改3：数值特征填充器声明为全局（numeric_imputer）- 解决测试集填充报错 --------------------------
numeric_imputer = None  # 全局变量：供测试集数值特征填充使用
if numeric_cols:
    numeric_imputer = SimpleImputer(strategy='median')  # 全局变量，测试集可访问
    X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
    print("数值型特征缺失值填充完成（中位数填充）")
else:
    print("无数值型特征，跳过缺失值填充")

# -------------------------- 数据集划分：3:1:1（训练集:验证集:测试集） --------------------------
# 第一步：将数据分为训练集（60%）和临时集（40%）
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=seed, stratify=y  # 分层抽样，保证类别分布一致
)
# 第二步：将临时集（40%）分为验证集（20%总数据）和测试集（20%总数据）
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp  # 分层抽样
)
train_df = pd.DataFrame(X_train)
train_df[target_col] = y_train
train_df.to_csv("train.csv", index=False, encoding="utf-8-sig")
print("✅ X_train + y_train 已保存为 train.csv")

print(f"数据集按3:1:1划分完成：")
print(f"训练集样本数：{X_train.shape[0]}（60%）")
print(f"验证集样本数：{X_val.shape[0]}（20%）")
print(f"测试集样本数：{X_test.shape[0]}（20%）")

# 重置索引（保持原有逻辑，此时y是Series，支持reset_index方法）
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

print(f"\n数据预处理完成！")
print(f"训练集特征形状：{X_train.shape}")
print(f"训练集标签形状：{y_train.shape}")
print(f"全局可用组件：numeric_imputer（数值填充）、y_encoder（目标编码）、label_encoders（类别编码）、cat_imputer（类别填充）")

# -------------------------- 2. 训练基准模型 --------------------------
param_grid = {
    # 核心必调参数（对效果影响最大，保留3个取值）
    'learning_rate': [0.05, 0.1, 0.15],  # 学习率：核心节奏参数
    'max_depth': [3, 4, 5],  # 树深度：限制复杂度核心
    'n_estimators': [100, 150, 200],  # 树数量：减少最大值，避免无效迭代

    # 次要参数（影响中等，保留2个取值）
    'gamma': [0, 0.2],  # 分裂阈值：0=无限制，0.2=适度限制
    'min_child_weight': [1, 3],  # 叶子节点权重：1=默认，3=适度限制
    'reg_lambda': [1, 3],  # L2正则：核心正则化，必调
    'subsample': [0.8, 1.0],  # 样本采样：0.8=随机，1.0=全样本
    'colsample_bytree': [0.8, 1.0]  # 特征采样：0.8=随机，1.0=全特征
}

# ------------------------------------------------------------------------------
# 3. 分类任务适配（保持原逻辑）
if n_classes == 2:
    objective = 'binary:logistic'
    eval_metric = 'auc'
    scoring = 'roc_auc'
    print(f"\n当前为二元分类任务，使用AUC-ROC作为评估指标")
else:
    objective = 'multi:softprob'
    eval_metric = 'mlogloss'
    scoring = 'accuracy'
    print(f"\n当前为多元分类任务（{n_classes}类），使用准确率作为评估指标")

# ------------------------------------------------------------------------------
# 4. 初始化模型（优化并行+早停）
xgb_model = xgb.XGBClassifier(
    objective=objective,
    eval_metric=eval_metric,
    num_class=n_classes if n_classes > 2 else None,
    random_state=seed,
    use_label_encoder=False,
    n_jobs=1,  # 关键：禁用XGB内部并行，避免与GridSearchCV冲突
    verbosity=0,
    early_stopping_rounds=20,  # 早停轮数从30→20，减少单轮训练时间
    scale_pos_weight=1 if n_classes == 2 else None  # 二分类不平衡时可调整
)

# ------------------------------------------------------------------------------
# 5. 超参数搜索（二选一：速度优先用随机搜索，效果优先用网格搜索）
search_type = "grid"  # "grid"（网格搜索）或 "random"（随机搜索）
n_iter = 100  # 随机搜索的采样次数（100次足够找到优质参数）

print(f"\n开始超参数调优（5折交叉验证 + {search_type}搜索）...")
if search_type == "grid":
    search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring=scoring,
        n_jobs=-1,  # 关键：使用所有CPU核心并行
        verbose=1,
        refit=True
    )
else:  # 随机搜索（推荐！速度提升10倍以上）
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=5,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        refit=True,
        random_state=seed  # 固定随机种子，结果可复现
    )

# 训练（早停依赖验证集，仅传递必要参数）
search.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False  # 关闭单轮训练日志，避免刷屏
)

best_model = search.best_estimator_

# ------------------------------------------------------------------------------
# 6. 性能评估（保持原逻辑，同时看训练集+验证集）
print(f"\n=== 最优超参数 ===")
for param, value in search.best_params_.items():
    print(f"{param}: {value}")

print(f"\n=== 交叉验证性能 ===")
cv_mean_score = search.best_score_
cv_std_score = search.cv_results_['std_test_score'][search.best_index_]
print(f"交叉验证平均得分: {cv_mean_score:.4f} (±{cv_std_score:.4f})")

print(f"\n=== 训练集性能 ===")
if n_classes == 2:
    y_train_proba = best_model.predict_proba(X_train)[:, 1]
    train_auc_roc = roc_auc_score(y_train, y_train_proba)
    train_auc_pr = average_precision_score(y_train, y_train_proba)
    print(f"AUC-ROC: {train_auc_roc:.4f} | AUC-PR: {train_auc_pr:.4f}")
else:
    y_train_pred = best_model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"准确率: {train_acc:.4f}")
    print("分类报告:\n", classification_report(
        y_train, y_train_pred, target_names=[f"类{i}" for i in range(n_classes)]
    ))

print(f"\n=== 验证集性能（泛化能力） ===")
if n_classes == 2:
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    val_auc_roc = roc_auc_score(y_val, y_val_proba)
    val_auc_pr = average_precision_score(y_val, y_val_proba)
    print(f"AUC-ROC: {val_auc_roc:.4f} | AUC-PR: {val_auc_pr:.4f}")
else:
    y_val_pred = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"准确率: {val_acc:.4f}")
    print("分类报告:\n", classification_report(
        y_val, y_val_pred, target_names=[f"类{i}" for i in range(n_classes)]
    ))

# -------------------------- 测试集性能评估 --------------------------
# 输入测试集文件路径
test_file = input("\n请输入测试集CSV文件路径（留空则使用内置测试集）: ").strip()
test_performance = None

if test_file:
    try:
        # 读取测试集
        try:
            test_data = pd.read_csv(test_file, encoding='utf-8')
        except UnicodeDecodeError:
            test_data = pd.read_csv(test_file, encoding='gbk')

        # 跳过测试集第一列
        if test_data.shape[1] > 1:
            test_data = test_data.iloc[:, 1:]

        # 检查目标列是否存在
        if target_col not in test_data.columns:
            raise ValueError(f"测试集中不存在目标列'{target_col}'")

        # 分离特征和目标
        X_test_input = test_data.drop(columns=[target_col])
        y_test_input = test_data[target_col]

        # 处理目标变量编码（复用训练集编码器）
        y_test_input = y_encoder.transform(y_test_input)

        # 处理测试集特征（复用训练集填充器和编码器）
        # 数值型特征填充（复用训练集的median）
        if numeric_cols and numeric_imputer is not None:
            X_test_input[numeric_cols] = numeric_imputer.transform(X_test_input[numeric_cols])

        # 类别型特征填充和编码（复用训练集的众数和编码映射）
        if categorical_cols and cat_imputer is not None:
            X_test_input[categorical_cols] = cat_imputer.transform(X_test_input[categorical_cols])
            for col in categorical_cols:
                if col in label_encoders:
                    # 处理测试集中可能出现的新类别
                    le = label_encoders[col]
                    X_test_input[col] = X_test_input[col].astype(str)
                    # 未知类别映射为-1或最常见类别（这里用-1）
                    mask = ~X_test_input[col].isin(le.classes_)
                    if mask.any():
                        print(f"警告：测试集特征'{col}'存在训练集未见过的类别，已映射为-1")
                        X_test_input.loc[mask, col] = le.classes_[0]  # 用训练集最常见类别填充
                    X_test_input[col] = le.transform(X_test_input[col])

        # 使用输入的测试集进行评估
        X_test = X_test_input
        y_test = y_test_input
        print("\n使用输入的测试集评估模型性能...")
    except Exception as e:
        print(f"测试集处理出错，将使用内置测试集：{str(e)}")

# 评估模型在测试集上的性能
print("\n=== 测试集性能（最终评估） ===")
if n_classes == 2:
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    test_auc_roc = roc_auc_score(y_test, y_test_proba)
    test_auc_pr = average_precision_score(y_test, y_test_proba)
    test_performance = test_auc_roc
    print(f"AUC-ROC: {test_auc_roc:.4f} | AUC-PR: {test_auc_pr:.4f}")
    print(f"与训练集对比: {test_auc_roc:.4f} vs {train_auc_roc:.4f} (差异: {test_auc_roc - train_auc_roc:.4f})")
else:
    y_test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_performance = test_acc
    print(f"准确率: {test_acc:.4f}")
    print(f"与训练集对比: {test_acc:.4f} vs {train_acc:.4f} (差异: {test_acc - train_acc:.4f})")
    print("分类报告:\n", classification_report(
        y_test, y_test_pred, target_names=[f"类{i}" for i in range(n_classes)]
    ))

# -------------------------- 3. SHAP值计算 --------------------------
print("\n开始计算SHAP值...")
explainer = TreeExplainer(best_model)
shap_values = explainer.shap_values(X_train)

# 明确核心维度变量
n_samples = X_train.shape[0]  # 样本数
n_features = X_train.shape[1]  # 特征数
print(f"基础维度信息：样本数={n_samples}, 特征数={n_features}, 类别数={n_classes}")

# 处理多分类场景
if n_classes > 2:
    # 1. 处理SHAP值可能的不同格式（列表或数组）
    if isinstance(shap_values, list):
        # 列表格式：检查每个元素形状是否为(样本数, 特征数)
        for i, sv in enumerate(shap_values):
            assert sv.shape == (n_samples, n_features), \
                f"列表中第{i}个类别SHAP值形状应为({n_samples}, {n_features})，实际为{sv.shape}"
        shap_array = np.array(shap_values)  # 转换为(类别数, 样本数, 特征数)
    else:
        # 数组格式：处理可能的维度顺序错误（如样本数在前的情况）
        shap_array = np.array(shap_values)
        # 常见错误格式1：(样本数, 特征数, 类别数) → 转换为(类别数, 样本数, 特征数)
        if shap_array.ndim == 3 and shap_array.shape == (n_samples, n_features, n_classes):
            shap_array = shap_array.transpose(2, 0, 1)
        # 常见错误格式2：(样本数, 类别数, 特征数) → 转换为(类别数, 样本数, 特征数)
        elif shap_array.ndim == 3 and shap_array.shape == (n_samples, n_classes, n_features):
            shap_array = shap_array.transpose(1, 0, 2)

    # 2. 验证转换后的维度正确性
    assert shap_array.ndim == 3, \
        f"多分类SHAP值应转换为3维数组，实际为{shap_array.ndim}维"
    assert shap_array.shape[0] == n_classes, \
        f"转换后第一维应为类别数{n_classes}，实际为{shap_array.shape[0]}"
    assert shap_array.shape[1] == n_samples, \
        f"转换后第二维应为样本数{n_samples}，实际为{shap_array.shape[1]}"
    assert shap_array.shape[2] == n_features, \
        f"转换后第三维应为特征数{n_features}，实际为{shap_array.shape[2]}"

    # 3. 沿类别轴取绝对值平均，得到(样本数, 特征数)
    shap_values = np.mean(np.abs(shap_array), axis=0)

# 处理二分类场景
else:
    # 确保是二维数组 (样本数, 特征数)
    shap_values = np.array(shap_values)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(n_samples, n_features)
    elif shap_values.ndim == 2 and shap_values.shape[1] == 1:
        # 处理部分版本返回(样本数,1)的情况
        shap_values = shap_values.reshape(n_samples, n_features)
    shap_values = np.abs(shap_values)
    assert shap_values.shape == (n_samples, n_features), \
        f"二分类SHAP值维度应为({n_samples}, {n_features})，实际为{shap_values.shape}"

# 最终维度验证
assert shap_values.shape == (n_samples, n_features), \
    f"处理后SHAP值维度应为({n_samples}, {n_features})，实际为{shap_values.shape}"

print(f"SHAP值矩阵形状：{shap_values.shape}（样本数：{n_samples}, 特征数：{n_features}）")

# -------------------------- 4. 全局特征重要性计算与归一化（分开处理数值和类别特征） --------------------------
numeric_indices = [X.columns.get_loc(col) for col in numeric_cols]
categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]

# 计算所有特征的原始重要性（不分开归一化）
all_raw_importance = []
all_feature_names = []
all_feature_types = []

# 收集数值特征重要性
if numeric_indices:
    numeric_importance = np.mean(np.abs(shap_values[:, numeric_indices]), axis=0)
    all_raw_importance.extend(numeric_importance)
    all_feature_names.extend(numeric_cols)
    all_feature_types.extend(['numeric'] * len(numeric_cols))

# 收集类别特征重要性
if categorical_indices:
    categorical_importance = np.mean(np.abs(shap_values[:, categorical_indices]), axis=0)
    all_raw_importance.extend(categorical_importance)
    all_feature_names.extend(categorical_cols)
    all_feature_types.extend(['categorical'] * len(categorical_cols))


# 统一归一化函数（使用所有特征的最大重要性作为基准）
def normalize_importance(importance_list, min_ratio=0.0005):
    importance_array = np.array(importance_list)
    max_imp = importance_array.max()
    if max_imp < 1e-10:
        return np.ones_like(importance_array)
    raw_ratio = importance_array / max_imp
    return raw_ratio.clip(min=min_ratio)


# 执行统一归一化
normalized_importance = normalize_importance(all_raw_importance)

# 合并结果
feature_importance_data = []
for i in range(len(all_feature_names)):
    feature_importance_data.append({
        'feature_name': all_feature_names[i],
        'feature_type': all_feature_types[i],
        'raw_importance': all_raw_importance[i],
        'normalized_importance': normalized_importance[i]
    })

feature_importance_df = pd.DataFrame(feature_importance_data)
feature_importance_df = feature_importance_df.sort_values('normalized_importance', ascending=False).reset_index(
    drop=True)

# -------------------------- 5. 结果输出 --------------------------
print("\n" + "=" * 60)
print("特征全局重要性（归一化后，[0,1]区间）：")
print("=" * 60)
print(feature_importance_df.to_string(index=False))

# 保存结果
output_path = "feature_importance_shap.csv"
feature_importance_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n特征重要性结果已保存至：{output_path}")

# -------------------------- 6. SHAP可视化 --------------------------
print("\n生成SHAP可视化图表（关闭图表后继续执行）...")
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 摘要点图
    shap.summary_plot(shap_values, X_train, feature_names=X.columns, plot_type="dot", show=False)
    plt.savefig("shap_summary_dot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 特征重要性条形图
    shap.summary_plot(shap_values, X_train, feature_names=X.columns, plot_type="bar", show=False)
    plt.savefig("shap_summary_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("SHAP可视化图表已保存为：shap_summary_dot.png、shap_summary_bar.png")
except Exception as e:
    print(f"可视化生成失败（不影响核心结果）：{str(e)}")

# -------------------------- 最终结论输出 --------------------------
print("\n" + "=" * 60)
print("SHAP特征贡献度分析全流程完成！")
print("=" * 60)
if not feature_importance_df.empty:
    print(f"关键结论：")
    print(
        f"1. 最重要特征：{feature_importance_df.iloc[0]['feature_name']}（归一化重要性：{feature_importance_df.iloc[0]['normalized_importance']:.4f}）")
    print(
        f"2. 最不重要特征：{feature_importance_df.iloc[-1]['feature_name']}（归一化重要性：{feature_importance_df.iloc[-1]['normalized_importance']:.4f}）")

# -------------------------- 7. Data Shapley样本贡献度计算 --------------------------
print("\n" + "=" * 60)
print("开始计算 Data Shapley 样本贡献度（高精度KNN-Shapley）...")
print("=" * 60)
# 核心参数（完全复用原有参数，无需新增）
T = 100  # 近邻数，越大越精确，建议按样本量调整
min_subset_size = max(20, int(X_train.shape[0] * 0.3))
max_subset_size = int(X_train.shape[0] * 0.5)
n_samples = X_train.shape[0]
all_classes = set(y_train.unique())
max_retry = 10
def compute_tmc_shapley():
    ds_raw = np.zeros(n_samples)
    K = min(T, n_samples - 1)
    if K <= 0:
        return ds_raw
    # --------------------------
    # 1. 生成加权叶子嵌入
    # --------------------------
    X_leaf = best_model.apply(X_train).reshape(n_samples, -1)
    n_trees = X_leaf.shape[1]
    tree_weights = 1 / (np.arange(n_trees) + 1)
    tree_weights = tree_weights / tree_weights.sum()
    # --------------------------
    # 2. 预计算Shapley对齐权重
    # --------------------------
    shap_weights = np.array([1/(k * (n_samples - k)) for k in range(1, K+1)])
    shap_weights = shap_weights / shap_weights.sum()
    # --------------------------
    # 3. 标签预处理
    # --------------------------
    y = y_train.values.reshape(-1)
    class_to_idx = {c:i for i, c in enumerate(sorted(all_classes))}
    y_enc = np.array([class_to_idx[label] for label in y])
    # --------------------------
    # 4. 逐行计算贡献（内存友好）
    # --------------------------
    if n_classes == 2:
        y_bin = y_enc
        for j in tqdm(range(n_samples), desc="计算样本贡献度"):
            # 逐行计算加权汉明距离
            dist_j = np.sum((X_leaf[j] != X_leaf) * tree_weights, axis=1)
            dist_j[j] = np.inf
            # 快速取TopK近邻
            knn_indices = np.argpartition(dist_j, K)[:K]
            knn_sorted = knn_indices[np.argsort(dist_j[knn_indices])]
            y_j = y_bin[j]
            # 加权计算贡献
            for rank, i in enumerate(knn_sorted):
                w = shap_weights[rank]
                ds_raw[i] += w / n_samples if y_bin[i] == y_j else -w / n_samples
    else:
        for j in tqdm(range(n_samples), desc="计算样本贡献度"):
            dist_j = np.sum((X_leaf[j] != X_leaf) * tree_weights, axis=1)
            dist_j[j] = np.inf
            knn_indices = np.argpartition(dist_j, K)[:K]
            knn_sorted = knn_indices[np.argsort(dist_j[knn_indices])]
            y_j = y_enc[j]
            for rank, i in enumerate(knn_sorted):
                w = shap_weights[rank]
                ds_raw[i] += w / n_samples if y_enc[i] == y_j else -w / n_samples
    # 对齐原TMC输出范围
    ds_raw = np.clip(ds_raw, -1, 1)
    return ds_raw
# 后续归一化、结果整理代码完全复用原有逻辑，无需修改
# -------------------------- 替换原有的归一化代码段（仅修改这部分即可） --------------------------
ds_raw = compute_tmc_shapley()

# ====================== 归一化配置参数（可根据你的数据集调整） ======================
# 核心参数：用多少分位的贡献值代替最大值作为缩放上限，避免离群点压缩分布
# 95~99.9之间：数值越小，分布越分散，越不会集中在尾部；越大越接近原始最大值的效果
PERCENTILE_UPPER = 99
# 是否把负贡献的样本值截断为0（你后续要筛负贡献样本，建议开，避免负分干扰）
CLIP_NEGATIVE_TO_ZERO = True
# 可选拉伸系数：如果还是觉得分布集中，可设置>1的数（比如1.5/2），拉伸低贡献区域的区分度，<1则压缩
STRETCH_POWER = 1.0
# ==================================================================================

# 1. 先处理负样本截断（可选）
if CLIP_NEGATIVE_TO_ZERO:
    ds_raw_clipped = np.clip(ds_raw, 0, None)
else:
    ds_raw_clipped = ds_raw

# 2. 计算稳健的上下边界（不用最大/最小值，用百分位避免离群点）
min_ds = ds_raw_clipped.min()
# 用指定分位的值代替最大值，彻底解决离群点压缩问题
upper_bound = np.percentile(ds_raw_clipped, PERCENTILE_UPPER)

if upper_bound - min_ds < 1e-10:
    # 极端情况：所有样本贡献一致，统一设为1
    ds_normalized = np.ones(n_samples, dtype=np.float64)
else:
    # 3. 百分位缩放，超过upper_bound的截断为1
    ds_normalized = np.clip((ds_raw_clipped - min_ds) / (upper_bound - min_ds), 0, 1)
    # 4. 可选幂次拉伸：进一步调整分布集中度，不改变[0,1]区间
    if STRETCH_POWER != 1.0:
        ds_normalized = np.power(ds_normalized, STRETCH_POWER)

# 后续生成sample_contribution_df的逻辑完全不变
sample_contribution_df = pd.DataFrame({
    'sample_index': range(n_samples),
    'ds_raw': ds_raw,
    'ds_normalized': ds_normalized
})
# -------------------------- 关键修改：筛选掉Data Shapley值为负的样本 --------------------------
print("\n" + "=" * 60)
print("筛选低于阈值的样本...")
print("=" * 60)

# 确保sample_contribution_df已定义
if 'sample_contribution_df' not in locals():
    raise NameError("变量'sample_contribution_df'未定义，请检查样本贡献度计算逻辑")

# 定义筛选阈值
threshold = -1

print(f"筛选前总样本数：{len(sample_contribution_df)}")
print(f"低于阈值的样本数（ds_raw < {threshold}）：{sum(sample_contribution_df['ds_raw'] < threshold)}")
print(f"高于等于阈值的样本数（ds_raw >= {threshold}）：{sum(sample_contribution_df['ds_raw'] >= threshold)}")

# 筛选高于等于阈值的样本
sample_contribution_filtered_df = sample_contribution_df[sample_contribution_df['ds_raw'] >= threshold].copy()
sample_contribution_filtered_df = sample_contribution_filtered_df.sort_values('ds_normalized',
                                                                              ascending=False).reset_index(drop=True)

print(f"\n筛选后保留样本数：{len(sample_contribution_filtered_df)}")
print(
    f"筛选后样本索引范围：{sample_contribution_filtered_df['sample_index'].min()} - {sample_contribution_filtered_df['sample_index'].max()}")

# 保存筛选后的样本贡献度结果
ds_output_path = "sample_contribution_ds_filtered.csv"
sample_contribution_filtered_df.to_csv(ds_output_path, index=False, encoding='utf-8-sig')

print("\n" + "=" * 60)
print(f"Data Shapley 样本贡献度计算（已筛选ds_raw < {threshold}的样本）完成！")
print("=" * 60)
print(f"采样次数 T：{T}")
print(f"筛选阈值：{threshold}")
print(f"\n筛选后样本贡献度统计（归一化后）：")
print(
    f"最高贡献度：{sample_contribution_filtered_df['ds_normalized'].max():.4f}（样本索引：{sample_contribution_filtered_df.loc[sample_contribution_filtered_df['ds_normalized'].idxmax(), 'sample_index']}）")
print(
    f"最低贡献度：{sample_contribution_filtered_df['ds_normalized'].min():.4f}（样本索引：{sample_contribution_filtered_df.loc[sample_contribution_filtered_df['ds_normalized'].idxmin(), 'sample_index']}）")
print(f"平均贡献度：{sample_contribution_filtered_df['ds_normalized'].mean():.4f}")
print(f"\n筛选后样本贡献度结果已保存至：{ds_output_path}")

# -------------------------- 新增：9. 数值特征完全域构建（数据增强边界确立） --------------------------
print("\n" + "=" * 100)
print("开始构建数值特征完全域（数据增强可靠边界）...")
print("=" * 100)

# -------------------------- 配置参数 --------------------------
BUSINESS_CONSTRAINTS = {
    # 示例：根据实际特征修改，确保键与numeric_cols中的特征名一致
    "Dependents": (0, 12000000),
    "Applicant_Income": (0, 1000000),
    "Coapplicant_Income": (0, 500000),
    "Loan_Amount": (0, 8500000),
}

# 2. 其他配置参数
IQR_COEFFICIENT = 2
DS_THRESHOLD = 0.05
OUTPUT_DOMAIN_CSV = "numeric_feature_final_domain.csv"

# -------------------------- 阶段1：数值特征筛选与基础统计 --------------------------
print("\n【阶段1：数值特征基础统计与有效样本筛选】")
domain_analysis_df = pd.DataFrame()  # 存储所有分析结果

for feat in numeric_cols:
    # 提取该特征的所有原始值
    feat_values = X_train[feat].values

    # 计算四分位数（Q1=25%，Q3=75%）
    Q1 = np.percentile(feat_values, 25)
    Q3 = np.percentile(feat_values, 75)
    IQR = Q3 - Q1

    # 异常值边界（IQR准则）
    lower_bound_iqr = Q1 - IQR_COEFFICIENT * IQR
    upper_bound_iqr = Q3 + IQR_COEFFICIENT * IQR

    # 筛选有效样本（剔除超出IQR边界的异常值）
    valid_mask = (feat_values >= lower_bound_iqr) & (feat_values <= upper_bound_iqr)
    valid_samples_cnt = valid_mask.sum()
    invalid_samples_cnt = len(feat_values) - valid_samples_cnt

    # 计算基础域参数
    valid_values = feat_values[valid_mask]
    mean_val = valid_values.mean() if valid_samples_cnt > 0 else 0
    std_val = valid_values.std() if valid_samples_cnt > 0 else 0
    base_domain_width = upper_bound_iqr - lower_bound_iqr

    # 存储阶段1结果
    domain_analysis_df = pd.concat([domain_analysis_df, pd.DataFrame({
        "特征名称": [feat],
        "Q1": [round(Q1, 4)],
        "Q3": [round(Q3, 4)],
        "IQR": [round(IQR, 4)],
        "基础域（下界）": [round(lower_bound_iqr, 4)],
        "基础域（上界）": [round(upper_bound_iqr, 4)],
        "基础域宽度": [round(base_domain_width, 4)],
        "有效样本数": [valid_samples_cnt],
        "无效样本数（IQR准则）": [invalid_samples_cnt],
        "均值": [round(mean_val, 4)],
        "标准差": [round(std_val, 4)]
    })], ignore_index=True)

print("阶段1完成：基础统计与有效样本筛选")
print("基础统计结果预览：")
print(domain_analysis_df[["特征名称", "基础域（下界）", "基础域（上界）", "有效样本数", "均值", "标准差"]].to_string(
    index=False))

# -------------------------- 阶段2：业务约束融合与初步完全域确定 --------------------------
print("\n【阶段2：业务约束融合与初步完全域确定】")

business_lower = []
business_upper = []
preliminary_lower = []
preliminary_upper = []

for feat in numeric_cols:
    # 获取该特征的基础域
    base_low = domain_analysis_df.loc[domain_analysis_df["特征名称"] == feat, "基础域（下界）"].iloc[0]
    base_high = domain_analysis_df.loc[domain_analysis_df["特征名称"] == feat, "基础域（上界）"].iloc[0]

    # 获取业务约束（无则使用基础域）
    if feat in BUSINESS_CONSTRAINTS:
        bus_low, bus_high = BUSINESS_CONSTRAINTS[feat]
        bus_low = min(bus_low, bus_high)
        bus_high = max(bus_low, bus_high)
    else:
        bus_low = base_low
        bus_high = base_high
        print(f"警告：特征'{feat}'未配置业务约束，默认使用基础域作为业务约束域")

    business_lower.append(round(bus_low, 4))
    business_upper.append(round(bus_high, 4))

    # 初步完全域 = 基础域 ∩ 业务约束域（取交集）
    prelim_low = max(base_low, bus_low)
    prelim_high = min(base_high, bus_high)
    # 避免交集为空（若为空则使用基础域）
    if prelim_low > prelim_high:
        print(f"警告：特征'{feat}'基础域与业务约束域无交集，使用基础域作为初步完全域")
        prelim_low = base_low
        prelim_high = base_high

    preliminary_lower.append(round(prelim_low, 4))
    preliminary_upper.append(round(prelim_high, 4))

# 更新结果表格
domain_analysis_df["业务约束域（下界）"] = business_lower
domain_analysis_df["业务约束域（上界）"] = business_upper
domain_analysis_df["初步完全域（下界）"] = preliminary_lower
domain_analysis_df["初步完全域（上界）"] = preliminary_upper

print("阶段2完成：业务约束融合")
print("初步完全域结果预览：")
print(domain_analysis_df[
          ["特征名称", "业务约束域（下界）", "业务约束域（上界）", "初步完全域（下界）", "初步完全域（上界）"]].to_string(
    index=False))

# -------------------------- 阶段3：高价值边缘样本回收与完全域动态修正 --------------------------
print("\n【阶段3：高价值边缘样本回收与完全域动态修正】")
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)


# 步骤1：找出被初步完全域过滤的样本集G（任一数值特征超出初步完全域即属于G）
def is_sample_filtered(sample_idx):
    # 注意：此处sample_idx是X_train的位置索引（因X_train已reset_index）
    sample = X_train.iloc[sample_idx]  # 用iloc（位置索引）
    for feat in numeric_cols:
        prelim_low = domain_analysis_df.loc[domain_analysis_df["特征名称"] == feat, "初步完全域（下界）"].iloc[0]
        prelim_high = domain_analysis_df.loc[domain_analysis_df["特征名称"] == feat, "初步完全域（上界）"].iloc[0]
        if not (prelim_low <= sample[feat] <= prelim_high):
            return True  # 有任一特征超出范围，属于被过滤样本
    return False


# 生成被过滤样本集G的位置索引（因X_train已reset_index，索引为0~len(X_train)-1）
G_indices = [idx for idx in range(len(X_train)) if is_sample_filtered(idx)]
print(f"被初步完全域过滤的样本数G：{len(G_indices)}（总样本数：{len(X_train)}）")

# 初始化保留样本索引（位置索引）
retained_indices = []

if not G_indices:
    print("无被过滤样本，初步完全域即为最终完全域")
    final_lower = preliminary_lower
    final_upper = preliminary_upper
    retained_indices = list(range(len(X_train)))  # 保留所有样本（位置索引）
else:
    # 步骤2：关联样本的Data Shapley值，筛选高价值样本（dsi≥θ）
    # 使用之前建立的映射关系
    G_ds = sample_contribution_filtered_df[sample_contribution_filtered_df['sample_index'].isin(G_indices)].copy()
    # 筛选高价值样本（dsi≥θ，使用归一化后的值）
    high_value_G = G_ds[G_ds["ds_normalized"] >= DS_THRESHOLD].copy()
    high_value_indices = high_value_G['sample_index'].tolist()  # 高价值样本的位置索引

    print(f"高价值边缘样本数（dsi≥{DS_THRESHOLD}）：{len(high_value_indices)}（G中样本数：{len(G_ds)}）")

    # 确定保留样本：未被过滤的样本 + 回收的高价值边缘样本（均为位置索引）
    non_filtered_indices = [idx for idx in range(len(X_train)) if idx not in G_indices]
    retained_indices = non_filtered_indices + high_value_indices

    if not high_value_indices:
        print("无符合条件的高价值边缘样本，初步完全域即为最终完全域")
        final_lower = preliminary_lower
        final_upper = preliminary_upper
    else:
        # 步骤3：回收高价值样本，扩展完全域边界
        final_lower = []
        final_upper = []
        high_value_samples = X_train.iloc[high_value_indices]  # 用位置索引取样本

        for feat in numeric_cols:
            prelim_low = domain_analysis_df.loc[domain_analysis_df["特征名称"] == feat, "初步完全域（下界）"].iloc[0]
            prelim_high = domain_analysis_df.loc[domain_analysis_df["特征名称"] == feat, "初步完全域（上界）"].iloc[0]
            # 高价值样本的该特征取值范围
            hv_feat_min = high_value_samples[feat].min()
            hv_feat_max = high_value_samples[feat].max()
            # 扩展边界
            final_low = min(prelim_low, hv_feat_min)
            final_high = max(prelim_high, hv_feat_max)
            final_lower.append(round(final_low, 4))
            final_upper.append(round(final_high, 4))

            print(
                f"特征'{feat}'：初步域[{prelim_low:.4f}, {prelim_high:.4f}] → 扩展后[{final_low:.4f}, {final_high:.4f}]（高价值样本取值范围：[{hv_feat_min:.4f}, {hv_feat_max:.4f}]）")

# 更新最终完全域到结果表格
domain_analysis_df["最终完全域（下界）"] = final_lower
domain_analysis_df["最终完全域（上界）"] = final_upper
domain_analysis_df["最终完全域宽度"] = [round(final_upper[i] - final_lower[i], 4) for i in range(len(numeric_cols))]

# 更新样本贡献度过滤表，仅保留回收后的样本
sample_contribution_filtered_df = sample_contribution_filtered_df[
    sample_contribution_filtered_df["sample_index"].isin(retained_indices)
].reset_index(drop=True)

# 计算原样本数用于打印
if 'high_value_indices' in locals():
    original_count = len(retained_indices) + len(G_indices) - len(high_value_indices)
else:
    original_count = len(X_train)  # 无过滤时原样本数等于当前数

print(f"\n训练集已更新为高价值样本回收后的集合，新样本数：{len(X_train)}（原样本数：{original_count}）")
print(f"样本贡献度过滤表已同步更新，新记录数：{len(sample_contribution_filtered_df)}")

# -------------------------- 结果输出与保存 --------------------------
print("\n" + "=" * 100)
print("数值特征完全域构建完成！")
print("=" * 100)

# 整理最终结果表格
final_domain_cols = [
    "特征名称", "基础域（下界）", "基础域（上界）", "业务约束域（下界）", "业务约束域（上界）",
    "初步完全域（下界）", "初步完全域（上界）", "最终完全域（下界）", "最终完全域（上界）",
    "基础域宽度", "最终完全域宽度", "均值", "标准差", "有效样本数", "无效样本数（IQR准则）"
]
final_domain_df = domain_analysis_df[final_domain_cols].copy()

# 保存结果到CSV
final_domain_df.to_csv(OUTPUT_DOMAIN_CSV, index=False, encoding='utf-8-sig')
print(f"\n完全域结果已保存至：{OUTPUT_DOMAIN_CSV}")

print("\n【数值特征最终完全域汇总表】")
print(final_domain_df.to_string(index=False))

print(f"\n【关键统计】")
print(f"1. 数值特征总数：{len(numeric_cols)}")
print(f"2. 被初步过滤的样本数：{len(G_indices)}")
print(f"3. 回收的高价值边缘样本数：{len(high_value_indices) if 'high_value_indices' in locals() else 0}")
print(f"4. 贡献度阈值θ：{DS_THRESHOLD}")
print(f"5. IQR异常值判断系数：{IQR_COEFFICIENT}")

# -------------------------- 8. 样本-特征单元格综合价值评分与增强次数计算（基于筛选后的样本） --------------------------
print("\n" + "=" * 80)
print("开始计算筛选后样本-特征单元格综合价值评分与增强次数n...")
print("=" * 80)

# 可配置参数
alpha = 0.7  # 特征重要性权重
beta = 0.3  # 样本贡献度权重
low_threshold = 0.2  # 低价值阈值
high_threshold = 0.8  # 高价值阈值
enhancement_options = {
    "low": [30],  # 低价值：仅1次增强
    "medium": [10],  # 中价值：适当增强
    "high": [20]  # 高价值：多次增强
}

# 参数校验
if not np.isclose(alpha + beta, 1.0):
    raise ValueError(f"权重α+β必须等于1！当前α={alpha}, β={beta}，和为{alpha + beta}")
if low_threshold >= high_threshold:
    raise ValueError(f"低阈值必须小于高阈值！当前低阈值={low_threshold}, 高阈值={high_threshold}")

# 数据关联（使用筛选后的样本）
sample_core = sample_contribution_filtered_df[['sample_index', 'ds_raw', 'ds_normalized']].copy()
feature_core = feature_importance_df[['feature_name', 'feature_type', 'normalized_importance']].copy()
feature_core.rename(columns={'normalized_importance': 'feat_norm_importance'}, inplace=True)

# 生成所有「筛选后样本-特征」组合
from itertools import product

sample_indices_filtered = sample_core['sample_index'].tolist()
feature_info = feature_core[['feature_name', 'feature_type']].values.tolist()
sample_feature_pairs = [(s, f, t) for s, (f, t) in product(sample_indices_filtered, feature_info)]

# 转换为DataFrame并关联数据
cell_df = pd.DataFrame(sample_feature_pairs, columns=['sample_index', 'feature_name', 'feature_type'])
cell_df = cell_df.merge(sample_core, on='sample_index', how='left')
cell_df = cell_df.merge(feature_core, on=['feature_name', 'feature_type'], how='left')

# 计算综合价值评分
cell_df['alpha'] = alpha
cell_df['beta'] = beta
cell_df['comprehensive_score'] = (alpha * cell_df['feat_norm_importance'] +
                                  beta * cell_df['ds_normalized'])


# 确定增强次数n（类别特征增强次数为0）
def get_enhancement_n(row):
    if row['feature_type'] == 'categorical':
        return 0  # 类别特征增强次数统一为0
    score = row['comprehensive_score']
    if score < low_threshold:
        return random.choice(enhancement_options['low'])
    elif low_threshold <= score <= high_threshold:
        return random.choice(enhancement_options['medium'])
    else:
        return random.choice(enhancement_options['high'])


random.seed(seed)
cell_df['enhancement_times'] = cell_df.apply(get_enhancement_n, axis=1)

# 结果整理与保存
result_cols = [
    'sample_index', 'feature_name', 'feature_type',
    'ds_normalized', 'feat_norm_importance',
    'alpha', 'beta', 'comprehensive_score',
    'enhancement_times'
]
final_result_df = cell_df[result_cols].copy()

enhancement_output_path = "sample_feature_enhancement_regulation_filtered.csv"
final_result_df.to_csv(enhancement_output_path, index=False, encoding='utf-8-sig')

# 结果统计与输出
print(f"\n参数配置总结：")
print(f"  权重配置：α={alpha}（特征重要性），β={beta}（样本贡献度）")
print(f"  阈值配置：低阈值={low_threshold}，高阈值={high_threshold}")
print(f"  增强规则：")
print(f"    - 类别特征 → 增强0次")
print(f"    - 数值特征且综合评分 < {low_threshold} → 增强{enhancement_options['low']}次")
print(f"    - 数值特征且{low_threshold} <= 综合评分 <= {high_threshold} → 增强{enhancement_options['medium']}次（随机）")
print(f"    - 数值特征且综合评分 > {high_threshold} → 增强{enhancement_options['high']}次（随机）")

print(f"\n数据规模统计（筛选后）：")
print(f"  保留样本数：{len(sample_indices_filtered)}（原始{len(sample_contribution_df)}个）")
print(f"  总特征数：{len(feature_info)}（数值型：{len(numeric_cols)}, 类别型：{len(categorical_cols)}）")
print(f"  总样本-特征单元格数：{len(final_result_df)}")

print(f"\n综合价值评分分布（筛选后）：")
print(f"  最小值：{final_result_df['comprehensive_score'].min():.4f}")
print(f"  最大值：{final_result_df['comprehensive_score'].max():.4f}")
print(f"  平均值：{final_result_df['comprehensive_score'].mean():.4f}")
print(f"  中位数：{final_result_df['comprehensive_score'].median():.4f}")

print(f"\n增强次数n分布（筛选后）：")
n_distribution = final_result_df['enhancement_times'].value_counts().sort_index()
for n, count in n_distribution.items():
    percentage = (count / len(final_result_df)) * 100
    print(f"  n={n}：{count}个单元格（{percentage:.1f}%）")

print(f"\n前10个单元格调控结果示例（筛选后）：")
print(final_result_df.head(10).to_string(index=False))

print(f"\n" + "=" * 80)
print(f"调控流程完成（已筛选负贡献样本）！结果已保存至：{enhancement_output_path}")
print("=" * 80)
print(f"\n生成文件汇总：")
print(f"1. 特征重要性：{output_path}")
print(f"2. 筛选后样本贡献度：{ds_output_path}")
print(f"3. 筛选后单元格调控结果：{enhancement_output_path}")
print(f"4. SHAP可视化图表：shap_summary_dot.png、shap_summary_bar.png")
# -------------------------- 新增：10. 差异化数据增强与性能重评估 --------------------------
print("\n" + "=" * 120)
print("开始执行差异化数据增强（高价值弱扰动、低价值强扰动）...")
print("=" * 120)
# 新增：DIDA增强计时起点
import time
dida_start_time = time.time()
# -------------------------- 增强参数配置（用户可根据业务场景调整） --------------------------
ENHANCEMENT_CONFIG = {
    "p": 2.5,  # 扰动衰减系数（1.2~3.0，关键特征保护取2.0~3.0）
    "epsilon":0.2,  # （基准阈值，0.05~0.2，越小扰动越弱）
    "max_combination": 10,  # 单样本最大增强组合数（避免笛卡尔积爆炸）
    "random_seed": seed,  # 随机种子（保证可复现）
    "aug_data_output": "augmented_dataset.csv",  # 增强后数据集保存路径
    'phi1': 0.7,
    'phi2': 0.3,
    'deta': 0.03

}
phi1 = ENHANCEMENT_CONFIG['phi1']
phi2 = ENHANCEMENT_CONFIG['phi2']
p = ENHANCEMENT_CONFIG['p']
epsilon = ENHANCEMENT_CONFIG['epsilon']
deta = ENHANCEMENT_CONFIG['deta']
# 固定随机种子
np.random.seed(ENHANCEMENT_CONFIG["random_seed"])
random.seed(ENHANCEMENT_CONFIG["random_seed"])

# -------------------------- 步骤1：准备基础数据（复用前序分析结果） --------------------------
print("\n【步骤1：准备基础数据】")
# 1. 筛选后的非负贡献样本（原始数据）
filtered_sample_indices = sample_contribution_filtered_df["sample_index"].tolist()

X_filtered = X_train.iloc[filtered_sample_indices].copy()
y_filtered = y_train.iloc[filtered_sample_indices].copy()

# 2. 数值特征-完全域映射（从之前的完全域结果中提取）
domain_map = {}
for _, row in final_domain_df.iterrows():
    feat = row["特征名称"]
    domain_map[feat] = {
        "lower": row["最终完全域（下界）"],
        "upper": row["最终完全域（上界）"],
        "width": row["最终完全域宽度"]
    }

# 3. 特征重要性映射（归一化后）
feat_importance_map = dict(zip(
    feature_importance_df["feature_name"],
    feature_importance_df["normalized_importance"]
))

# 4. 样本贡献度映射（归一化后）
sample_contribution_map = dict(zip(
    sample_contribution_filtered_df["sample_index"],
    sample_contribution_filtered_df["ds_normalized"]
))

# 5. 单元格增强次数映射（从之前的调控结果中提取）
cell_enhancement_map = {}
for _, row in final_result_df.iterrows():
    sample_idx = row["sample_index"]
    feat_name = row["feature_name"]
    cell_enhancement_map[(sample_idx, feat_name)] = row["enhancement_times"]

print("基础数据准备完成：数值特征数={}，类别特征数={}".format(
    len(numeric_cols), len(categorical_cols)
))

# -------------------------- 步骤2：扰动强度函数计算 --------------------------
print("\n【步骤2：计算扰动强度】")


# fi：特征j的归一化重要性；si：样本i的归一化贡献度；ε：基准阈值；p：衰减系数
def calculate_disturbance_strength(feat_importance, sample_contribution):
    return epsilon * ((1 - (phi1 * feat_importance + phi2 * sample_contribution)) ** p) + deta


# 预计算所有（样本-数值特征）的扰动强度
disturbance_strengths = {}
for sample_idx in filtered_sample_indices:
    sample_contrib = sample_contribution_map[sample_idx]
    for feat in numeric_cols:
        feat_import = feat_importance_map[feat]
        s_ij = calculate_disturbance_strength(feat_import, sample_contrib)
        disturbance_strengths[(sample_idx, feat)] = s_ij

print("扰动强度计算完成：共{}个（样本-数值特征）单元格".format(
    len(disturbance_strengths)
))

# 输出扰动强度统计（验证逻辑合理性）
s_values = list(disturbance_strengths.values())
print(f"扰动强度统计：最大值={max(s_values):.4f}，最小值={min(s_values):.4f}，平均值={np.mean(s_values):.4f}")

# -------------------------- 步骤3：单特征增强数据生成与去重 --------------------------
print("\n【步骤3：单特征增强数据生成与去重】")
# 存储每个（样本-特征）的去重增强值（包含原始值，确保不丢失原始信息）

sample_feat_aug_values = {}

for sample_idx in tqdm(filtered_sample_indices, desc="生成单特征增强值"):
    sample_feat_aug_values[sample_idx] = {}
    sample_data = X_train.iloc[sample_idx]  # 原始样本数据

    for feat in numeric_cols:
        # 获取基础参数
        n_ij = cell_enhancement_map[(sample_idx, feat)]  # 增强次数
        s_ij = disturbance_strengths[(sample_idx, feat)]  # 扰动强度
        domain = domain_map[feat]
        original_val = sample_data[feat]
        domain_width = domain["width"]

        # 生成增强值（包含原始值，去重后保留）
        aug_values = {original_val}  # 用集合自动去重，初始包含原始值

        if n_ij > 0 and domain_width > 1e-10:  # 仅当增强次数>0且域宽度有效时生成
            for _ in range(n_ij):
                # 随机采样R∈[-1,1]
                R = np.random.uniform(-1, 1)
                # 计算增强值：x_ij^aug = x_ij + s_ij * R * 域宽度（相对于域宽度的扰动）
                aug_val = original_val + s_ij * R * domain_width
                # 强制截断到完全域内
                aug_val = np.clip(aug_val, domain["lower"], domain["upper"])
                aug_values.add(round(aug_val, 6))  # 保留6位小数，避免浮点精度导致的重复

        # 转换为列表（去重后）
        sample_feat_aug_values[sample_idx][feat] = list(aug_values)
        # 日志：若增强后无新值（仅原始值），提示
        if len(sample_feat_aug_values[sample_idx][feat]) == 1:
            print(f"警告：样本{sample_idx}的特征{feat}增强后无新值（扰动强度过小或增强次数不足）")
#------------------------------------------多特征笛卡尔积组合--------------------------------------------------------------
print("\n【步骤4：多特征笛卡尔积组合】")
augmented_samples = []  # 存储所有增强样本（不含原始样本）
# 检查target_col是否已定义
if 'target_col' not in locals():
    raise ValueError("变量 'target_col' 未定义，请确保前面代码已定义目标标签列名")
print(f"当前目标标签列名：target_col = '{target_col}'")

# -------------------------- 🔴 仅需修改这里的比例参数即可自定义分配规则 --------------------------
# 百分比配置（按样本贡献度从高到低排名）
TOP_RATIO_3 = 0.1  # 前10%的样本：生成3条增强样本
TOP_RATIO_1 = 0.5  # 前10% ~ 前50%的样本（累计前50%，中间40%）：生成1条增强样本
# 剩余后50%的样本：自动舍去，不增强（比例 1 - TOP_RATIO_1 = 0.5）
# ----------------------------------------------------------------------------------------
# 参数合法性校验（避免配置错误）
if not (0 < TOP_RATIO_3 < TOP_RATIO_1 <= 1):
    raise ValueError(f"百分比配置错误！需满足 0 < TOP_RATIO_3 < TOP_RATIO_1 <= 1，当前值：TOP_RATIO_3={TOP_RATIO_3}, TOP_RATIO_1={TOP_RATIO_1}")

MAX_CARTESIAN_PRODUCT = 100  # 每个样本最大尝试组合数，保留原参数无需修改

# -------------------------- 预计算缓存（仅执行1次，避免循环内重复查询） --------------------------
# 预存每个样本的评分，避免循环内每次过滤DataFrame（O(n)→O(1)）
sample_score_map = final_result_df.groupby('sample_index')['ds_normalized'].mean().to_dict()
# 预存所有数值特征索引，避免循环内重复取列
numeric_col_indices = [X_train.columns.get_loc(feat) for feat in numeric_cols]

# 🔴 新增：预计算全局百分比阈值（仅计算1次，所有样本共用）
all_sample_scores = [sample_score_map[idx] for idx in filtered_sample_indices]
# 计算两个分段阈值：interpolation='higher'保证严格取到对应比例的样本
thresh_top3 = np.percentile(all_sample_scores, 100*(1 - TOP_RATIO_3), interpolation='higher')  # 前10%门槛
thresh_top1 = np.percentile(all_sample_scores, 100*(1 - TOP_RATIO_1), interpolation='higher')  # 前50%门槛

# 可选：打印分段统计，方便调试确认比例
cnt_top3 = sum(score >= thresh_top3 for score in all_sample_scores)
cnt_top1 = sum((score >= thresh_top1) and (score < thresh_top3) for score in all_sample_scores)
cnt_drop = sum(score < thresh_top1 for score in all_sample_scores)
print(f"增强比例分配：前{TOP_RATIO_3*100:.0f}%({cnt_top3}个样本)→3条，中间{(TOP_RATIO_1-TOP_RATIO_3)*100:.0f}%({cnt_top1}个样本)→1条，后{(1-TOP_RATIO_1)*100:.0f}%({cnt_drop}个样本)→舍去")
# ----------------------------------------------------------------------------------------

for sample_idx in tqdm(filtered_sample_indices, desc="组合增强样本"):
    # 从缓存取当前样本的评分
    sample_score = sample_score_map[sample_idx]

    # 🔴 按排名百分比确定增强条数（完全替代原来的固定阈值判断）
    if sample_score >= thresh_top3:
        max_comb = 3
    elif sample_score >= thresh_top1:
        max_comb = 1
    else:
        max_comb = 0  # 后50%直接不增强，跳过后续所有计算

    # 🔴 提前跳过不需要增强的样本，节省大量计算时间
    if max_comb == 0:
        continue

    # 获取该样本所有数值特征的去重增强值
    feat_aug_lists = []
    for feat in numeric_cols:
        aug_vals = sample_feat_aug_values[sample_idx][feat]
        feat_aug_lists.append(aug_vals)

    # 不生成全量笛卡尔积，先计算总组合数，仅当组合数很少时才生成全量
    total_combinations = 1
    for lst in feat_aug_lists:
        total_combinations *= len(lst)
        # 提前终止：超过最大限制就不用再乘了
        if total_combinations > MAX_CARTESIAN_PRODUCT:
            break

    combinations = []
    if total_combinations <= min(max_comb, MAX_CARTESIAN_PRODUCT):
        # 组合数很少时直接生成全量
        combinations = list(product(*feat_aug_lists))
    else:
        # 组合数大时直接随机采样，完全避免生成全量笛卡尔积（核心提速点）
        sampled_combo_set = set()
        # 最多采样10倍max_comb次，避免死循环
        max_try = max_comb * 10
        try_cnt = 0
        while len(sampled_combo_set) < max_comb and try_cnt < max_try:
            # 每个特征随机选一个增强值，生成组合
            combo = tuple(random.choice(lst) for lst in feat_aug_lists)
            sampled_combo_set.add(combo)
            try_cnt += 1
        combinations = list(sampled_combo_set)

    # 提前存储原始样本数值特征元组，跳过原始组合的判断速度提升10倍
    original_sample = X_train.iloc[sample_idx]
    original_label = y_train.iloc[sample_idx]
    original_numeric_tuple = tuple(original_sample[feat] for feat in numeric_cols)

    for combo in combinations:
        # 直接对比元组，避免循环调用np.isclose
        if combo == original_numeric_tuple:
            continue

        # 原地修改Series，替代pd.concat，样本构建速度提升5倍
        aug_sample = original_sample.copy()
        # 批量赋值数值特征，避免循环查列名
        aug_sample.iloc[numeric_col_indices] = combo
        # 直接新增字段，不用concat
        aug_sample[target_col] = original_label
        aug_sample['parent_idx'] = sample_idx

        augmented_samples.append(aug_sample)

# 后续构建aug_df、保存的逻辑完全不变，和原代码100%兼容
if augmented_samples:
    aug_df = pd.DataFrame(augmented_samples)
    # ✅ 完全保留原有列顺序，兼容后续逻辑
    feature_cols = X.columns.tolist()
    aug_df = aug_df[feature_cols + [target_col, 'parent_idx']]

    aug_df.to_csv('augmented_dataset.csv', index=False, encoding='utf-8-sig')

    print(f"生成增强样本总数：{len(aug_df)}")
    print(f"增强样本已包含parent_idx字段，将用于后续类别增强的贡献度匹配")
else:
    aug_df = pd.DataFrame(columns=X.columns.tolist() + [target_col, 'parent_idx'])
    print("警告：未生成任何增强样本（可能是增强次数为0或扰动强度过小）")
# -------------------------- 关键新增：反向转换类别特征（数字→原文） --------------------------
print("\n【步骤4.5：还原类别特征为原始文本】")


def reverse_encode_categorical(df):
    """反向转换类别特征：数字→原始文本"""
    df_copy = df.copy()
    for col in categorical_cols:
        if col in df_copy.columns:
            le = label_encoders[col]  # 复用之前保存的编码器
            # 反向转换（注意：确保输入为int类型，避免浮点错误）
            df_copy[col] = le.inverse_transform(df_copy[col].astype(int))
    return df_copy


# 1. 还原增强样本的类别特征
if not aug_df.empty:
    aug_df = reverse_encode_categorical(aug_df)
    print("增强样本类别特征已还原为原始文本")

# 2. 还原原始样本的类别特征
original_df = X_train.copy()
original_df[target_col] = y_train
original_df = reverse_encode_categorical(original_df)
print("原始样本类别特征已还原为原始文本")
print(f"原始样本标签列名：'{target_col}'（已与target_col一致）")

# -------------------------- 步骤5：合并原始样本与增强样本（构建完整数据集） --------------------------
print("\n【步骤5：构建完整数据集】")
# 合并数据集
if not aug_df.empty:
    complete_df = pd.concat([aug_df], ignore_index=True)
else:
    complete_df = original_df
    print("无增强样本，完整数据集=原始筛选后样本")

# 保存完整数据集（文件名为augmented_dataset.csv，标签列名为target_col）
complete_df.to_csv(ENHANCEMENT_CONFIG["aug_data_output"], index=False, encoding='utf-8-sig')
print(f"增强后完整数据集已保存至：{ENHANCEMENT_CONFIG['aug_data_output']}")
print(f"完整数据集规模：{len(complete_df)}条（ 增强{len(aug_df) if not aug_df.empty else 0}条）")
print(f"数据集列名：{complete_df.columns.tolist()}")
print(f"标签列名验证：'{target_col}'（已确认与target_col一致）")

# 验证类别特征还原效果和标签列
if categorical_cols:  # 先判断是否有类别特征
    print(f"类别特征示例（以{categorical_cols[0]}为例）：{complete_df[categorical_cols[0]].unique()[:5]}")
else:
    print("没有识别到类别型特征，无需验证类别特征还原效果")

# 额外验证：查看标签列的前5个值
print(f"标签列（{target_col}）前5个值：{complete_df[target_col].head().tolist()}")

# -------------------------- 2.4.1 数值型特征离散化 --------------------------
print("\n" + "=" * 100)
print("2.4.1 开始执行数值型特征离散化...")
print("=" * 100)

# 离散化配置（可根据业务需求调整）
DISCRETIZATION_CONFIG = {
    # 特征名: (离散化方法, 区间数量, 区间命名前缀)
    # 离散化方法: 'equal_width' (等距划分), 'equal_frequency' (等频划分)
    # 若未指定，默认使用等频划分，区间数量为5
}

# 为未配置的数值特征设置默认离散化参数（假设 numeric_cols 已在前面代码定义）
if 'numeric_cols' not in locals():
    raise ValueError("变量 'numeric_cols' 未定义，请确保前面代码已定义数值型特征列表")

for feat in numeric_cols:
    if feat not in DISCRETIZATION_CONFIG:
        DISCRETIZATION_CONFIG[feat] = ('equal_frequency', 5, feat[0].upper())

# 存储离散化结果的字典
discretization_results = {}
# 存储离散化映射（用于后续查询）
discretization_maps = {}

# 假设 X_filtered（过滤后的原始数据集）和 final_domain_df（完全域表）已在前面代码定义
if 'X_filtered' not in locals():
    raise ValueError("变量 'X_filtered' 未定义，请确保前面代码已生成过滤后的数据集")
if 'final_domain_df' not in locals():
    raise ValueError("变量 'final_domain_df' 未定义，请确保前面代码已生成完全域表")

for feat in numeric_cols:
    method, n_bins, prefix = DISCRETIZATION_CONFIG[feat]
    values = X_filtered[feat].values

    # 获取该特征的完全域范围
    domain_row = final_domain_df[final_domain_df["特征名称"] == feat].iloc[0]
    domain_min = domain_row["最终完全域（下界）"]
    domain_max = domain_row["最终完全域（上界）"]

    # 确保所有值都在完全域范围内
    values_clamped = np.clip(values, domain_min, domain_max)

    # 等距划分
    if method == 'equal_width':
        bins = np.linspace(domain_min, domain_max, n_bins + 1)
        # 调整最后一个边界以确保包含最大值
        bins[-1] = max(bins[-1], values_clamped.max())

    # 等频划分
    elif method == 'equal_frequency':
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(values_clamped, percentiles)
        # 确保边界唯一
        bins = np.unique(bins)
        # 如果边界数量不足，调整为等距划分
        if len(bins) < n_bins + 1:
            print(f"警告：特征'{feat}'因分布特殊，从等频划分自动转为等距划分")
            bins = np.linspace(domain_min, domain_max, n_bins + 1)

    # 生成区间标签
    labels = [f"{prefix}{i + 1}: [{bins[i]:.2f}, {bins[i + 1]:.2f})" for i in range(len(bins) - 1)]
    # 最后一个区间包含上限
    labels[-1] = labels[-1].replace(")", "]")

    # 执行离散化
    discretized = np.digitize(values_clamped, bins, right=False) - 1
    # 确保所有值都被正确分类
    discretized = np.clip(discretized, 0, len(labels) - 1)

    # 计算每个区间的样本覆盖率
    coverage = np.zeros(len(labels))
    for i in range(len(labels)):
        coverage[i] = np.mean(discretized == i) * 100

    # 保存结果
    discretization_results[feat] = {
        'method': method,
        'bins': bins,
        'labels': labels,
        'discretized_values': discretized,
        'coverage': coverage
    }

    # 创建离散化映射（值到区间的映射）
    value_to_bin = {}
    # 为每个区间的关键值建立映射（更高效的方式）
    for i in range(len(bins) - 1):
        bin_start = round(bins[i], 4)
        bin_end = round(bins[i + 1], 4)
        # 存储区间边界和中间值的映射，方便后续快速查询
        value_to_bin[bin_start] = i
        value_to_bin[bin_end] = i
        value_to_bin[round((bin_start + bin_end) / 2, 4)] = i
    discretization_maps[feat] = value_to_bin

    # 打印离散化信息
    print(f"\n特征'{feat}'离散化结果：")
    print(f"  方法：{method}，区间数量：{len(labels)}")
    print(f"  区间划分：")
    for i, label in enumerate(labels):
        print(f"    {label}  覆盖率：{coverage[i]:.2f}%")

    # 验证总覆盖率
    total_coverage = np.sum(coverage)
    print(f"  总覆盖率：{total_coverage:.2f}%")
    if total_coverage < 99:
        print(f"  警告：特征'{feat}'离散化覆盖率低于99%，可能需要调整参数")

# 创建离散化后的数据集
X_discretized = X_filtered.copy()
for feat in numeric_cols:
    X_discretized[feat] = discretization_results[feat]['discretized_values']

# -------------------------- 2.4 统一特征约束挖掘（合并原2.4.2+2.4.3，输出完整概率分布） --------------------------
print("\n" + "=" * 100)
print("2.4 开始统一挖掘特征约束，构建全概率联合约束表...")
print("=" * 100)
from scipy.stats import f_oneway, chi2_contingency
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from itertools import product

# 统一约束配置（兼容原有参数，新增模式开关）
CONSTRAINT_CONFIG = {
    # 模式开关
    'full_pair_mode': False,  # True=全量单特征配对（对应原2.4.2逻辑）；False=强关联多特征筛选（对应原2.4.3逻辑）
    # 约束过滤参数
    'hard_threshold': 0.95,   # 硬约束阈值（可选，仅打标用，不截断分布）
    'soft_threshold': 0.6,    # 软约束阈值（可选，仅打标用，不截断分布）
    'min_prob_threshold': 0.0, # 最小概率阈值，默认0即输出所有取值的完整分布
    'min_sample_threshold': 5, # 组合最小样本数，低于则跳过
    # 多特征模式专属参数（full_pair_mode=False时生效）
    'max_features': 2,        # 最大联合特征数量（1=单特征，2=双特征联合，以此类推）
    'corr_threshold': 0.1,    # 关联强度阈值
    'pvalue_threshold': 0.05  # 统计显著性阈值
}

# 必要变量校验
required_vars = ['categorical_cols', 'numeric_cols', 'X_discretized']
for var in required_vars:
    if var not in locals():
        raise ValueError(f"变量 '{var}' 未定义，请确保前面代码已生成")

# -------------------------- 1. 关联性计算函数（支持类别-数值、类别-类别特征对） --------------------------
def compute_correlation(target_cat_feat, rel_feat, X):
    """计算关联特征与目标类别特征的关联分数和显著性"""
    # 目标是类别特征，关联特征分两种情况处理
    target_vals = X[target_cat_feat].values
    rel_vals = X[rel_feat].values

    # 过滤样本量不足的情况
    if len(np.unique(target_vals)) < 2 or len(np.unique(rel_vals)) < 2:
        return 0.0, 1.0

    # 情况1：关联特征是数值/离散化数值 → 用ANOVA+互信息回归
    if rel_feat in numeric_cols:
        groups = [rel_vals[target_vals == val] for val in np.unique(target_vals) if len(rel_vals[target_vals == val]) >= 5]
        if len(groups) < 2:
            return 0.0, 1.0
        f_stat, p_value = f_oneway(*groups)
        mi_score = mutual_info_regression(rel_vals.reshape(-1,1), target_vals, random_state=seed)[0]
        f_norm = min(f_stat / (f_stat + 10), 1.0)
        corr_score = 0.6 * mi_score + 0.4 * f_norm

    # 情况2：关联特征是类别特征 → 用卡方检验+互信息分类
    else:
        # 构建列联表
        contingency_table = pd.crosstab(rel_vals, target_vals)
        if (contingency_table < 5).any().any():
            return 0.0, 1.0
        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
        mi_score = mutual_info_classif(rel_vals.reshape(-1,1), target_vals, random_state=seed)[0]
        chi2_norm = min(chi2_stat / (chi2_stat + 100), 1.0)
        corr_score = 0.6 * mi_score + 0.4 * chi2_norm

    return round(corr_score, 4), round(p_value, 4)

# -------------------------- 2. 为每个目标类别特征筛选关联特征 --------------------------
joint_feature_groups = {}
all_rel_feats = X_discretized.columns.tolist() # 关联特征候选集：所有特征（数值离散化+类别）

for target_cat in categorical_cols:
    print(f"\n处理目标类别特征：{target_cat}")
    # 排除自身
    candidate_rel_feats = [f for f in all_rel_feats if f != target_cat]

    # 模式1：全量单特征配对（对应原2.4.2逻辑）
    if CONSTRAINT_CONFIG['full_pair_mode']:
        selected = [[f] for f in candidate_rel_feats] # 每个关联特征单独成组
        joint_feature_groups[target_cat] = selected
        print(f"  全量配对模式：共生成{len(selected)}组单特征关联")
        continue

    # 模式2：强关联多特征筛选（对应原2.4.3逻辑）
    # 计算所有候选特征的关联分数
    corr_results = []
    for f in candidate_rel_feats:
        score, pval = compute_correlation(target_cat, f, X_discretized)
        if score >= CONSTRAINT_CONFIG['corr_threshold'] and pval <= CONSTRAINT_CONFIG['pvalue_threshold']:
            corr_results.append((f, score))
    # 按分数排序，取top N生成所有可能的特征组合
    corr_results.sort(key=lambda x: x[1], reverse=True)
    top_feats = [f for f, _ in corr_results[:CONSTRAINT_CONFIG['max_features']]]
    if not top_feats:
        # 无强关联特征时fallback到最高关联的1个特征
        top_feats = [candidate_rel_feats[0]]
        print(f"  警告：无显著关联特征，fallback至特征{top_feats[0]}")
    # 生成1~max_features个特征的所有组合
    selected = []
    for k in range(1, CONSTRAINT_CONFIG['max_features']+1):
        from itertools import combinations
        selected.extend([list(combo) for combo in combinations(top_feats, k)])
    joint_feature_groups[target_cat] = selected
    print(f"  强关联模式：共生成{len(selected)}组特征组合（关联特征：{top_feats}）")

# -------------------------- 3. 计算所有特征组合的完整条件概率分布 --------------------------
joint_constraint_rules = []
total_rules = 0

for target_cat, feature_groups in joint_feature_groups.items():
    target_unique_vals = np.unique(X_discretized[target_cat].values) # 目标类别所有可能取值
    for feat_group in feature_groups:
        print(f"\n计算【{target_cat}】的特征组合{feat_group}的条件概率分布...")
        # 获取特征组合的所有值
        feat_values = [X_discretized[f].values for f in feat_group]
        target_values = X_discretized[target_cat].values
        # 生成所有可能的取值组合
        value_ranges = [np.unique(vals) for vals in feat_values]
        all_combos = list(product(*value_ranges))
        valid_combo_cnt = 0

        for combo in all_combos:
            # 生成组合掩码
            mask = np.ones(len(target_values), dtype=bool)
            for i, f in enumerate(feat_group):
                mask &= (feat_values[i] == combo[i])
            sample_cnt = np.sum(mask)
            if sample_cnt < CONSTRAINT_CONFIG['min_sample_threshold']:
                continue
            valid_combo_cnt += 1
            # 计算目标类别所有取值的条件概率（完整分布，不截断）
            total = sample_cnt
            prob_list = []
            for target_val in target_unique_vals:
                count = np.sum(target_values[mask] == target_val)
                p = count / total
                if p >= CONSTRAINT_CONFIG['min_prob_threshold']:
                    # 打标约束类型
                    if p >= CONSTRAINT_CONFIG['hard_threshold']:
                        constraint_type = 'hard'
                    elif p >= CONSTRAINT_CONFIG['soft_threshold']:
                        constraint_type = 'soft'
                    else:
                        constraint_type = 'weak'
                    prob_list.append({
                        'target_val': target_val,
                        'prob': round(p, 4),
                        'constraint_type': constraint_type
                    })
            # 归一化概率（避免过滤后和不为1）
            total_prob = sum(item['prob'] for item in prob_list)
            for item in prob_list:
                item['prob'] = round(item['prob'] / total_prob, 4)
                # 写入规则
                joint_constraint_rules.append({
                    '联合特征': ', '.join(feat_group),
                    '特征值组合': ', '.join(map(str, combo)),
                    '目标类别特征': target_cat,
                    '候选取值': item['target_val'],
                    '条件概率': item['prob'],
                    '约束类型': item['constraint_type'],
                    '样本数': sample_cnt
                })
        total_rules += valid_combo_cnt
        print(f"  特征组合{feat_group}有效组合数：{valid_combo_cnt}，生成{len(target_unique_vals)*valid_combo_cnt}条分布规则")

# -------------------------- 4. 生成并保存联合约束表 --------------------------
if joint_constraint_rules:
    joint_constraint_table = pd.DataFrame(joint_constraint_rules)
    # 排序：按目标特征、联合特征、概率降序
    joint_constraint_table = joint_constraint_table.sort_values(
        by=['目标类别特征', '联合特征', '条件概率'],
        ascending=[True, True, False]
    ).reset_index(drop=True)
    # 保存
    joint_constraint_table.to_csv('joint_constraint_table.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 全概率联合约束表构建完成，已保存至：joint_constraint_table.csv")
    print(f"  总规则数：{len(joint_constraint_table)}")
    print(f"  硬约束数：{sum(joint_constraint_table['约束类型'] == 'hard')}")
    print(f"  软约束数：{sum(joint_constraint_table['约束类型'] == 'soft')}")
    print(f"  约束表示例（完整分布）：")
    print(joint_constraint_table.head(10).to_string(index=False))
else:
    print("\n⚠️ 未挖掘到任何有效约束规则，请调低阈值或检查数据分布")


# -------------------------- 2.4.4 增强集的生成（严格对齐论文DIDA类别特征增强） --------------------------
print("\n" + "=" * 100)
print("2.4.4 开始生成增强集（严格对齐论文DIDA类别特征增强）...")
print("=" * 100)

# 增强集生成配置（对齐论文扰动函数参数，完全保留原有端口）
AUGMENTATION_CONFIG = {
    'raw_augmented_path': 'augmented_dataset.csv',  # 数值增强后的D1路径
    'final_augmented_path': 'final_augmented_dataset.csv',  # 最终D2输出路径
    'merge_original_data': True,  # 是否对齐论文定义：D2 = 原始数据集 ∪ 增强样本
    'deduplicate': True,  # 合并后是否去重重复样本
    'fallback_strategy': 'keep'  # 无约束时回退策略：'keep'=保持原值/'marginal'=边缘分布采样
}

# 依赖变量检查（完全保留原有依赖，不增不减）
required_vars = ['numeric_cols', 'categorical_cols', 'target_col', 'label_encoders',
                 'feat_importance_map', 'sample_contribution_map', 'joint_feature_groups',
                 'joint_constraint_table', 'discretization_results', 'ENHANCEMENT_CONFIG']
for var in required_vars:
    if var not in locals():
        raise ValueError(f"变量 '{var}' 未定义，请先运行前面步骤")

# -------------------------- 1. 数据读取与预处理（完全保留原有逻辑） --------------------------
# 1.1 读取数值增强后的中间集D1
try:
    raw_aug_df = pd.read_csv(AUGMENTATION_CONFIG['raw_augmented_path'], encoding='utf-8-sig')
    print(f"✅ 成功读取原始增强集 D1：{len(raw_aug_df)} 行")
except FileNotFoundError:
    raise FileNotFoundError(f"原始增强集文件未找到: {AUGMENTATION_CONFIG['raw_augmented_path']}")

# 1.2 标签列校验与修正
if target_col not in raw_aug_df.columns:
    lower_cols = [col.lower() for col in raw_aug_df.columns]
    target_col_lower = target_col.lower()
    if target_col_lower in lower_cols:
        matched_col = raw_aug_df.columns[lower_cols.index(target_col_lower)]
        raw_aug_df.rename(columns={matched_col: target_col}, inplace=True)
    else:
        last_col = raw_aug_df.columns[-1]
        raw_aug_df.rename(columns={last_col: target_col}, inplace=True)
print(f"✅ 标签列确认: '{target_col}'")

# 1.3 父样本索引校验与处理（用于匹配样本贡献度）
if 'parent_idx' not in raw_aug_df.columns:
    print("⚠️  D1未携带parent_idx字段，将默认使用行号作为父样本索引（可能导致贡献度匹配错误）")
    raw_aug_df['parent_idx'] = raw_aug_df.index
raw_aug_df['parent_idx'] = raw_aug_df['parent_idx'].astype(int)  # 强制转为整数，避免索引匹配失败

# 1.4 D1类别特征转编码（因为联合约束表存储的是编码后的数字，需先对齐类型）
print(f"\n🔄 预处理D1类别特征：文本转编码，对齐约束表格式")
for C in categorical_cols:
    if C == target_col or C not in raw_aug_df.columns:
        continue
    le = label_encoders[C]
    # 处理未知类别，映射为训练集最常见类别
    raw_aug_df[C] = raw_aug_df[C].astype(str)
    unknown_mask = ~raw_aug_df[C].isin(le.classes_)
    if unknown_mask.any():
        most_common = le.classes_[np.argmax(np.bincount(le.transform(le.classes_)))]
        raw_aug_df.loc[unknown_mask, C] = most_common
        print(f"  特征{C}：修复{unknown_mask.sum()}个未知类别，映射为{most_common}")
    raw_aug_df[C] = le.transform(raw_aug_df[C])


# -------------------------- 2. 核心函数：严格对齐论文公式 --------------------------
def compute_p_keep(b_prime_j, ds_prime_i):
    """
    严格对齐论文3.2节差异化扰动函数定义：
    p_keep = (1 - (ϕ1·b'_j + ϕ2·ds'_i))^p + ε
    输出自动截断到[0,1]区间，保证概率合法性
    """
    phi1 = ENHANCEMENT_CONFIG['phi1']
    phi2 = ENHANCEMENT_CONFIG['phi2']
    p = ENHANCEMENT_CONFIG['p']
    epsilon = ENHANCEMENT_CONFIG['epsilon']

    weighted_value = phi1 * b_prime_j + phi2 * ds_prime_i
    base = max(1.0 - weighted_value, 0.0)  # 避免base为负导致复数
    p_keep = (base ** p)
    return float(np.clip(p_keep, 0.0, 1.0))


def process_sample_per_paper(sample, sample_idx):
    """
    严格对齐论文4.3.2节类别特征增强流程：
    输入：D1（数值已增强）样本 xi
    输出：xi_final = [数值特征增强结果, 合规增强的类别特征]
    """
    new_sample = sample.copy()
    fallback_strategy = AUGMENTATION_CONFIG['fallback_strategy']

    # 遍历每个类别特征 C，跳过标签列
    for C in categorical_cols:
        if C == target_col:
            continue

        # ===================== 论文步骤 1：获取双维度价值分数（完全保留） =====================
        b_prime_j = feat_importance_map.get(C, 0.0)
        parent_idx = sample['parent_idx']
        ds_prime_i = sample_contribution_map.get(parent_idx, 0.0)

        # ===================== 论文步骤 2：计算保留原值概率（完全保留） =====================
        p_keep = compute_p_keep(b_prime_j, ds_prime_i)
        u = np.random.uniform(0, 1)

        # ===================== 论文步骤 3：概率决策（完全保留） =====================
        if u >= p_keep:
            new_sample[C] = sample[C]
            continue

        # ===================== 🔧 适配新约束表：低价值特征约束查询（仅修改此部分） =====================
        try:
            # 校验特征是否有关联约束
            if C not in joint_feature_groups or len(joint_feature_groups[C]) == 0:
                new_sample[C] = sample[C]
                continue

            # 兼容新旧joint_feature_groups格式：
            # 旧格式：joint_feature_groups[C] = [feat1, feat2]（单组特征）
            # 新格式：joint_feature_groups[C] = [[feat1], [feat2], [feat1,feat2]]（多组特征）
            feature_groups = joint_feature_groups[C]
            if not isinstance(feature_groups[0], list):
                feature_groups = [feature_groups]

            # 【优化逻辑】按特征组长度降序，优先匹配更长的联合约束（更严格，符合论文约束优先级）
            feature_groups_sorted = sorted(feature_groups, key=lambda x: len(x), reverse=True)
            matched_rows = None

            # 遍历所有特征组，找到第一个匹配的约束
            for A_list in feature_groups_sorted:
                # 构造联合特征组合（原有逻辑完全保留，仅循环遍历多组特征）
                combo = []
                for feat in A_list:
                    if feat not in sample:
                        combo.append('0')
                        continue
                    feat_val = sample[feat]
                    # 数值特征映射到离散化区间
                    if feat in numeric_cols and feat in discretization_results:
                        bins = discretization_results[feat]['bins']
                        bin_idx = np.digitize(feat_val, bins, right=False) - 1
                        bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
                        combo.append(str(bin_idx))
                    # 类别特征直接取编码值
                    elif feat in categorical_cols:
                        combo.append(str(int(feat_val)))
                    else:
                        combo.append(str(feat_val))
                combo_str = ', '.join(combo)
                joint_feat_str = ', '.join(A_list)

                # 查询约束表
                mask = (
                        (joint_constraint_table['目标类别特征'] == C) &
                        (joint_constraint_table['联合特征'] == joint_feat_str) &
                        (joint_constraint_table['特征值组合'] == combo_str)
                )
                matched_rows = joint_constraint_table[mask]
                if not matched_rows.empty:
                    break  # 找到匹配的约束，停止查询更短的特征组

            # 按完整概率分布采样（原有逻辑完全保留，支持新约束表的全概率输出）
            if matched_rows is not None and not matched_rows.empty:
                candidates = matched_rows['候选取值'].values.astype(int)
                probs = matched_rows['条件概率'].values
                probs = probs / probs.sum()  # 二次归一化保证概率合法性
                selected = np.random.choice(candidates, p=probs)
                new_sample[C] = selected
            else:
                # 无匹配约束回退（原有逻辑完全保留）
                new_sample[C] = sample[C] if fallback_strategy == 'keep' else sample[C]

        except Exception as e:
            print(f"⚠️  样本{sample_idx}特征{C}增强失败，保持原值：{str(e)}")
            new_sample[C] = sample[C]

    # 强制标签与原样本一致（完全保留）
    new_sample[target_col] = sample[target_col]
    return new_sample


# -------------------------- 3. 批量生成类别增强样本（完全保留原有逻辑） --------------------------
print(f"\n🚀 开始按论文流程生成类别增强样本，共 {len(raw_aug_df)} 个样本...")
final_samples = []
for idx, row in tqdm(raw_aug_df.iterrows(), total=len(raw_aug_df), desc="生成D2增强样本"):
    augmented_sample = process_sample_per_paper(row, idx)
    final_samples.append(augmented_sample)

category_aug_df = pd.DataFrame(final_samples)
# 删除中间字段parent_idx，避免后续训练干扰
category_aug_df = category_aug_df.drop(columns=['parent_idx'], errors='ignore')
final_aug_df = category_aug_df


# 7. 关键步骤：统一转换为文本格式
print("\n【统一转换为文本格式】")


def ensure_text_format(df, categorical_cols, label_encoders, target_col):
    """确保所有类别特征都是文本格式（新增强制类型转换）"""
    df_copy = df.copy()

    for col in categorical_cols:
        if col in df_copy.columns:
            sample_value = df_copy[col].iloc[0] if len(df_copy) > 0 else None

            if pd.api.types.is_numeric_dtype(df_copy[col]) or (
                    isinstance(sample_value, (int, float, np.number)) and not pd.isna(sample_value)):
                print(f"转换数字到文本: {col}")
                if col in label_encoders:
                    le = label_encoders[col]
                    try:
                        # -------------------------- 新增：强制类型处理 start --------------------------
                        # 1. 先处理NaN：用0填充（或根据业务用中位数/众数，这里选0因为在范围0~len(le.classes_)-1内）
                        if df_copy[col].isna().any():
                            print(f"  提示：{col} 列存在NaN，已用0填充")
                            df_copy[col] = df_copy[col].fillna(0)

                        # 2. 强制转换类型：先转float（处理字符串数字如'1.0'），再转int（符合inverse_transform要求）
                        # 用pd.to_numeric确保先转为数值类型，errors='coerce'把无法转换的转为NaN（后续再填充）
                        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)  # 二次处理无法转换的字符
                        df_copy[col] = df_copy[col].astype(float).astype(int)  # 强制归一为int
                        # -------------------------- 新增：强制类型处理 end --------------------------

                        # 重新检查数值范围（处理填充/转换后的数据）
                        valid_mask = df_copy[col].between(0, len(le.classes_) - 1)
                        if valid_mask.all():
                            # 批量转换：此时数据已是int类型，且在范围
                            df_copy[col] = le.inverse_transform(df_copy[col])
                            print(f"  {col} 列批量转换成功（强制转为int后无异常）")
                        else:
                            # 仍有超出范围的值，走逐行安全转换
                            def safe_inverse_transform(x):
                                try:
                                    if 0 <= x < len(le.classes_):
                                        return le.inverse_transform([int(x)])[0]
                                    else:
                                        return le.classes_[0]
                                except:
                                    return le.classes_[0]

                            df_copy[col] = df_copy[col].apply(safe_inverse_transform)
                            print(f"  {col} 列存在异常值，已逐行安全转换")

                    except Exception as e:
                        print(f"  转换 {col} 失败: {str(e)}，尝试逐行转换")

                        # 批量转换失败后，退回到逐行转换（保底方案）
                        def safe_inverse_transform_fallback(x):
                            try:
                                #  fallback：先强制转int，再尝试转换
                                x_int = int(float(x)) if not pd.isna(x) else 0
                                if 0 <= x_int < len(le.classes_):
                                    return le.inverse_transform([x_int])[0]
                                else:
                                    return le.classes_[0]
                            except:
                                return le.classes_[0]

                        df_copy[col] = df_copy[col].apply(safe_inverse_transform_fallback)
                else:
                    # 无编码器：强制转为字符串（避免数字字符串如'1'看起来像数字）
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0).astype(int).astype(str)
                    print(f"  {col} 无编码器，已强制转为字符串格式")
            else:
                print(f"已是文本格式: {col}")

    # 处理目标列（同样新增强制类型处理）
    if target_col in df_copy.columns and target_col in categorical_cols:
        if target_col in label_encoders:
            le_target = label_encoders[target_col]
            if pd.api.types.is_numeric_dtype(df_copy[target_col]) or (
                    isinstance(df_copy[target_col].iloc[0], (int, float, np.number)) and not pd.isna(
                    df_copy[target_col].iloc[0])):
                print(f"转换目标列 {target_col} 到文本")
                try:
                    # 目标列同样强制处理类型
                    df_copy[target_col] = pd.to_numeric(df_copy[target_col], errors='coerce').fillna(0).astype(
                        float).astype(int)
                    valid_mask_target = df_copy[target_col].between(0, len(le_target.classes_) - 1)
                    if valid_mask_target.all():
                        df_copy[target_col] = le_target.inverse_transform(df_copy[target_col])
                    else:
                        df_copy[target_col] = df_copy[target_col].apply(
                            lambda x: le_target.inverse_transform([int(x)])[0] if 0 <= x < len(le_target.classes_) else
                            le_target.classes_[0])
                except Exception as e:
                    print(f"  目标列 {target_col} 转换失败: {str(e)}，已逐行修复")
                    df_copy[target_col] = df_copy[target_col].apply(lambda x: le_target.classes_[0] if pd.isna(x) else (
                        le_target.inverse_transform([int(float(x))])[0] if 0 <= int(float(x)) < len(
                            le_target.classes_) else le_target.classes_[0]))

    return df_copy


# 执行格式统一


# 1. 先执行原有的文本格式转换
final_aug_df = ensure_text_format(final_aug_df, categorical_cols, label_encoders, target_col)


# 2. 新增：二次扫描类别列，修复残留数字（核心函数）
def scan_and_fix_numeric_in_cate_cols(df, categorical_cols, label_encoders, target_col):
    """
    扫描类别特征列，将残留的数字（数值型/数字字符串）替换为对应的类别文本
    :param df: 处理后的数据集
    :param categorical_cols: 类别特征列列表
    :param label_encoders: 类别编码器字典（含数字→文本映射）
    :param target_col: 目标列（单独处理，确保一致性）
    :return: 无数字残留的数据集
    """
    df_copy = df.copy()
    # 合并需要处理的列：类别列 + 目标列（若目标列是类别列）
    cols_to_scan = categorical_cols.copy()
    if target_col in df_copy.columns and target_col not in cols_to_scan:
        cols_to_scan.append(target_col)

    print("\n=== 开始二次扫描类别列，修复残留数字 ===")
    for col in cols_to_scan:
        if col not in df_copy.columns:
            print(f"  列 {col} 不存在，跳过")
            continue
        if col not in label_encoders:
            print(f"  列 {col} 无对应编码器，无法映射数字→文本，跳过")
            continue

        le = label_encoders[col]  # 获取该列的编码器（关键：含数字→文本映射）
        col_data = df_copy[col].copy()
        fixed_count = 0  # 统计修复的数字数量

        # -------------------------- 核心：判断并修复数字 --------------------------
        def fix_numeric_value(val):
            nonlocal fixed_count  # 允许修改外部变量fixed_count
            try:
                # 步骤1：判断是否为数字（覆盖数值型、数字字符串，如 1、'2'、'3.0'）
                # 先尝试转为整数（因LabelEncoder编码通常是整数），无法转则视为非数字
                val_int = int(float(val))  # 先转float（处理'3.0'），再转int（匹配编码）

                # 步骤2：用编码器反向映射数字→文本（确保在有效范围）
                if 0 <= val_int < len(le.classes_):
                    fixed_val = le.inverse_transform([val_int])[0]
                    fixed_count += 1
                    return fixed_val
                else:
                    # 数字超出编码器范围：返回编码器第一个类别（兜底，避免异常）
                    print(
                        f"  列 {col}：数字 {val_int} 超出编码器范围（0~{len(le.classes_) - 1}），用默认类别 {le.classes_[0]} 替换")
                    fixed_count += 1
                    return le.classes_[0]
            except (ValueError, TypeError, pd.errors.EmptyDataError):
                # 无法转为数字（如 '男'、'未知'），直接返回原值
                return val

        # -------------------------- 执行修复 --------------------------
        # 对列中每个元素应用修复函数（批量处理）
        df_copy[col] = col_data.apply(fix_numeric_value)

        # -------------------------- 打印修复结果 --------------------------
        if fixed_count > 0:
            print(f"  列 {col}：修复了 {fixed_count} 个数字，已替换为对应类别文本")
        else:
            print(f"  列 {col}：未检测到残留数字，无需修复")

    print("=== 二次扫描修复完成 ===")
    return df_copy


# 3. 调用二次扫描函数，确保类别列无数字残留
final_aug_df = scan_and_fix_numeric_in_cate_cols(final_aug_df, categorical_cols, label_encoders, target_col)
# 8. 验证结果
print(f"\n【结果验证】")
print(f"最终数据集形状: {final_aug_df.shape}")
print(f"列名: {final_aug_df.columns.tolist()}")

print(f"\n数据类型检查:")
for col in final_aug_df.columns:
    dtype = final_aug_df[col].dtype
    sample_val = final_aug_df[col].iloc[0] if len(final_aug_df) > 0 else None
    print(f"  {col}: {dtype}, 示例: {sample_val}")

print(f"\n类别特征唯一值:")
for col in categorical_cols:
    if col in final_aug_df.columns:
        unique_vals = final_aug_df[col].unique()
        print(f"  {col}: {list(unique_vals)}")

# 9. 保存最终结果
final_aug_df.to_csv(AUGMENTATION_CONFIG['final_augmented_path'], index=False, encoding='utf-8-sig')
print(f"\n最终增强集已保存: {AUGMENTATION_CONFIG['final_augmented_path']}")

# 10. 显示样本示例
print(f"\n最终增强集前5行示例:")
display_cols = [col for col in list(numeric_cols) + list(categorical_cols) + [target_col] if
                col in final_aug_df.columns]
print(final_aug_df[display_cols].head(10).to_string(index=False))
# -------------------------- 新增：DIDA增强耗时统计 + 合并到现有CSV --------------------------
# 计算总耗时
import os
dida_elapsed_time = round(time.time() - dida_start_time, 2)
print(f"\n✅ DIDA增强总耗时：{dida_elapsed_time} 秒")

# 构造DIDA的耗时记录
dida_time_record = pd.DataFrame({
    "增强方法名称": ["DIDA"],
    "总耗时(秒)": [dida_elapsed_time]
})

# 读取已有的增强耗时统计CSV（如果存在）
time_csv_path = "增强方法耗时统计.csv"
if os.path.exists(time_csv_path):
    # 读取现有CSV，合并DIDA数据，去重（避免重复运行重复添加）
    existing_time_df = pd.read_csv(time_csv_path, encoding='utf-8-sig')
    # 先删除已有的DIDA记录（如果有），再追加新记录
    existing_time_df = existing_time_df[existing_time_df["增强方法名称"] != "DIDA"]
    merged_time_df = pd.concat([existing_time_df, dida_time_record], ignore_index=True)
else:
    # CSV不存在，直接创建新的
    merged_time_df = dida_time_record

# 按耗时从小到大排序
merged_time_df = merged_time_df.sort_values(by="总耗时(秒)", ascending=True).reset_index(drop=True)

# 保存回CSV
merged_time_df.to_csv(time_csv_path, index=False, encoding='utf-8-sig')
print(f"✅ DIDA耗时已追加到 {os.path.abspath(time_csv_path)}")
print("\n更新后的增强方法耗时统计：")
print(merged_time_df.to_string(index=False))
# -------------------------- 步骤6：基于增强数据集重新训练模型并评估性能 --------------------------
print("\n" + "=" * 100)
print("【步骤6：重新训练模型与性能评估（使用处理后的最终增强集）】")
print("=" * 100)

# -------------------------- 1. 加载处理后的最终增强集 --------------------------
try:
    final_aug_df = pd.read_csv('final_augmented_dataset.csv', encoding='utf-8-sig')
    print(f"成功加载最终增强集：{len(final_aug_df)} 个样本")
except FileNotFoundError:
    raise FileNotFoundError("最终增强集文件未找到，请先执行数据增强步骤")
except Exception as e:
    raise RuntimeError(f"加载增强集失败：{str(e)}")

# -------------------------- 2. 处理最终增强集的特征编码（与前置预处理逻辑一致） --------------------------
print("\n对最终增强集进行特征编码...")

# 确保增强集特征与原始特征一致
X_aug = final_aug_df[X.columns].copy() if all(col in final_aug_df.columns for col in X.columns) else None
y_aug = final_aug_df[target_col].copy() if target_col in final_aug_df.columns else None

if X_aug is None or y_aug is None:
    raise ValueError(f"增强集缺少必要列！需包含所有原始特征列和目标列'{target_col}'")

# 2.1 类别特征编码（复用前置代码的label_encoders）
for col in categorical_cols:
    if col in X_aug.columns and col in label_encoders:
        le = label_encoders[col]
        current_values = X_aug[col].astype(str)

        # 处理未知值（映射为最常见类别）
        unknown_mask = ~current_values.isin(le.classes_)
        if unknown_mask.any():
            print(f"警告：特征 {col} 中有 {unknown_mask.sum()} 个未知值，映射为训练集最常见类别")
            most_common = le.inverse_transform([np.argmax(np.bincount(le.transform(le.classes_)))]).item()
            current_values[unknown_mask] = most_common

        X_aug[col] = le.transform(current_values)

# 2.2 数值特征填充（复用前置代码的numeric_imputer）
if numeric_cols and numeric_imputer is not None:
    X_aug[numeric_cols] = numeric_imputer.transform(X_aug[numeric_cols])

# 2.3 目标变量编码（复用前置代码的y_encoder）
if y_aug.dtype == object and y_encoder is not None:
    unknown_y_mask = ~y_aug.isin(y_encoder.classes_)
    if unknown_y_mask.any():
        print(f"警告：目标变量中有 {unknown_y_mask.sum()} 个未知值，映射为训练集最常见类别")
        most_common_y = y_encoder.inverse_transform([np.argmax(np.bincount(y))]).item()
        y_aug[unknown_y_mask] = most_common_y
    y_aug = y_encoder.transform(y_aug)

# -------------------------- 3. 合并训练集（原始筛选集 + 增强集）并划分 --------------------------
print("\n合并原始筛选数据与最终增强集，并划分训练集/验证集...")

# 原始筛选集编码（确保与增强集编码逻辑一致）
X_filtered_encoded = X_filtered.copy()
for col in categorical_cols:
    if col in X_filtered_encoded.columns and col in label_encoders:
        le = label_encoders[col]
        current_values = X_filtered_encoded[col].astype(str)
        unknown_mask = ~current_values.isin(le.classes_)
        if unknown_mask.any():
            most_common = le.inverse_transform([np.argmax(np.bincount(le.transform(le.classes_)))]).item()
            current_values[unknown_mask] = most_common
        X_filtered_encoded[col] = le.transform(current_values)

# 数值特征填充
if numeric_cols and numeric_imputer is not None:
    X_filtered_encoded[numeric_cols] = numeric_imputer.transform(X_filtered_encoded[numeric_cols])

# 合并数据集
X_train_full = pd.concat([X_filtered, X_aug], ignore_index=True)
y_train_full = pd.concat([y_filtered, y_aug], ignore_index=True)
print(f"完整增强数据集规模：{len(X_train_full)} 个样本（原始筛选：{len(X_train)} + 增强：{len(X_aug)}）")

# -------------------------- 4. 重新训练模型（基础XGBoost模型） --------------------------
print("\n【重新训练基础XGBoost模型】")

# 使用最佳参数重新训练
aug_model = xgb.XGBClassifier(**search.best_params_)
aug_model.set_params(
    random_state=seed,
    use_label_encoder=False,
    early_stopping_rounds=30,
    verbose=False
)

# 训练模型
aug_model.fit(
    X_train_full, y_train_full,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"训练完成！最佳迭代轮数：{aug_model.best_iteration}")

# 【你现有的代码，完全不动】
# -------------------------- 5. 准备所有数据集和模型 --------------------------
print("\n" + "=" * 100)
print("【准备多模型与多数据集对比】")
print("=" * 100)
from pytorch_tabnet.tab_model import TabNetClassifier  # 必须加！

# 定义所有要对比的数据集
datasets = {
    "原始数据集": (X_train, y_train),
    "自定义增强数据集": (X_train_full, y_train_full)
}

# 加载其他增强数据集
aug_datasets = [
    ("增强数据集1_SMOTE_原始格式.csv", "SMOTE增强数据集"),
    ("增强数据集2_CTGAN_原始格式.csv", "CTGAN增强数据集"),
    ("增强数据集3_SMOTE-CDNN_原始格式.csv", "SMOTE-CDNN增强数据集"),
    ("增强数据集4_TabDDPM_原始格式.csv", "TabDDPM增强数据集"),
    ("增强数据集5_MDLM_原始格式.csv", "MDLM增强数据集"),
    ("增强数据集6_TANDEM_原始格式.csv", "TANDEM增强数据集")
    # ↑↑↑ 你现有代码到这里结束，下面直接粘贴新增代码 ↑↑↑
]
# ==================================================
# 🔹 增强数据集质量度量（直接插在aug_datasets列表后）
# ==================================================
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
import seaborn as sns
# -------------------------- 配置（按需调整即可） --------------------------
METRIC_CONFIG = {
    "num_bins": 10,  # 数值特征离散化分箱数（和之前约束挖掘逻辑对齐）
    "unique_sim_threshold": 0.95,  # 样本独特性判断阈值：相似度>0.95视为重复样本
    "output_metric_path": "aug_dataset_quality_metrics.csv",  # 数据集级指标输出
    "output_feature_metric_path": "aug_feature_level_metrics.csv",  # 特征级指标输出
    "output_class_metric_path": "aug_class_level_metrics.csv",  # 类内指标输出
    "fig_size": (14, 7)
}


# -------------------------- 复用已有预处理逻辑（和后续训练对齐） --------------------------
def preprocess_aug_dataset(file_path, aug_name):
    """预处理增强数据集，格式和训练集完全一致，同时加入datasets字典供后续训练用"""
    try:
        # 读取数据+跳过第一列ID（和你之前逻辑完全对齐）
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='gbk')
        if df.shape[1] > 1:
            df = df.iloc[:, 1:]

        # 校验目标列
        if target_col not in df.columns:
            print(f"❌ 增强集 {aug_name} 缺失目标列{target_col}，跳过")
            return None, None

        # 分离特征标签+预处理
        X_aug = df.drop(columns=[target_col])
        y_aug = df[target_col]
        y_aug = y_encoder.transform(y_aug)

        # 数值特征填充
        if numeric_cols:
            X_aug[numeric_cols] = numeric_imputer.transform(X_aug[numeric_cols])
        # 类别特征填充+编码
        if categorical_cols:
            X_aug[categorical_cols] = cat_imputer.transform(X_aug[categorical_cols])
            for col in categorical_cols:
                le = label_encoders[col]
                X_aug[col] = X_aug[col].astype(str)
                mask = ~X_aug[col].isin(le.classes_)
                if mask.any():
                    X_aug.loc[mask, col] = le.classes_[0]
                X_aug[col] = le.transform(X_aug[col])

        # 强制列顺序和训练集一致
        X_aug = X_aug[X_train.columns.tolist()]
        # 加入训练用的datasets字典
        datasets[aug_name] = (X_aug, y_aug)
        print(f"✅ 加载增强集 {aug_name}，样本量：{len(X_aug)}，已加入训练队列")
        return X_aug, y_aug
    except Exception as e:
        print(f"❌ 增强集 {aug_name} 加载失败：{str(e)}，跳过")
        return None, None


# -------------------------- 指标计算工具函数 --------------------------
def get_feature_bins(X, numeric_cols, num_bins=10):
    """基于原始训练集生成统一分箱边界，所有增强集共用"""
    bins = {}
    for feat in numeric_cols:
        _, bin_edges = pd.qcut(X[feat], q=num_bins, retbins=True, duplicates='drop')
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        bins[feat] = bin_edges
    return bins


def calc_js_divergence(p, q):
    """计算JS散度，对称化KL，值域[0,1]"""
    return jensenshannon(p, q, base=2) ** 2  # 平方后和KL值域对齐


def calc_wasserstein_distance(p_data, q_data, bins):
    """计算数值特征的Wasserstein距离"""
    p_binned = np.digitize(p_data, bins, right=False)
    q_binned = np.digitize(q_data, bins, right=False)
    p_counts = np.bincount(p_binned, minlength=len(bins))
    q_counts = np.bincount(q_binned, minlength=len(bins))
    p_dist = p_counts / p_counts.sum()
    q_dist = q_counts / q_counts.sum()
    return wasserstein_distance(np.arange(len(bins)), np.arange(len(bins)), p_dist, q_dist)


def calc_entropy(data, bins=None):
    """计算特征信息熵，数值特征需先分箱"""
    if bins is not None:
        binned = np.digitize(data, bins, right=False)
        counts = np.bincount(binned, minlength=len(bins))
    else:
        counts = np.bincount(data.astype(int))
    probs = counts / counts.sum()
    return entropy(probs, base=2)


def calc_coverage(p_data, q_unique):
    """计算增强集特征取值覆盖率：p的取值覆盖q的比例"""
    p_unique = np.unique(p_data)
    return len(np.intersect1d(p_unique, q_unique)) / len(q_unique)


def calc_sample_uniqueness(X_aug, X_train, threshold=0.95):
    """计算样本独特性：和原始训练集相似度低于阈值的样本比例"""
    # 归一化后计算余弦相似度
    X_aug_norm = (X_aug - X_train.min()) / (X_train.max() - X_train.min() + 1e-10)
    X_train_norm = (X_train - X_train.min()) / (X_train.max() - X_train.min() + 1e-10)
    sim_matrix = cosine_similarity(X_aug_norm, X_train_norm)
    max_sim = sim_matrix.max(axis=1)
    unique_ratio = (max_sim < threshold).mean()
    return unique_ratio


# -------------------------- 主计算逻辑 --------------------------
print("\n" + "=" * 100)
print("开始计算增强数据集质量指标...")
print("=" * 100)

# 预处理所有增强集
all_datasets = datasets.copy()  # 包含原始数据集和自定义增强集
for aug_path, aug_name in aug_datasets:
    X_aug, y_aug = preprocess_aug_dataset(aug_path, aug_name)
    if X_aug is not None:
        all_datasets[aug_name] = (X_aug, y_aug)

# 准备基准：原始训练集+统一分箱
X_base, y_base = all_datasets["原始数据集"]
feature_bins = get_feature_bins(X_base, numeric_cols, num_bins=METRIC_CONFIG['num_bins'])
all_feats = numeric_cols + categorical_cols

# 结果存储
dataset_metrics = []  # 数据集级指标
feature_metrics = []  # 特征级指标
class_metrics = []  # 类内指标

# 遍历所有数据集计算
for dataset_name, (X, y) in tqdm(all_datasets.items(), desc="计算质量指标"):
    print(f"\n📍 处理数据集：{dataset_name}")
    # -------------------------- 1. 数据集级指标 --------------------------
    total_js = 0.0
    total_wd = 0.0
    total_entropy = 0.0
    total_coverage = 0.0
    valid_feat_cnt = 0

    # 计算每个特征的指标
    for feat in all_feats:
        p_data = X[feat].values
        q_data = X_base[feat].values
        bins = feature_bins.get(feat, None)
        feat_type = "numeric" if feat in numeric_cols else "categorical"

        # 分布一致性指标
        if feat in numeric_cols:
            # 数值特征用分箱后分布计算
            q_binned = np.digitize(q_data, bins, right=False)
            p_binned = np.digitize(p_data, bins, right=False)
            q_counts = np.bincount(q_binned, minlength=len(bins))
            p_counts = np.bincount(p_binned, minlength=len(bins))
            q_dist = q_counts / q_counts.sum()
            p_dist = p_counts / p_counts.sum()
            js = calc_js_divergence(p_dist, q_dist)
            wd = calc_wasserstein_distance(p_data, q_data, bins)
        else:
            # 类别特征用全类别分布计算
            q_unique = np.unique(q_data)
            q_counts = np.array([np.sum(q_data == c) for c in q_unique])
            p_counts = np.array([np.sum(p_data == c) for c in q_unique])
            q_dist = q_counts / q_counts.sum()
            p_dist = p_counts / p_counts.sum()
            js = calc_js_divergence(p_dist, q_dist)
            wd = wasserstein_distance(q_unique, q_unique, p_dist, q_dist)

        # 多样性指标
        ent = calc_entropy(p_data, bins=bins)
        coverage = calc_coverage(p_data, np.unique(q_data))

        # 累加
        total_js += js
        total_wd += wd
        total_entropy += ent
        total_coverage += coverage
        valid_feat_cnt += 1

        # 保存特征级指标
        feature_metrics.append({
            "数据集名称": dataset_name,
            "特征名称": feat,
            "特征类型": feat_type,
            "JS散度(vs原始)": round(js, 4),
            "Wasserstein距离(vs原始)": round(wd, 4),
            "信息熵": round(ent, 4),
            "特征取值覆盖率": round(coverage, 4)
        })

    # 平均数据集级指标
    avg_js = round(total_js / valid_feat_cnt, 4)
    avg_wd = round(total_wd / valid_feat_cnt, 4)
    avg_entropy = round(total_entropy / valid_feat_cnt, 4)
    avg_coverage = round(total_coverage / valid_feat_cnt, 4)

    # 计算样本独特性（仅增强集计算，原始集跳过）
    uniqueness = 1.0 if dataset_name == "原始数据集" else round(
        calc_sample_uniqueness(X, X_base, METRIC_CONFIG['unique_sim_threshold']), 4)

    # -------------------------- 2. 类内多样性指标 --------------------------
    for cls in np.unique(y_base):
        cls_mask = (y == cls)
        if cls_mask.sum() < 5:
            continue
        cls_entropy = 0.0
        for feat in all_feats:
            bins = feature_bins.get(feat, None)
            cls_entropy += calc_entropy(X[feat][cls_mask].values, bins=bins)
        avg_cls_entropy = round(cls_entropy / valid_feat_cnt, 4)
        class_metrics.append({
            "数据集名称": dataset_name,
            "类别": label_encoders[target_col].inverse_transform([cls])[0] if target_col in label_encoders else cls,
            "类别样本量": cls_mask.sum(),
            "类内平均信息熵": avg_cls_entropy
        })

    # 保存数据集级指标
    dataset_metrics.append({
        "数据集名称": dataset_name,
        "样本量": len(X),
        "平均JS散度(vs原始)": avg_js,
        "平均Wasserstein距离(vs原始)": avg_wd,
        "平均信息熵": avg_entropy,
        "平均特征取值覆盖率": avg_coverage,
        "样本独特性": uniqueness,
        # 综合得分：归一化后加权，越高越好（权重可根据需求调整）
        "综合质量得分": round(
            0.4 * (1 - avg_js / avg_js if avg_js > 0 else 1) +  # 分布一致性占40%
            0.3 * avg_entropy / max([m['平均信息熵'] for m in dataset_metrics] + [avg_entropy]) +  # 多样性占30%
            0.2 * avg_coverage +  # 覆盖率占20%
            0.1 * uniqueness  # 独特性占10%
            , 4)
    })
    print(f"✅  {dataset_name} 综合质量得分：{dataset_metrics[-1]['综合质量得分']}")

# -------------------------- 保存结果 --------------------------
# 保存CSV
dataset_metric_df = pd.DataFrame(dataset_metrics).sort_values(by="综合质量得分", ascending=False).reset_index(drop=True)
feature_metric_df = pd.DataFrame(feature_metrics)
class_metric_df = pd.DataFrame(class_metrics)

dataset_metric_df.to_csv(METRIC_CONFIG['output_metric_path'], index=False, encoding='utf-8-sig')
feature_metric_df.to_csv(METRIC_CONFIG['output_feature_metric_path'], index=False, encoding='utf-8-sig')
class_metric_df.to_csv(METRIC_CONFIG['output_class_metric_path'], index=False, encoding='utf-8-sig')
print(f"\n✅ 数据集级质量指标已保存：{METRIC_CONFIG['output_metric_path']}")
print(f"✅ 特征级质量指标已保存：{METRIC_CONFIG['output_feature_metric_path']}")
print(f"✅ 类内质量指标已保存：{METRIC_CONFIG['output_class_metric_path']}")

# -------------------------- 可视化生成 --------------------------
print("\n🎨 生成质量对比可视化...")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 图1：综合质量得分对比
plt.figure(figsize=METRIC_CONFIG['fig_size'])
sorted_score = dataset_metric_df.set_index("数据集名称")["综合质量得分"].sort_values()
sorted_score.plot(kind='barh', color='#2ca02c', edgecolor='black')
plt.title("增强数据集综合质量得分对比（越高越好）", fontsize=14, pad=20)
plt.xlabel("综合质量得分", fontsize=12)
plt.ylabel("数据集", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
for i, v in enumerate(sorted_score.values):
    plt.text(v + 0.01, i, f"{v:.4f}", va='center', fontsize=10)
plt.tight_layout()
plt.savefig("aug_quality_score.png", dpi=300, bbox_inches='tight')
plt.close()

# 图2：分布一致性对比
plt.figure(figsize=METRIC_CONFIG['fig_size'])
x = np.arange(len(dataset_metric_df))
width = 0.35
plt.bar(x - width / 2, dataset_metric_df["平均JS散度(vs原始)"], width, label='平均JS散度（越低越好）', color='#1f77b4')
plt.bar(x + width / 2,
        dataset_metric_df["平均Wasserstein距离(vs原始)"] / dataset_metric_df["平均Wasserstein距离(vs原始)"].max(),
        width, label='归一化Wasserstein距离（越低越好）', color='#ff7f0e')
plt.xticks(x, dataset_metric_df["数据集名称"], rotation=45, ha='right')
plt.title("增强数据集分布一致性对比", fontsize=14, pad=20)
plt.ylabel("归一化指标值", fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("aug_distribution_consistency.png", dpi=300, bbox_inches='tight')
plt.close()

# 图3：多样性对比
plt.figure(figsize=METRIC_CONFIG['fig_size'])
x = np.arange(len(dataset_metric_df))
width = 0.25
plt.bar(x - width, dataset_metric_df["平均信息熵"] / dataset_metric_df["平均信息熵"].max(), width,
        label='归一化平均信息熵（越高越好）', color='#9467bd')
plt.bar(x, dataset_metric_df["平均特征取值覆盖率"], width, label='特征取值覆盖率（越高越好）', color='#8c564b')
plt.bar(x + width, dataset_metric_df["样本独特性"], width, label='样本独特性（越高越好）', color='#e377c2')
plt.xticks(x, dataset_metric_df["数据集名称"], rotation=45, ha='right')
plt.title("增强数据集多样性对比", fontsize=14, pad=20)
plt.ylabel("指标值", fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("aug_diversity_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# 图4：特征级信息熵热力图
plt.figure(figsize=(12, 8))
entropy_pivot = feature_metric_df.pivot(index="特征名称", columns="数据集名称", values="信息熵")
sns.heatmap(entropy_pivot, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0.5)
plt.title("各数据集特征信息熵热力图（越深越高）", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig("aug_feature_entropy_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ 所有可视化图表已保存")

# -------------------------- 核心结论输出 --------------------------
print("\n" + "=" * 100)
print("✨ 增强数据集质量评估完成！核心结论：")
print("=" * 100)
best_dataset = dataset_metric_df.iloc[0]
print(f"1. 综合质量最高的增强集：{best_dataset['数据集名称']}（综合得分：{best_dataset['综合质量得分']:.4f}）")
print(f"   分布一致性：平均JS散度={best_dataset['平均JS散度(vs原始)']:.4f}，样本独特性={best_dataset['样本独特性']:.2%}")
worst_js = dataset_metric_df.loc[dataset_metric_df["平均JS散度(vs原始)"].idxmax()]
print(f"2. 分布偏离最严重的增强集：{worst_js['数据集名称']}（平均JS散度={worst_js['平均JS散度(vs原始)']:.4f}）")
low_coverage = dataset_metric_df.loc[dataset_metric_df["平均特征取值覆盖率"].idxmin()]
print(f"3. 特征丢失最严重的增强集：{low_coverage['数据集名称']}（平均覆盖率={low_coverage['平均特征取值覆盖率']:.2%}）")



for aug_path, ds_name in aug_datasets:
    try:
        aug_df = pd.read_csv(aug_path, encoding='utf-8-sig')
        print(f"成功加载{ds_name}：{len(aug_df)} 个样本")

        # ------------------- 新增调试代码 -------------------
        # 1. 检查目标列是否缺失
        missing_target = target_col not in aug_df.columns
        # 2. 检查特征列中缺失的具体列
        missing_features = [col for col in X.columns if col not in aug_df.columns]
        # 3. 打印缺失信息
        if missing_target:
            print(f"  调试：{ds_name}缺失目标列 → {target_col}")
        if missing_features:
            print(f"  调试：{ds_name}缺失特征列 → {missing_features}")
        # ---------------------------------------------------

        # 原校验逻辑
        X_aug_current = aug_df[X.columns].copy() if all(col in aug_df.columns for col in X.columns) else None
        y_aug_current = aug_df[target_col].copy() if target_col in aug_df.columns else None

        if X_aug_current is None or y_aug_current is None:
            print(f"警告：{ds_name}缺少必要列，跳过该数据集")
            continue

        # 类别特征编码
        for col in categorical_cols:
            if col in X_aug_current.columns and col in label_encoders:
                le = label_encoders[col]
                current_values = X_aug_current[col].astype(str)
                unknown_mask = ~current_values.isin(le.classes_)
                if unknown_mask.any():
                    most_common = le.inverse_transform([np.argmax(np.bincount(le.transform(le.classes_)))]).item()
                    current_values[unknown_mask] = most_common
                X_aug_current[col] = le.transform(current_values)

        # 数值特征填充
        if numeric_cols and numeric_imputer is not None:
            X_aug_current[numeric_cols] = numeric_imputer.transform(X_aug_current[numeric_cols])

        # 目标变量编码
        if y_aug_current.dtype == object and y_encoder is not None:
            unknown_y_mask = ~y_aug_current.isin(y_encoder.classes_)
            if unknown_y_mask.any():
                most_common_y = y_encoder.inverse_transform([np.argmax(np.bincount(y))]).item()
                y_aug_current[unknown_y_mask] = most_common_y
            y_aug_current = y_encoder.transform(y_aug_current)

        datasets[ds_name] = (X_aug_current, y_aug_current)

    except FileNotFoundError:
        print(f"警告：{aug_path}文件未找到，跳过该数据集")
    except Exception as e:
        print(f"警告：加载{aug_path}失败：{str(e)}，跳过该数据集")

# 定义9个机器学习模型
models = {
    "AdaBoost": AdaBoostClassifier(random_state=seed),
    "决策树": DecisionTreeClassifier(random_state=seed),
    "极端随机树": ExtraTreesClassifier(random_state=seed, n_jobs=-1),
    "梯度提升": GradientBoostingClassifier(random_state=seed),
    "k近邻": KNeighborsClassifier(n_jobs=-1),
    "逻辑回归": LogisticRegression(random_state=seed, n_jobs=-1, max_iter=1000),
    "神经网络": MLPClassifier(random_state=seed, max_iter=500),
    "随机森林": RandomForestClassifier(random_state=seed, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(
        random_state=seed,
        use_label_encoder=False,
        eval_metric='logloss' if n_classes == 2 else 'mlogloss'
    ),
    "TabNet": TabNetClassifier(
        n_d=32,
        n_a=32,
        n_steps=3,  # 别用 6，太复杂容易不收敛
        gamma=1.0,
        n_independent=1,
        n_shared=1,
        mask_type="entmax",  # 比 sparsemax 稳定太多
        lambda_sparse=0.001,
        optimizer_params=dict(lr=2e-3),  # 1e-4 太小，学得太慢
        verbose=1,
        device_name="cpu"
)


}
plt.close('all')
# -------------------------- 6. 多模型多数据集训练与评估 --------------------------
print("\n" + "=" * 100)
print("【多模型多数据集训练与评估】")
print("=" * 100)

# 存储所有性能结果
performance_results = {}


# 评估函数
# 评估函数（修复后）
def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    results = {}

    # 训练模型：区分XGBoost和其他模型
    if isinstance(model, xgb.XGBClassifier):
        # XGBoost需要传入eval_set以支持早停
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],  # 传入验证集
            verbose=False  # 关闭日志输出
        )
    elif 'TabNet' in str(type(model)):
        # 把你的 X 和 y 全部这样改！
        X_train = X_tr.to_numpy()  # 必须！
        y_train = y_tr.to_numpy()  # 必须！
        X_val = X_val.to_numpy()  # 必须！
        y_val = y_val.to_numpy()  # 必须！
        X_test = X_test.to_numpy()  # 必须！
        y_test = y_test.to_numpy()  # 必须！
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            max_epochs=100,  # 必须加
            patience=10,  # 早停
            batch_size=128,

        )


    else:
        # 其他模型正常训练
        model.fit(X_train, y_train)

    # 训练集评估
    if n_classes == 2:
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_train_pred = model.predict(X_train)
        results['train_auc'] = roc_auc_score(y_train, y_train_proba)
        results['train_auc_pr'] = average_precision_score(y_train, y_train_proba)
        results['train_f1'] = f1_score(y_train, y_train_pred)
    else:
        y_train_pred = model.predict(X_train)
        results['train_acc'] = accuracy_score(y_train, y_train_pred)
        results['train_f1'] = f1_score(y_train, y_train_pred, average='weighted')

    # 验证集评估
    if n_classes == 2:
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_val_pred = model.predict(X_val)
        results['val_auc'] = roc_auc_score(y_val, y_val_proba)
        results['val_auc_pr'] = average_precision_score(y_val, y_val_proba)
        results['val_f1'] = f1_score(y_val, y_val_pred)
    else:
        y_val_pred = model.predict(X_val)
        results['val_acc'] = accuracy_score(y_val, y_val_pred)
        results['val_f1'] = f1_score(y_val, y_val_pred, average='weighted')

    # 测试集评估
    if n_classes == 2:
        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = model.predict(X_test)
        results['test_auc'] = roc_auc_score(y_test, y_test_proba)
        results['test_auc_pr'] = average_precision_score(y_test, y_test_proba)
        results['test_f1'] = f1_score(y_test, y_test_pred)
    else:
        y_test_pred = model.predict(X_test)
        results['test_acc'] = accuracy_score(y_test, y_test_pred)
        results['test_f1'] = f1_score(y_test, y_test_pred, average='weighted')

    return results


# 批量训练与评估
for ds_name, (X_tr, y_tr) in datasets.items():
    performance_results[ds_name] = {}
    print(f"\n----- 开始在 {ds_name} 上训练所有模型 -----")

    for model_name, model in models.items():
        print(f"训练 {model_name}...")
        # 深拷贝模型避免参数污染
        from copy import deepcopy

        model_copy = deepcopy(model)

        # 特殊处理XGBoost的早停
        if model_name == "XGBoost":
            model_copy.set_params(early_stopping_rounds=30)
            model_copy.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            results = evaluate_model(model_copy, X_tr, y_tr, X_val, y_val, X_test, y_test)
        else:
            results = evaluate_model(model_copy, X_tr, y_tr, X_val, y_val, X_test, y_test)

        performance_results[ds_name][model_name] = results
        print(f"{model_name} 训练完成")

# -------------------------- 7. 性能对比展示 --------------------------
print("\n" + "=" * 80)
print("【多模型多数据集性能综合对比】")
print("=" * 80)

# 打印测试集对比结果
if n_classes == 2:
    print("\n二分类模型性能对比（测试集AUC-ROC）：")
    print(f"{'数据集':<20} | {'模型':<10} | AUC-ROC | AUC-PR  | F1分数")
    print("-" * 70)
    for ds_name, model_results in performance_results.items():
        for model_name, results in model_results.items():
            print(
                f"{ds_name:<20} | {model_name:<10} | {results['test_auc']:.4f} | {results['test_auc_pr']:.4f} | {results['test_f1']:.4f}")
else:
    print("\n多分类模型性能对比（测试集准确率）：")
    print(f"{'数据集':<20} | {'模型':<10} | 准确率  | 加权F1")
    print("-" * 60)
    for ds_name, model_results in performance_results.items():
        for model_name, results in model_results.items():
            print(f"{ds_name:<20} | {model_name:<10} | {results['test_acc']:.4f} | {results['test_f1']:.4f}")

import csv
import os

# 配置CSV文件名（可自行修改）
csv_file_name = "模型性能对比结果.csv"
# 获取文件绝对路径（方便你找到文件）
save_path = os.path.abspath(csv_file_name)

# 写入CSV文件（自动适配二分类/多分类）
with open(csv_file_name, "w", newline="", encoding="utf-8-sig") as f:
    csv_writer = csv.writer(f)

    # 1. 写入表头（和控制台打印完全对应）
    if n_classes == 2:
        csv_writer.writerow(["数据集", "模型", "AUC-ROC", "AUC-PR", "F1分数"])
    else:
        csv_writer.writerow(["数据集", "模型", "准确率", "加权F1"])

    # 2. 写入所有模型性能数据
    for dataset_name, model_res in performance_results.items():
        for model_name, metrics in model_res.items():
            if n_classes == 2:
                # 二分类指标
                row = [
                    dataset_name,
                    model_name,
                    round(metrics["test_auc"], 4),
                    round(metrics["test_auc_pr"], 4),
                    round(metrics["test_f1"], 4)
                ]
            else:
                # 多分类指标
                row = [
                    dataset_name,
                    model_name,
                    round(metrics["test_acc"], 4),
                    round(metrics["test_f1"], 4)
                ]
            csv_writer.writerow(row)

# 提示保存成功
print(f"\n✅ CSV表格已生成：{save_path}")
print("📌 可用 Excel/WPS 直接打开，支持排序、筛选、绘图做性能对比！")










# -------------------------- 8. 性能对比可视化 --------------------------
print("\n" + "=" * 80)
print("【生成多模型多数据集性能对比可视化图表】")
print("=" * 80)

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 准备可视化数据
ds_names = list(datasets.keys())
model_names = list(models.keys())
metric_name = "AUC-ROC" if n_classes == 2 else "准确率"
metric_key = "test_auc" if n_classes == 2 else "test_acc"

# 绘制条形图
fig, ax = plt.subplots(figsize=(14, 8))
width = 0.8 / len(model_names)
x = np.arange(len(ds_names))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22','#17becf']

for i, model_name in enumerate(model_names):
    scores = [performance_results[ds][model_name][metric_key] for ds in ds_names]
    rects = ax.bar(
        x - 0.4 + width / 2 + i * width,
        scores,
        width,
        label=model_name,
        color=colors[i]
    )

    # 添加数值标签
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# 设置图表属性
ax.set_title(f'不同数据集上各模型的测试集{metric_name}对比', fontsize=14)
ax.set_xlabel('数据集', fontsize=12)
ax.set_ylabel(metric_name, fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(ds_names, rotation=45, ha='right', fontsize=10)
ax.set_ylim(0, 1.0)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()

# 保存图像
output_img = "多模型多数据集性能对比.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"多模型多数据集性能对比图表已保存至：{output_img}")
plt.close()

# 生成F1分数对比图
f1_metric_name = "F1分数" if n_classes == 2 else "加权F1分数"
f1_metric_key = "test_f1"

fig, ax = plt.subplots(figsize=(14, 8))
for i, model_name in enumerate(model_names):
    scores = [performance_results[ds][model_name][f1_metric_key] for ds in ds_names]
    rects = ax.bar(
        x - 0.4 + width / 2 + i * width,
        scores,
        width,
        label=model_name,
        color=colors[i]
    )

    # 添加数值标签
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

ax.set_title(f'不同数据集上各模型的测试集{f1_metric_name}对比', fontsize=14)
ax.set_xlabel('数据集', fontsize=12)
ax.set_ylabel(f1_metric_name, fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(ds_names, rotation=45, ha='right', fontsize=10)
ax.set_ylim(0, 1.0)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
output_img_f1 = "多模型多数据集F1对比.png"
plt.savefig(output_img_f1, dpi=300, bbox_inches='tight')
print(f"多模型多数据集F1对比图表已保存至：{output_img_f1}")
plt.close()

# -------------------------- 新增：单模型多数据集对比可视化 --------------------------
print("\n" + "=" * 80)
print("【生成多模型多数据集性能对比可视化图表】")
print("=" * 80)

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 定义指标参数
if n_classes == 2:
    metrics = [
        {
            "name": "AUC-ROC",
            "key": "test_auc"
        },
        {
            "name": "F1分数",
            "key": "test_f1"
        }
    ]
else:
    metrics = [
        {
            "name": "准确率",
            "key": "test_acc"
        },
        {
            "name": "F1分数",
            "key": "test_f1"
        }
    ]

# 数据集名称列表和颜色映射
ds_names = list(datasets.keys())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # 可扩展更多颜色
color_map = {ds: colors[i % len(colors)] for i, ds in enumerate(ds_names)}

# 模型名称列表
model_names = list(performance_results[ds_names[0]].keys())
x = np.arange(len(model_names))  # 模型索引
width = 0.8 / len(ds_names)  # 每个数据集条形的宽度（根据数据集数量动态调整）

# 为每个指标生成对比图
for metric in metrics:
    fig, ax = plt.subplots(figsize=(12, 6))

    # 为每个数据集绘制条形
    for i, ds in enumerate(ds_names):
        # 提取该数据集上所有模型的指标分数
        scores = [performance_results[ds][model][metric["key"]] for model in model_names]

        # 计算条形位置（同一模型的不同数据集条形并列显示）
        rects = ax.bar(
            x - 0.4 + width / 2 + i * width,  # 位置调整
            scores,
            width,
            label=ds,
            color=color_map[ds]
        )

        # 添加数值标签
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f'{height:.4f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=9
            )

    # 设置图表属性
    ax.set_title(f'不同模型在各数据集上的{metric["name"]}对比', fontsize=14)
    ax.set_xlabel('模型', fontsize=12)
    ax.set_ylabel(metric["name"], fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1.0)  # 指标分数通常在0-1之间
    ax.legend(title="数据集", bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放右侧

    plt.tight_layout()
    output = f'多模型多数据集_{metric["name"]}_对比.png'
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"已保存：{output}")
    plt.close()

print("\n所有多模型多数据集对比图表生成完成！")
print("\n" + "=" * 100)
print("所有模型训练与对比完成！")
print("=" * 100)


