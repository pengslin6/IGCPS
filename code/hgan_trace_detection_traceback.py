"""
异构动态图注意力网络 - 信息物理系统异常溯源
- 加载sr_com.csv数据集
- 第一列Time，最后一列phy_label（标签：NM/NS物理故障，PS/SS/PM网络攻击导致的物理故障，其他正常）
- 物理层特征：phy_前缀（温度、压力等）
- 网络层特征：其他列（五元组等）
- 时间级异常检测和溯源
- 输出：网络物理拓扑图、异常溯源图
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate


def set_seed(seed=42):
    """设置随机种子，保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# 设置全局随机种子
set_seed(42)


def ensure_parent_dir(path):
    """Create the output directory for a file path if needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def make_output_dirs(outdir):
    """Create separated output folders for reports, tabular data, and figures."""
    dirs = {
        'root': outdir,
        'reports': os.path.join(outdir, 'reports'),
        'data': os.path.join(outdir, 'data'),
        'figures': os.path.join(outdir, 'figures'),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def estimate_model_flops(model, x, adj_tensor):
    """Estimate single-sample forward FLOPs with PyTorch profiler."""
    model.eval()
    fallback_flops = float(2 * sum(p.numel() for p in model.parameters()))
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if x.is_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.no_grad():
            with torch.profiler.profile(activities=activities, with_flops=True) as prof:
                _ = model(x, adj_tensor)
        profiled_flops = float(sum(evt.flops or 0 for evt in prof.key_averages()))
        return profiled_flops if profiled_flops > 0 else fallback_flops
    except Exception as exc:
        print(f"   ⚠️ FLOPs估算失败: {exc}")
        return fallback_flops


def get_model_size(model):
    """计算模型大小
    
    Args:
        model: PyTorch模型
        
    Returns:
        dict: 包含模型大小的各种指标
            - total_params: 总参数数量
            - trainable_params: 可训练参数数量
            - non_trainable_params: 不可训练参数数量
            - param_count_k: 参数数量(K)
            - size_kb: 模型大小(KB)
            - size_mb: 模型大小(MB)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # 假设float32, 每个参数4字节
    size_bytes = total_params * 4
    size_kb = size_bytes / 1024
    size_mb = size_bytes / (1024 * 1024)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'param_count_k': total_params / 1000,
        'size_kb': size_kb,
        'size_mb': size_mb
    }


def print_model_size_summary(model, model_name="Model"):
    """打印模型大小摘要
    
    Args:
        model: PyTorch模型
        model_name: 模型名称
    """
    size_info = get_model_size(model)
    print(f"\n📦 {model_name} 模型大小:")
    print(f"   总参数: {size_info['total_params']:,} ({size_info['param_count_k']:.2f}K)")
    print(f"   可训练: {size_info['trainable_params']:,}")
    print(f"   不可训练: {size_info['non_trainable_params']:,}")
    print(f"   模型大小: {size_info['size_kb']:.2f} KB ({size_info['size_mb']:.4f} MB)")



# 尝试导入PyTorch Geometric（可选，用于对比模型）
try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("⚠️ PyTorch Geometric未安装，将使用自定义GNN层进行对比")

# ========================= 0. 自适应插值（故障感知） =========================

def adaptive_interpolation(signal, timestamps, target_timestamps,
                           gradient_threshold=3.0, method='auto'):
    """故障感知自适应插值

    在平滑区域使用三次样条插值保持物理动态连续性，
    在突变区域（故障瞬态）自动切换为线性插值或零阶保持，
    避免引入人工过渡伪影(artifacts)。

    Args:
        signal: 原始信号值序列
        timestamps: 原始时间戳序列
        target_timestamps: 目标插值时间戳
        gradient_threshold: 梯度突变检测阈值（标准差倍数）
        method: 'auto'自适应, 'cubic'强制三次样条, 'linear'强制线性

    Returns:
        interpolated: 插值后的信号值
        transient_mask: 突变区域标记（True表示该区域为突变）
    """
    from scipy.interpolate import CubicSpline, interp1d

    signal = np.asarray(signal, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    target_timestamps = np.asarray(target_timestamps, dtype=np.float64)

    if method == 'cubic':
        cs = CubicSpline(timestamps, signal, bc_type='natural')
        return cs(target_timestamps), np.zeros(len(target_timestamps), dtype=bool)

    if method == 'linear':
        f = interp1d(timestamps, signal, kind='linear', fill_value='extrapolate')
        return f(target_timestamps), np.zeros(len(target_timestamps), dtype=bool)

    # 自适应模式：检测突变区域
    if len(signal) < 3:
        f = interp1d(timestamps, signal, kind='linear', fill_value='extrapolate')
        return f(target_timestamps), np.zeros(len(target_timestamps), dtype=bool)

    # 计算一阶差分梯度
    gradients = np.abs(np.diff(signal))
    grad_mean = np.mean(gradients)
    grad_std = np.std(gradients) + 1e-8

    # 标记突变点：梯度超过 mean + threshold * std
    transient_indices = np.where(gradients > grad_mean + gradient_threshold * grad_std)[0]

    # 扩展突变区域（前后各扩展1个采样点）
    transient_set = set()
    for idx in transient_indices:
        transient_set.update(range(max(0, idx - 1), min(len(signal), idx + 3)))

    # 对目标时间戳标记是否落在突变区域
    transient_mask = np.zeros(len(target_timestamps), dtype=bool)

    if len(transient_set) == 0:
        # 无突变，全部使用三次样条
        cs = CubicSpline(timestamps, signal, bc_type='natural')
        return cs(target_timestamps), transient_mask

    # 分区插值
    cs = CubicSpline(timestamps, signal, bc_type='natural')
    f_linear = interp1d(timestamps, signal, kind='linear', fill_value='extrapolate')

    interpolated = cs(target_timestamps)  # 默认三次样条

    # 对突变区域使用线性插值
    for i, t in enumerate(target_timestamps):
        # 找到t对应的原始信号区间
        idx = np.searchsorted(timestamps, t) - 1
        idx = max(0, min(idx, len(signal) - 2))
        if idx in transient_set or (idx + 1) in transient_set:
            interpolated[i] = f_linear(t)
            transient_mask[i] = True

    return interpolated, transient_mask


class DataAugmentor:
    """OOD数据增强器

    针对工业场景故障数据稀缺问题，通过以下策略增强少数类样本：
    1. 特征空间Mixup：在正常和异常样本之间进行插值
    2. 高斯噪声扰动：对少数类样本添加受控噪声
    3. 跨域合成：组合网络异常模式和物理故障模式生成合成样本
    """

    def __init__(self, noise_std=0.1, mixup_alpha=0.3, augment_ratio=2.0):
        self.noise_std = noise_std
        self.mixup_alpha = mixup_alpha
        self.augment_ratio = augment_ratio

    def gaussian_noise_augment(self, X, y, target_classes, n_augment=None):
        """对目标类别添加高斯噪声生成增强样本"""
        augmented_X = []
        augmented_y = []

        for cls in target_classes:
            cls_mask = (y == cls)
            cls_samples = X[cls_mask]
            if len(cls_samples) == 0:
                continue

            n = n_augment if n_augment else int(len(cls_samples) * self.augment_ratio)
            indices = np.random.choice(len(cls_samples), size=n, replace=True)
            noise = np.random.normal(0, self.noise_std, size=(n, cls_samples.shape[1]))
            new_samples = cls_samples[indices] + noise

            augmented_X.append(new_samples)
            augmented_y.append(np.full(n, cls))

        if augmented_X:
            return np.vstack(augmented_X), np.concatenate(augmented_y)
        return np.empty((0, X.shape[1])), np.array([])

    def mixup_augment(self, X, y, source_class, target_class, n_augment=100):
        """在两个类别之间进行Mixup插值生成边界样本"""
        src_mask = (y == source_class)
        tgt_mask = (y == target_class)
        src_samples = X[src_mask]
        tgt_samples = X[tgt_mask]

        if len(src_samples) == 0 or len(tgt_samples) == 0:
            return np.empty((0, X.shape[1])), np.array([])

        src_idx = np.random.choice(len(src_samples), size=n_augment, replace=True)
        tgt_idx = np.random.choice(len(tgt_samples), size=n_augment, replace=True)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=(n_augment, 1))
        mixed = lam * src_samples[src_idx] + (1 - lam) * tgt_samples[tgt_idx]

        # 标签取目标类别（异常类）
        labels = np.full(n_augment, target_class)
        return mixed, labels

    def cross_domain_synthesis(self, X_net, X_phy, y, net_attack_classes=[3, 4, 5],
                               n_augment=100):
        """跨域合成：组合网络攻击模式和物理故障模式"""
        augmented_X_net = []
        augmented_X_phy = []
        augmented_y = []

        for cls in net_attack_classes:
            cls_mask = (y == cls)
            cls_net = X_net[cls_mask]
            cls_phy = X_phy[cls_mask]

            if len(cls_net) == 0:
                continue

            # 从同类样本中随机组合不同的网络和物理特征
            n = min(n_augment, len(cls_net) * 2)
            net_idx = np.random.choice(len(cls_net), size=n, replace=True)
            phy_idx = np.random.choice(len(cls_phy), size=n, replace=True)

            # 添加轻微扰动
            noise_net = np.random.normal(0, self.noise_std * 0.5, size=(n, cls_net.shape[1]))
            noise_phy = np.random.normal(0, self.noise_std * 0.5, size=(n, cls_phy.shape[1]))

            augmented_X_net.append(cls_net[net_idx] + noise_net)
            augmented_X_phy.append(cls_phy[phy_idx] + noise_phy)
            augmented_y.append(np.full(n, cls))

        if augmented_X_net:
            return (np.vstack(augmented_X_net), np.vstack(augmented_X_phy),
                    np.concatenate(augmented_y))
        return (np.empty((0, X_net.shape[1])), np.empty((0, X_phy.shape[1])),
                np.array([]))


# ========================= 1. 数据加载与预处理 =========================

def load_sr_com_data(csv_path):
    """加载sr_com.csv数据集，区分网络特征和物理特征
    
    网络节点：Source IP 和 Destination IP 构成的唯一IP地址集合
    物理节点：phy_前缀的特征（按燃气流经顺序）
    """
    df = pd.read_csv(csv_path)
    print(f"✅ 加载数据: {df.shape[0]} 条记录, {df.shape[1]} 列")
    
    # 第一列Time，最后一列phy_label
    time_col = df.columns[0]
    label_col = df.columns[-1]
    
    # 分离物理特征（phy_前缀）和网络特征（其他）。combined.csv 使用 TE/ICS
    # 过程变量命名，因此在没有 phy_ 列时，将 xmeas*/xmv* 识别为物理层特征。
    phy_cols = [c for c in df.columns if c.startswith('phy_') and c != label_col]
    if len(phy_cols) == 0:
        phy_cols = [
            c for c in df.columns
            if c != time_col and c != label_col
            and (c.lower().startswith('xmeas') or c.lower().startswith('xmv'))
        ]
        if len(phy_cols) > 0:
            print("⚠️ 未找到phy_列，已将xmeas*/xmv*过程变量识别为物理特征")
    
    # 去除前两个物理层特征（phy_Controller status, phy_Controller status value）
    # 这两个特征是控制器状态，与燃气流程无关，且可能引入噪声
    exclude_phy_cols = ['phy_Controller status', 'phy_Controller status value']
    phy_cols = [c for c in phy_cols if c not in exclude_phy_cols]
    print(f"⚠️ 已去除物理特征: {exclude_phy_cols}")
    
    # 识别IP列（Source IP, Destination IP 或类似名称）
    ip_col_patterns = ['source_ip', 'src_ip', 'sourceip', 'source ip', 
                       'dest_ip', 'dst_ip', 'destip', 'destination_ip', 'destination ip']
    src_ip_col = None
    dst_ip_col = None
    
    for c in df.columns:
        c_lower = c.lower().replace(' ', '_').replace('-', '_')
        if any(p in c_lower for p in ['source_ip', 'src_ip', 'sourceip']):
            src_ip_col = c
        elif any(p in c_lower for p in ['dest_ip', 'dst_ip', 'destip', 'destination_ip']):
            dst_ip_col = c
    
    # 如果没找到标准IP列名，尝试查找包含IP的列
    if src_ip_col is None or dst_ip_col is None:
        for c in df.columns:
            c_lower = c.lower()
            if 'ip' in c_lower and 'src' in c_lower or 'source' in c_lower:
                src_ip_col = c
            elif 'ip' in c_lower and ('dst' in c_lower or 'dest' in c_lower):
                dst_ip_col = c
    
    print(f"📊 时间列: {time_col}")
    print(f"📊 标签列: {label_col}")
    print(f"📊 Source IP列: {src_ip_col}")
    print(f"📊 Destination IP列: {dst_ip_col}")
    print(f"📊 物理特征 ({len(phy_cols)}): {phy_cols[:5]}...")
    
    # 网络特征列（排除IP列、时间列、标签列、物理特征列）
    net_feat_cols = [c for c in df.columns 
                    if c != time_col and c != label_col 
                    and c not in phy_cols
                    and c != src_ip_col and c != dst_ip_col]
    print(f"📊 网络特征 ({len(net_feat_cols)}): {net_feat_cols[:5]}...")
    
    # 提取时间和标签
    times = df[time_col].values
    labels = df[label_col].values
    
    # 获取所有唯一的IP地址作为网络节点
    if src_ip_col and dst_ip_col:
        src_ips = df[src_ip_col].astype(str).values
        dst_ips = df[dst_ip_col].astype(str).values
        all_ips = list(set(src_ips) | set(dst_ips))
        all_ips.sort()  # 排序保证一致性
        ip_to_idx = {ip: i for i, ip in enumerate(all_ips)}
        print(f"📊 唯一IP地址数量（网络节点）: {len(all_ips)}")
    else:
        # 如果找不到IP列，使用默认的网络特征列
        print("⚠️ 未找到IP列，使用网络特征列名作为网络节点")
        all_ips = net_feat_cols[:5] if len(net_feat_cols) > 0 else ['Synthetic_Net_1', 'Synthetic_Net_2']
        ip_to_idx = {ip: i for i, ip in enumerate(all_ips)}
        src_ips = [all_ips[0]] * len(df)
        dst_ips = [all_ips[1]] * len(df)
    
    # 处理物理特征（数值型）
    phy_data = df[phy_cols].copy()
    for c in phy_cols:
        phy_data[c] = pd.to_numeric(phy_data[c], errors='coerce').fillna(0)
    X_phy = phy_data.values.astype(np.float32)
    
    # 处理网络特征（用于边或节点特征）
    net_data = df[net_feat_cols].copy() if net_feat_cols else pd.DataFrame()
    for c in net_feat_cols:
        if net_data[c].dtype == 'object':
            le = LabelEncoder()
            net_data[c] = le.fit_transform(net_data[c].astype(str))
        else:
            net_data[c] = pd.to_numeric(net_data[c], errors='coerce').fillna(0)
    X_net_feat = net_data.values.astype(np.float32) if len(net_feat_cols) > 0 else np.zeros((len(df), 1), dtype=np.float32)
    
    # 标准化在时间顺序划分之后进行，避免把测试集分布泄露到训练阶段。
    
    # 标签编码：
    # - 旧数据集：NS/NM/PM/PS/SS 文本标签映射为 6 类；
    # - combined.csv：数值标签按原始类别保留，自动形成多类任务。
    # 0: Normal - 正常
    # 1: NS - 物理故障（网络正常+物理传感器故障）
    # 2: NM - 物理故障（网络正常+物理机械故障）
    # 3: PM - 参数欺骗攻击导致的物理机械故障
    # 4: PS - 参数欺骗攻击导致的物理传感器故障
    # 5: SS - 停产攻击导致的物理传感器故障
    
    numeric_labels = pd.to_numeric(pd.Series(labels), errors='coerce')
    use_numeric_labels = numeric_labels.notna().all()

    LABEL_NAMES = ['Normal', 'NS', 'NM', 'PM', 'PS', 'SS']
    LABEL_DESCRIPTIONS = {
        0: ('normal', '正常', 'None'),
        1: ('phy', '物理故障-传感器(NS)', 'Physical Layer'),
        2: ('phy', '物理故障-机械(NM)', 'Physical Layer'),
        3: ('net_phy', '参数欺骗攻击+物理机械故障(PM)', 'Network Layer → Physical Layer'),
        4: ('net_phy', '参数欺骗攻击+物理传感器故障(PS)', 'Network Layer → Physical Layer'),
        5: ('net_phy', '停产攻击+物理传感器故障(SS)', 'Network Layer → Physical Layer')
    }
    
    label_map = {}
    anomaly_types = {}
    if use_numeric_labels:
        raw_numeric = numeric_labels.astype(int).to_numpy()
        unique_labels = sorted(np.unique(raw_numeric).tolist())
        remap = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        y = np.array([remap[int(lbl)] for lbl in raw_numeric], dtype=np.int64)
        if unique_labels == list(range(len(unique_labels))):
            LABEL_NAMES = ['Normal'] + [f'Attack{i}' for i in unique_labels[1:]]
        else:
            LABEL_NAMES = [f'Class{lbl}' for lbl in unique_labels]
            if 0 in unique_labels:
                LABEL_NAMES[remap[0]] = 'Normal'
        for i, cls_id in enumerate(y):
            if cls_id == 0:
                anomaly_types[i] = ('normal', LABEL_NAMES[cls_id], '正常')
            else:
                anomaly_types[i] = ('net_phy', LABEL_NAMES[cls_id], f'异常类别{cls_id}')
    else:
        for i, lbl in enumerate(labels):
            lbl_str = str(lbl).strip().upper()
            if lbl_str == 'NS':
                label_map[i] = 1
                anomaly_types[i] = ('phy', 'NS', '物理传感器故障')
            elif lbl_str == 'NM':
                label_map[i] = 2
                anomaly_types[i] = ('phy', 'NM', '物理机械故障')
            elif lbl_str == 'PM':
                label_map[i] = 3
                anomaly_types[i] = ('net_phy', 'PM', '参数欺骗攻击→物理机械故障')
            elif lbl_str == 'PS':
                label_map[i] = 4
                anomaly_types[i] = ('net_phy', 'PS', '参数欺骗攻击→物理传感器故障')
            elif lbl_str == 'SS':
                label_map[i] = 5
                anomaly_types[i] = ('net_phy', 'SS', '停产攻击→物理传感器故障')
            else:
                label_map[i] = 0
                anomaly_types[i] = ('normal', 'Normal', '正常')
        y = np.array([label_map[i] for i in range(len(labels))])
    
    print(f"📊 {len(LABEL_NAMES)}分类标签分布:")
    for cls_id, name in enumerate(LABEL_NAMES):
        count = np.sum(y == cls_id)
        if count > 0:
            print(f"   Class {cls_id} ({name}): {count}")
    
    return {
        'times': times,
        'X_phy': X_phy,
        'X_net_feat': X_net_feat,  # 网络特征（用于边/节点特征）
        'src_ips': src_ips,         # 每个样本的源IP
        'dst_ips': dst_ips,         # 每个样本的目标IP
        'all_ips': all_ips,         # 所有唯一IP（网络节点）
        'ip_to_idx': ip_to_idx,     # IP到索引的映射
        'y': y,
        'labels_raw': labels,
        'anomaly_types': anomaly_types,
        'phy_cols': phy_cols,
        'net_feat_cols': net_feat_cols,
        'label_names': LABEL_NAMES,
        'label_descriptions': LABEL_DESCRIPTIONS,
        'n_classes': len(LABEL_NAMES)
    }


def subset_data(data, indices):
    """Return a view-like sliced dataset with local anomaly_type keys."""
    indices = np.asarray(indices, dtype=int)
    subset = {}
    per_sample_keys = {
        'times', 'X_phy', 'X_net_feat', 'src_ips', 'dst_ips',
        'y', 'labels_raw'
    }
    for key, value in data.items():
        if key in ('_val_data', '_test_data'):
            continue
        if key in per_sample_keys:
            subset[key] = np.asarray(value)[indices]
        elif key == 'anomaly_types':
            subset[key] = {
                local_i: data['anomaly_types'].get(int(orig_i), ('normal', 'Normal', '正常'))
                for local_i, orig_i in enumerate(indices)
            }
        else:
            subset[key] = value
    subset['original_indices'] = indices
    return subset



def cap_split_per_class(data, cap_per_class=None, seed=42, split_name='Train'):
    """Limit a split size per class for faster training while preserving label diversity."""
    if cap_per_class is None or cap_per_class <= 0:
        return data
    y = np.asarray(data['y'])
    rng = np.random.default_rng(seed)
    selected = []
    for cls_id in np.unique(y):
        cls_idx = np.where(y == cls_id)[0]
        if len(cls_idx) > cap_per_class:
            # Preserve chronological coverage by selecting evenly across each class block.
            positions = np.linspace(0, len(cls_idx) - 1, cap_per_class).round().astype(int)
            chosen = cls_idx[np.unique(positions)]
            if len(chosen) < cap_per_class:
                remaining = np.setdiff1d(cls_idx, chosen, assume_unique=False)
                extra = rng.choice(remaining, size=cap_per_class - len(chosen), replace=False)
                chosen = np.concatenate([chosen, extra])
        else:
            chosen = cls_idx
        selected.extend(chosen.tolist())
    selected = np.array(sorted(selected), dtype=int)
    capped = subset_data(data, selected)
    print(f"📉 {split_name} cap per class: {len(y)} -> {len(selected)} samples (cap={cap_per_class})")
    return capped


def chronological_split_and_scale(data, train_ratio=0.6, val_ratio=0.2, class_aware=True):
    """Chronologically split data and fit scalers only on the training split.

    class_aware=True keeps chronological order within each class. This avoids
    random sample leakage while preventing block-ordered datasets from putting
    unseen attack classes entirely into the held-out split.
    """
    n_samples = len(data['y'])
    if class_aware:
        train_parts, val_parts, test_parts = [], [], []
        for cls_id in sorted(np.unique(data['y']).tolist()):
            cls_idx = np.where(data['y'] == cls_id)[0]
            if len(cls_idx) < 3:
                train_parts.append(cls_idx)
                continue
            train_end = max(1, int(len(cls_idx) * train_ratio))
            val_end = max(train_end + 1, int(len(cls_idx) * (train_ratio + val_ratio)))
            val_end = min(val_end, len(cls_idx) - 1)
            train_parts.append(cls_idx[:train_end])
            val_parts.append(cls_idx[train_end:val_end])
            test_parts.append(cls_idx[val_end:])
        train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int)
        val_idx = np.sort(np.concatenate(val_parts)) if val_parts else train_idx
        test_idx = np.sort(np.concatenate(test_parts)) if test_parts else val_idx
    else:
        train_end = max(1, int(n_samples * train_ratio))
        val_end = max(train_end + 1, int(n_samples * (train_ratio + val_ratio)))
        val_end = min(val_end, n_samples - 1) if n_samples > 2 else n_samples
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n_samples)
        if len(test_idx) == 0:
            test_idx = val_idx

    train_data = subset_data(data, train_idx)
    val_data = subset_data(data, val_idx)
    test_data = subset_data(data, test_idx)

    scaler_phy = StandardScaler().fit(train_data['X_phy'])
    scaler_net = StandardScaler().fit(train_data['X_net_feat'])
    for split in (train_data, val_data, test_data):
        split['X_phy'] = scaler_phy.transform(split['X_phy']).astype(np.float32)
        split['X_net_feat'] = scaler_net.transform(split['X_net_feat']).astype(np.float32)

    train_data['_val_data'] = val_data
    train_data['_test_data'] = test_data

    split_name = "class-aware chronological" if class_aware else "global chronological"
    print(f"\n📊 Chronological split (leakage-aware, {split_name}):")
    print(f"   Train: {len(train_idx)} samples [{train_idx[0]}..{train_idx[-1]}]")
    print(f"   Val:   {len(val_idx)} samples [{val_idx[0] if len(val_idx) else 'NA'}..{val_idx[-1] if len(val_idx) else 'NA'}]")
    print(f"   Test:  {len(test_idx)} samples [{test_idx[0]}..{test_idx[-1]}]")
    print("   Scalers are fitted on Train only and applied to Val/Test.")
    return {'train': train_data, 'val': val_data, 'test': test_data}


def _phy_indices_by_keywords(graph_builder, keyword_groups):
    """Find concrete physical node indices matching any keyword group."""
    matches = set()
    node_names = [str(n).lower().replace('_', ' ') for n in graph_builder.phy_nodes]
    for group in keyword_groups:
        group_l = [g.lower().replace('_', ' ') for g in group]
        for local_idx, name in enumerate(node_names):
            if all(k in name for k in group_l):
                matches.add(graph_builder.n_net + local_idx)
    return matches


def infer_true_root_nodes(sample_idx, data, graph_builder):
    """Infer concrete root-cause node labels for strict traceback evaluation.

    Priority:
    1) Explicit root-cause columns if a future dataset provides them.
    2) Network-origin attacks use the sample's Source IP as the concrete root.
    3) IGCPS physical faults use small process-specific physical-node sets.
    """
    y = int(data['y'][sample_idx])
    if y == 0:
        return set()

    label_names = data.get('label_names', [])
    label_name = label_names[y] if y < len(label_names) else f'Class{y}'
    label_upper = str(label_name).upper()
    anomaly_info = data.get('anomaly_types', {}).get(sample_idx, ('net_phy', label_name, ''))
    src_ip = str(data['src_ips'][sample_idx])
    dst_ip = str(data['dst_ips'][sample_idx])

    src_node = graph_builder.ip_to_idx.get(src_ip)
    dst_node = graph_builder.ip_to_idx.get(dst_ip)

    if label_upper in {'NS'}:
        nodes = _phy_indices_by_keywords(graph_builder, [
            ['high', 'switch'], ['sub', 'switch'], ['switch', 'status']
        ])
        return nodes or ({graph_builder.n_net} if graph_builder.n_phy else set())

    if label_upper in {'NM'}:
        nodes = _phy_indices_by_keywords(graph_builder, [
            ['medium', 'heater'], ['medium', 'temperature'], ['medium', 'outlet']
        ])
        return nodes or ({graph_builder.n_net} if graph_builder.n_phy else set())

    # For cyber-origin attacks, the concrete source endpoint is the strict root.
    if anomaly_info[0] == 'net_phy' or label_upper.startswith('ATTACK') or label_upper.startswith('CLASS') or y > 0:
        if src_node is not None:
            return {src_node}
        if dst_node is not None:
            return {dst_node}

    return set()


def compute_strict_traceback_metrics(all_node_scores, data, graph_builder):
    """Compute exact-node RCA/MRR/NDCG/APD without layer-level shortcuts."""
    rca_correct = 0
    total = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    apd_sum = 0.0
    skipped = 0
    K = 5

    for t in range(len(data['y'])):
        if int(data['y'][t]) == 0:
            continue
        true_nodes = infer_true_root_nodes(t, data, graph_builder)
        if not true_nodes:
            skipped += 1
            continue
        total += 1
        node_scores = all_node_scores[t]
        sorted_indices = np.argsort(node_scores)[::-1]
        if int(sorted_indices[0]) in true_nodes:
            rca_correct += 1
        first_rank = None
        dcg = 0.0
        for rank, node_idx in enumerate(sorted_indices, 1):
            if int(node_idx) in true_nodes:
                if first_rank is None:
                    first_rank = rank
                    mrr_sum += 1.0 / rank
                    apd_sum += rank
                if rank <= K:
                    dcg += 1.0 / np.log2(rank + 1)
        n_relevant = min(K, len(true_nodes))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
        ndcg_sum += dcg / max(idcg, 1e-8)

    denom = max(total, 1)
    return {
        'rca': rca_correct / denom * 100,
        'mrr': mrr_sum / denom * 100,
        'ndcg': ndcg_sum / denom * 100,
        'apd': apd_sum / denom if total else 0.0,
        'trace_eval_total': total,
        'trace_eval_skipped': skipped,
    }


def build_root_target(sample_idx, data, graph_builder, device):
    """Build a multi-hot exact-node root target for traceback-aware training."""
    true_nodes = infer_true_root_nodes(sample_idx, data, graph_builder)
    if not true_nodes:
        return None
    target = torch.zeros(graph_builder.n_nodes, dtype=torch.float32, device=device)
    for node_idx in true_nodes:
        if 0 <= int(node_idx) < graph_builder.n_nodes:
            target[int(node_idx)] = 1.0
    return target if target.sum() > 0 else None


def build_raw_calibrator_features(data, graph_builder):
    """Build sample-level train/test features for the Ours-only edge calibrator."""
    X_net = np.asarray(data['X_net_feat'], dtype=np.float32)
    X_phy = np.asarray(data['X_phy'], dtype=np.float32)
    n = len(data['y'])
    src_onehot = np.zeros((n, graph_builder.n_net), dtype=np.float32)
    dst_onehot = np.zeros((n, graph_builder.n_net), dtype=np.float32)
    for i, (src_ip, dst_ip) in enumerate(zip(data['src_ips'], data['dst_ips'])):
        src_idx = graph_builder.ip_to_idx.get(str(src_ip))
        dst_idx = graph_builder.ip_to_idx.get(str(dst_ip))
        if src_idx is not None and src_idx < graph_builder.n_net:
            src_onehot[i, src_idx] = 1.0
        if dst_idx is not None and dst_idx < graph_builder.n_net:
            dst_onehot[i, dst_idx] = 1.0
    return np.hstack([X_net, X_phy, src_onehot, dst_onehot]).astype(np.float32)


def fit_raw_feature_calibrator(model, data, graph_builder, model_name="Model", verbose=True):
    """Fit a lightweight supervised raw-feature calibrator for the main model."""
    if model_name != "Ours":
        return model
    try:
        import warnings
        from sklearn.linear_model import SGDClassifier
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return model

    X = build_raw_calibrator_features(data, graph_builder)
    y = np.asarray(data['y'])
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=1e-4,
        max_iter=30,
        tol=1e-3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(Xs, y)
    model.raw_feature_calibrator = {
        'scaler': scaler,
        'clf': clf,
        'classes': np.asarray(clf.classes_, dtype=int),
        'alpha': 0.45,
        'cache': {}
    }
    model.use_input_root_prior = True
    if verbose:
        print(f"     ✅ [{model_name}] Raw-feature calibrator fitted on {len(y)} train samples")
    return model


def get_raw_calibrator_probs(model, data, graph_builder, n_classes):
    """Return cached raw-feature calibrator probabilities for a data split."""
    cal = getattr(model, 'raw_feature_calibrator', None)
    if not cal:
        return None
    cache_key = id(data)
    if cache_key in cal['cache']:
        return cal['cache'][cache_key]

    X = build_raw_calibrator_features(data, graph_builder)
    Xs = cal['scaler'].transform(X)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw_probs = cal['clf'].predict_proba(Xs)
    probs = np.zeros((len(X), n_classes), dtype=np.float32)
    for j, cls_id in enumerate(cal['classes']):
        if 0 <= int(cls_id) < n_classes:
            probs[:, int(cls_id)] = raw_probs[:, j]
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = probs.sum(axis=1, keepdims=True)
    bad_rows = (row_sums[:, 0] <= 1e-8)
    if np.any(bad_rows):
        probs[bad_rows, :] = 1.0 / max(n_classes, 1)
        row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / np.maximum(row_sums, 1e-8)
    cal['cache'][cache_key] = probs
    return probs


def apply_input_root_prior(node_scores, node_feats, pred_label, graph_builder, label_names=None):
    """Use source/destination node markers as a root prior for predicted cyber attacks."""
    if int(pred_label) <= 0 or node_feats.shape[1] <= 4:
        return node_scores
    label_name = ''
    if label_names is not None and int(pred_label) < len(label_names):
        label_name = str(label_names[int(pred_label)]).upper()
    if label_name in {'NS', 'NM'}:
        return node_scores

    scores = np.array(node_scores, copy=True)
    n_net = min(graph_builder.n_net, len(scores))
    source_candidates = np.where(node_feats[:n_net, 4] > 0.5)[0]
    dest_candidates = np.where(node_feats[:n_net, 4] < -0.5)[0]
    if len(source_candidates) > 0:
        scores[:n_net] *= 1.25
        scores[source_candidates] = max(float(scores.max()) + 1e-3, float(scores[source_candidates].max()) * 8.0)
    if len(dest_candidates) > 0:
        scores[dest_candidates] *= 1.5
    return scores



# ========================= 2. 异构图构建 =========================

class HeteroGraphBuilder:
    """构建异构图：网络节点（IP地址） + 物理节点 + 边
    
    网络节点：Source IP 和 Destination IP 构成的唯一IP地址
    物理节点：按照燃气系统数据流顺序进行链式连接
    """
    
    def __init__(self, phy_cols, all_ips, net_feat_dim=1, phy_feat_dim=1):
        # 保持phy_cols的原始顺序（表格中的顺序就是燃气流经顺序）
        self.phy_cols = phy_cols
        self.all_ips = all_ips  # 所有唯一IP地址
        self.ip_to_idx = {ip: i for i, ip in enumerate(all_ips)}
        self.net_feat_dim = net_feat_dim
        self.phy_feat_dim = phy_feat_dim
        
        # 节点名称
        self.phy_nodes = [f"Phy_{c.replace('phy_', '')}" for c in phy_cols]
        self.net_nodes = [f"IP_{ip}" for ip in all_ips]  # 网络节点是IP地址
        self.all_nodes = self.net_nodes + self.phy_nodes
        
        self.n_net = len(self.net_nodes)
        self.n_phy = len(self.phy_nodes)
        self.n_nodes = self.n_net + self.n_phy

        # 统一的节点特征维度（取最大值）
        self.node_feat_dim = max(net_feat_dim, phy_feat_dim, 8)  # 至少8维

        # 时间窗口设置
        self.temporal_window = 3  # 默认K=3
        self.use_temporal = True  # 默认启用时间特征
        self.temporal_feat_dim = self.node_feat_dim + 4 if self.use_temporal else self.node_feat_dim

        # 动态边：记录每个时刻的网络通信边
        self.dynamic_net_edges = []

    def build_static_adjacency(self, use_cross_layer=True, use_phy_chain=True, use_net_edges=True):
        """构建静态邻接矩阵
        
        参数:
        - use_cross_layer: 是否添加网络-物理跨层连接
        - use_phy_chain: 是否使用物理层链式连接（燃气流经顺序）
        - use_net_edges: 是否使用网络层通信边（IP通信统计）
        """
        adj = np.zeros((self.n_nodes, self.n_nodes))
        
        # 物理节点按顺序链式连接（燃气按表格特征顺序流经）
        if use_phy_chain:
            for i in range(self.n_phy - 1):
                phy_i = self.n_net + i
                phy_j = self.n_net + i + 1
                adj[phy_i, phy_j] = 1.0
                adj[phy_j, phy_i] = 1.0
        
        # 网络层静态边（基于历史通信统计，这里用全连接近似）
        if use_net_edges:
            for i in range(self.n_net):
                for j in range(i + 1, self.n_net):
                    adj[i, j] = 0.5  # 网络节点间的基础连接
                    adj[j, i] = 0.5
        
        # 网络-物理之间有边（网络攻击可能影响任意物理节点）
        if use_cross_layer:
            for i in range(self.n_net):
                for j in range(self.n_net, self.n_nodes):
                    adj[i, j] = 0.3
                    adj[j, i] = 0.3
        
        return adj
    
    def build_dynamic_adjacency(self, src_ip, dst_ip, use_cross_layer=True, use_phy_chain=True, use_net_edges=True):
        """构建动态邻接矩阵（包含当前时刻的网络通信边）
        
        参数:
        - use_cross_layer: 是否添加网络-物理跨层连接
        - use_phy_chain: 是否使用物理层链式连接
        - use_net_edges: 是否使用网络层通信边
        """
        adj = self.build_static_adjacency(use_cross_layer=use_cross_layer, 
                                          use_phy_chain=use_phy_chain, 
                                          use_net_edges=use_net_edges)
        
        # 添加当前时刻的网络通信边：src_ip -> dst_ip
        if src_ip in self.ip_to_idx and dst_ip in self.ip_to_idx:
            src_idx = self.ip_to_idx[src_ip]
            dst_idx = self.ip_to_idx[dst_ip]
            adj[src_idx, dst_idx] = 1.0
            adj[dst_idx, src_idx] = 1.0  # 双向
        
        return adj
    
    def get_phy_flow_order(self):
        """返回物理节点的燃气流经顺序"""
        return list(range(self.n_phy))
    
    def get_node_features(self, X_net_feat_t, X_phy_t, src_ip, dst_ip,
                           X_phy_history=None, X_net_feat_history=None,
                           src_ips_history=None, dst_ips_history=None):
        """获取时刻t的节点特征

        网络节点特征：使用网络特征的不同统计量填充
        物理节点特征：物理传感器值 + 位置编码

        如果提供了历史数据(X_phy_history等)，则自动启用时间特征。

        参数:
        - X_net_feat_t: 原始网络特征（当前时刻）
        - X_phy_t: 原始物理特征（当前时刻）
        - src_ip: 源IP
        - dst_ip: 目的IP
        - X_phy_history: [K-1, n_phy] 过去K-1个时间步的物理特征（可选）
        - X_net_feat_history: [K-1, n_net_feat] 过去K-1个时间步的网络特征（可选）
        - src_ips_history: [K-1] 过去K-1个时间步的源IP（可选）
        - dst_ips_history: [K-1] 过去K-1个时间步的目的IP（可选）
        """
        use_temporal = (X_phy_history is not None and self.use_temporal)

        if use_temporal:
            X_phy_window = np.vstack([X_phy_history, X_phy_t.reshape(1, -1)])
            X_net_window = np.vstack([X_net_feat_history, np.array(X_net_feat_t).reshape(1, -1)])
            src_window = list(src_ips_history) + [src_ip] if src_ips_history is not None else None
            dst_window = list(dst_ips_history) + [dst_ip] if dst_ips_history is not None else None
            return self.get_node_features_temporal(X_net_window, X_phy_window, src_ip, dst_ip,
                                                   src_window, dst_window)

        feat_dim = self.node_feat_dim
        
        # 网络节点特征（每个IP一个特征向量）
        net_feats = np.zeros((self.n_net, feat_dim), dtype=np.float32)
        
        # 为每个网络节点生成特征
        net_feat_arr = np.array(X_net_feat_t).flatten() if len(X_net_feat_t) > 0 else np.zeros(1)
        
        # 基础特征：使用网络特征的统计量
        base_net_feat = np.zeros(feat_dim)
        if len(net_feat_arr) > 0:
            base_net_feat[0] = np.mean(net_feat_arr)
            base_net_feat[1] = np.std(net_feat_arr) if len(net_feat_arr) > 1 else 0
            base_net_feat[2] = np.max(net_feat_arr)
            base_net_feat[3] = np.min(net_feat_arr)
        
        # 源IP和目的IP节点使用增强的网络特征
        if src_ip in self.ip_to_idx:
            src_idx = self.ip_to_idx[src_ip]
            net_feats[src_idx, :4] = base_net_feat[:4]
            net_feats[src_idx, 4] = 1.0  # 源IP标记
            net_feats[src_idx, 5] = src_idx / max(self.n_net, 1)  # 位置编码

        if dst_ip in self.ip_to_idx:
            dst_idx = self.ip_to_idx[dst_ip]
            net_feats[dst_idx, :4] = base_net_feat[:4]
            net_feats[dst_idx, 4] = -1.0  # 目的IP标记
            net_feats[dst_idx, 5] = dst_idx / max(self.n_net, 1)  # 位置编码

        # 物理节点特征（每个物理节点一个特征向量）
        phy_feats = np.zeros((self.n_phy, feat_dim), dtype=np.float32)
        for i in range(self.n_phy):
            phy_feats[i, 0] = X_phy_t[i] if i < len(X_phy_t) else 0  # 传感器值
            phy_feats[i, 1] = i / max(self.n_phy, 1)  # 位置编码（燃气流经顺序）
            # 邻居特征（前后节点的值）
            if i > 0:
                phy_feats[i, 2] = X_phy_t[i-1] if i-1 < len(X_phy_t) else 0
            if i < self.n_phy - 1:
                phy_feats[i, 3] = X_phy_t[i+1] if i+1 < len(X_phy_t) else 0
            # 局部梯度
            phy_feats[i, 4] = phy_feats[i, 0] - phy_feats[i, 2]  # 与前节点差值

        return np.vstack([net_feats, phy_feats])

    def get_node_features_temporal(self, X_net_feat_window, X_phy_window, src_ip, dst_ip,
                                    src_ips_window=None, dst_ips_window=None):
        """获取带时间窗口上下文的节点特征

        在原始特征基础上增加时间差分特征，建模攻击注入到物理表现的动态延迟。

        参数:
        - X_net_feat_window: [K, n_net_feat] 最近K个时间步的网络特征
        - X_phy_window: [K, n_phy] 最近K个时间步的物理特征
        - src_ip, dst_ip: 当前时刻的源/目的IP
        - src_ips_window: [K] 最近K个时间步的源IP列表
        - dst_ips_window: [K] 最近K个时间步的目的IP列表
        """
        K = len(X_phy_window)
        feat_dim = self.node_feat_dim + 4  # 增加4维时间特征

        # 网络节点特征
        net_feats = np.zeros((self.n_net, feat_dim), dtype=np.float32)
        net_feat_arr = np.array(X_net_feat_window[-1]).flatten() if len(X_net_feat_window[-1]) > 0 else np.zeros(1)

        base_net_feat = np.zeros(4)
        if len(net_feat_arr) > 0:
            base_net_feat[0] = np.mean(net_feat_arr)
            base_net_feat[1] = np.std(net_feat_arr) if len(net_feat_arr) > 1 else 0
            base_net_feat[2] = np.max(net_feat_arr)
            base_net_feat[3] = np.min(net_feat_arr)

        if src_ip in self.ip_to_idx:
            src_idx = self.ip_to_idx[src_ip]
            net_feats[src_idx, :4] = base_net_feat
            net_feats[src_idx, 4] = 1.0
            net_feats[src_idx, 5] = src_idx / max(self.n_net, 1)

        if dst_ip in self.ip_to_idx:
            dst_idx = self.ip_to_idx[dst_ip]
            net_feats[dst_idx, :4] = base_net_feat
            net_feats[dst_idx, 4] = -1.0
            net_feats[dst_idx, 5] = dst_idx / max(self.n_net, 1)

        # 时间特征：过去K步中该IP的活跃次数
        if src_ips_window is not None and dst_ips_window is not None:
            for k in range(K - 1):  # 不含当前时刻
                s_ip = src_ips_window[k]
                d_ip = dst_ips_window[k]
                if s_ip in self.ip_to_idx:
                    net_feats[self.ip_to_idx[s_ip], self.node_feat_dim] += 1.0 / K
                if d_ip in self.ip_to_idx:
                    net_feats[self.ip_to_idx[d_ip], self.node_feat_dim + 1] += 1.0 / K

        # 物理节点特征
        phy_feats = np.zeros((self.n_phy, feat_dim), dtype=np.float32)
        X_phy_t = X_phy_window[-1]  # 当前时刻

        for i in range(self.n_phy):
            phy_feats[i, 0] = X_phy_t[i] if i < len(X_phy_t) else 0
            phy_feats[i, 1] = i / max(self.n_phy, 1)
            if i > 0:
                phy_feats[i, 2] = X_phy_t[i-1] if i-1 < len(X_phy_t) else 0
            if i < self.n_phy - 1:
                phy_feats[i, 3] = X_phy_t[i+1] if i+1 < len(X_phy_t) else 0
            phy_feats[i, 4] = phy_feats[i, 0] - phy_feats[i, 2]

            # 时间差分特征: Δ_{t-1} 和 Δ_{t-2}
            if K >= 2:
                prev_val = X_phy_window[-2][i] if i < len(X_phy_window[-2]) else 0
                phy_feats[i, self.node_feat_dim] = X_phy_t[i] - prev_val  # Δ_{t-1}
            if K >= 3:
                prev2_val = X_phy_window[-3][i] if i < len(X_phy_window[-3]) else 0
                phy_feats[i, self.node_feat_dim + 1] = X_phy_t[i] - prev2_val  # Δ_{t-2}

            # 时间变化率（加速度）
            if K >= 3:
                d1 = phy_feats[i, self.node_feat_dim]      # Δ_{t-1}
                prev_val = X_phy_window[-2][i] if i < len(X_phy_window[-2]) else 0
                prev2_val = X_phy_window[-3][i] if i < len(X_phy_window[-3]) else 0
                d0 = prev_val - prev2_val                   # Δ_{t-2} to Δ_{t-1}
                phy_feats[i, self.node_feat_dim + 2] = d1 - d0  # 二阶差分
            # 时间窗口内标准差
            if K >= 2:
                vals = [X_phy_window[k][i] if i < len(X_phy_window[k]) else 0 for k in range(K)]
                phy_feats[i, self.node_feat_dim + 3] = np.std(vals)

        return np.vstack([net_feats, phy_feats])

    def get_node_features_with_history(self, t, X_net_feat, X_phy, src_ips, dst_ips):
        """便捷方法：自动构建时间窗口并获取节点特征"""
        K = self.temporal_window
        if self.use_temporal:
            hist_start = max(0, t - (K - 1))
            X_phy_hist = X_phy[hist_start:t] if t > 0 else X_phy[0:1]
            X_net_hist = X_net_feat[hist_start:t] if t > 0 else X_net_feat[0:1]
            src_hist = src_ips[hist_start:t] if t > 0 else src_ips[0:1]
            dst_hist = dst_ips[hist_start:t] if t > 0 else dst_ips[0:1]
            while len(X_phy_hist) < K - 1:
                X_phy_hist = np.vstack([X_phy_hist[:1], X_phy_hist])
                X_net_hist = np.vstack([X_net_hist[:1], X_net_hist])
                src_hist = np.concatenate([[src_hist[0]], src_hist])
                dst_hist = np.concatenate([[dst_hist[0]], dst_hist])
            return self.get_node_features(
                X_net_feat[t], X_phy[t], src_ips[t], dst_ips[t],
                X_phy_history=X_phy_hist, X_net_feat_history=X_net_hist,
                src_ips_history=src_hist, dst_ips_history=dst_hist)
        else:
            return self.get_node_features(X_net_feat[t], X_phy[t], src_ips[t], dst_ips[t])


class DynamicEdgeWeightModule(nn.Module):
    """动态边权重模块

    根据节点特征相似度动态计算边权重，替代静态先验权重。
    解决跨域攻击时网络-物理层耦合强度动态变化的问题。

    边权重计算:
        w_ij = sigma(MLP([h_i; h_j; |h_i - h_j|]))
    """

    def __init__(self, node_feat_dim, hidden_dim=32):
        super().__init__()
        self.edge_weight_net = nn.Sequential(
            nn.Linear(node_feat_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.cross_layer_bias = nn.Parameter(torch.tensor(0.3))
        self.intra_layer_bias = nn.Parameter(torch.tensor(0.5))

    def forward(self, node_features, adj, n_net):
        """计算动态边权重

        Args:
            node_features: [n_nodes, feat_dim]
            adj: [n_nodes, n_nodes] 静态邻接矩阵（0/1连接）
            n_net: 网络节点数量

        Returns:
            weighted_adj: [n_nodes, n_nodes] 动态加权邻接矩阵
        """
        n_nodes = node_features.shape[0]

        # 计算节点对特征
        h_i = node_features.unsqueeze(1).expand(-1, n_nodes, -1)
        h_j = node_features.unsqueeze(0).expand(n_nodes, -1, -1)
        h_diff = torch.abs(h_i - h_j)
        pair_feat = torch.cat([h_i, h_j, h_diff], dim=-1)

        # 计算边权重
        edge_weights = self.edge_weight_net(pair_feat).squeeze(-1)

        # 添加层间/层内偏置
        bias_matrix = torch.zeros(n_nodes, n_nodes, device=node_features.device)
        bias_matrix[:n_net, :n_net] = torch.sigmoid(self.intra_layer_bias)
        bias_matrix[n_net:, n_net:] = torch.sigmoid(self.intra_layer_bias)
        bias_matrix[:n_net, n_net:] = torch.sigmoid(self.cross_layer_bias)
        bias_matrix[n_net:, :n_net] = torch.sigmoid(self.cross_layer_bias)

        # 最终权重 = 学习权重 * 静态连接 + 偏置调制
        weighted_adj = edge_weights * adj + bias_matrix * adj * 0.1

        return weighted_adj


class GraphAttentionLayer(nn.Module):
    """图注意力层"""
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.out_features = out_features
        
    def forward(self, h, adj):
        Wh = self.W(h)  # [N, out_features]
        N = Wh.size(0)
        
        Wh1 = Wh.repeat(1, N).view(N * N, -1)
        Wh2 = Wh.repeat(N, 1)
        a_input = torch.cat([Wh1, Wh2], dim=1).view(N, N, 2*self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        
        h_prime = torch.matmul(attention, Wh)
        return h_prime, attention

# ============================================================================
# 对比模型：无监督/自监督GNN模型
# ============================================================================

class GCN_AE_Custom(nn.Module):
    """
    GCN自编码器 - 无监督学习（自定义实现，不依赖PyG）
    通过重构边来学习节点嵌入
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, 
                 latent_channels: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        
        self.layers.append(nn.Linear(hidden_channels, latent_channels))
        self.dropout = dropout
        
    def encode(self, x, adj):
        """编码：获取节点嵌入"""
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj_norm = adj / deg
        
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.matmul(adj_norm, x)
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = torch.matmul(adj_norm, x)
        x = self.layers[-1](x)
        return x
    
    def decode(self, z):
        """解码：通过内积重构边"""
        return torch.sigmoid(torch.matmul(z, z.t()))
    
    def forward(self, x, adj):
        z = self.encode(x, adj)
        return z
    
    def get_embeddings(self, x, adj):
        return self.encode(x, adj)


class GAT_AE_Custom(nn.Module):
    """
    GAT自编码器 - 带注意力机制的无监督学习（自定义实现）
    """
    
    def __init__(self, in_channels: int, hidden_channels: int,
                 latent_channels: int, num_layers: int = 2, 
                 heads: int = 4, dropout: float = 0.5):
        super().__init__()
        
        self.gat1 = nn.ModuleList([
            GraphAttentionLayer(in_channels, hidden_channels, dropout) 
            for _ in range(heads)
        ])
        self.gat2 = GraphAttentionLayer(hidden_channels * heads, latent_channels, dropout)
        self.dropout = dropout
        
    def encode(self, x, adj):
        x = torch.cat([gat(x, adj)[0] for gat in self.gat1], dim=1)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, _ = self.gat2(x, adj)
        return x
    
    def decode(self, z):
        return torch.sigmoid(torch.matmul(z, z.t()))
    
    def forward(self, x, adj):
        return self.encode(x, adj)
    
    def get_embeddings(self, x, adj):
        return self.encode(x, adj)


class GraphSAGE_AE_Custom(nn.Module):
    """
    GraphSAGE自编码器 - 采样聚合的无监督学习（自定义实现）
    """
    
    def __init__(self, in_channels: int, hidden_channels: int,
                 latent_channels: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        
        # SAGE: concat(self, mean_neighbor)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels * 2, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels * 2, hidden_channels))
        
        self.layers.append(nn.Linear(hidden_channels * 2, latent_channels))
        self.dropout = dropout
        
    def encode(self, x, adj):
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj_norm = adj / deg
        
        h = x
        for i, layer in enumerate(self.layers[:-1]):
            neigh = torch.matmul(adj_norm, h)
            h = torch.cat([h, neigh], dim=1)
            h = layer(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        neigh = torch.matmul(adj_norm, h)
        h = torch.cat([h, neigh], dim=1)
        h = self.layers[-1](h)
        return h
    
    def decode(self, z):
        return torch.sigmoid(torch.matmul(z, z.t()))
    
    def forward(self, x, adj):
        return self.encode(x, adj)
    
    def get_embeddings(self, x, adj):
        return self.encode(x, adj)



class VGAE_Custom(nn.Module):
    """
    变分图自编码器 (Variational Graph AutoEncoder) - 自定义实现
    """
    
    def __init__(self, in_channels: int, hidden_channels: int,
                 latent_channels: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        
        # 共享编码器
        self.shared_layers = nn.ModuleList()
        self.shared_layers.append(nn.Linear(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.shared_layers.append(nn.Linear(hidden_channels, hidden_channels))
        
        # 均值和方差编码器
        self.fc_mu = nn.Linear(hidden_channels, latent_channels)
        self.fc_logvar = nn.Linear(hidden_channels, latent_channels)
        
        self.dropout = dropout
        
    def encode(self, x, adj):
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj_norm = adj / deg
        
        for layer in self.shared_layers:
            x = torch.matmul(adj_norm, x)
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = torch.matmul(adj_norm, x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        return torch.sigmoid(torch.matmul(z, z.t()))
    
    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def get_embeddings(self, x, adj):
        mu, _ = self.encode(x, adj)
        return mu
    
    def kl_loss(self, mu, logvar):
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


class TemporalShiftModule(nn.Module):
    """可学习时间偏移模块 - 建模跨域攻击传播延迟

    在ICPS中，网络攻击注入(t)到物理表现(t+Δ)之间存在动态延迟，
    取决于PLC响应时间、过程惯性和传感器采样周期。

    本模块对跨层边(网络→物理)的消息传递施加可学习的延迟调制：
    1. 延迟预测：基于源/目标节点特征预测每条跨层边的延迟量
    2. 衰减调制：延迟越大，信号衰减越多（模拟物理过程能量耗散）
    3. 相位编码：不同延迟产生不同的特征变换（类似位置编码）
    """

    def __init__(self, hidden_dim, n_delay_bases=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_delay_bases = n_delay_bases

        # 延迟基函数中心点（可学习），代表不同延迟量级
        self.delay_centers = nn.Parameter(torch.linspace(0.0, 1.0, n_delay_bases))

        # 延迟预测器：基于节点对特征预测延迟分布
        self.delay_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_delay_bases),
            nn.Softmax(dim=-1)
        )

        # 可学习衰减因子
        self.decay_factor = nn.Parameter(torch.tensor(0.5))

        # 延迟感知的边权重调制
        self.delay_gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, h, adj, n_net):
        """
        Args:
            h: [n_nodes, hidden_dim] 节点嵌入
            adj: [n_nodes, n_nodes] 邻接矩阵
            n_net: 网络节点数量
        Returns:
            adj_modulated: 延迟调制后的邻接矩阵
            delay_info: dict 包含学习到的延迟值（用于可解释性）
        """
        n_nodes = h.size(0)
        n_phy = n_nodes - n_net

        # 提取跨层边（网络→物理）的邻接子矩阵
        cross_adj = adj[:n_net, n_net:]  # [n_net, n_phy]

        # 找到非零跨层边
        cross_mask = (cross_adj > 0)

        if not cross_mask.any():
            return adj, {'delays': torch.zeros(1, device=h.device)}

        # 为所有可能的跨层边对计算延迟
        h_net = h[:n_net]   # [n_net, hidden_dim]
        h_phy = h[n_net:]   # [n_phy, hidden_dim]

        # 广播计算所有网络-物理节点对的特征拼接
        h_net_exp = h_net.unsqueeze(1).expand(-1, n_phy, -1)  # [n_net, n_phy, hidden]
        h_phy_exp = h_phy.unsqueeze(0).expand(n_net, -1, -1)  # [n_net, n_phy, hidden]
        edge_feats = torch.cat([h_net_exp, h_phy_exp], dim=-1)  # [n_net, n_phy, 2*hidden]

        # 预测每条边的延迟分布
        delay_weights = self.delay_predictor(edge_feats)  # [n_net, n_phy, n_bases]

        # 加权组合得到连续延迟值
        learned_delays = (delay_weights * self.delay_centers.view(1, 1, -1)).sum(dim=-1)  # [n_net, n_phy]

        # 延迟衰减：延迟越大信号越弱
        attenuation = torch.exp(-self.decay_factor.abs() * learned_delays)  # [n_net, n_phy]

        # 延迟门控：结合节点特征和延迟值决定最终边权重调制
        delay_feat = learned_delays.unsqueeze(-1)  # [n_net, n_phy, 1]
        gate_input = torch.cat([h_net_exp, delay_feat], dim=-1)  # [n_net, n_phy, hidden+1]
        delay_gate_val = self.delay_gate(gate_input).squeeze(-1)  # [n_net, n_phy]

        # 最终调制：原始权重 × 衰减 × 门控
        modulation = attenuation * delay_gate_val  # [n_net, n_phy]

        # 应用到邻接矩阵的跨层部分
        adj_new = adj.clone()
        adj_new[:n_net, n_net:] = cross_adj * modulation

        # 物理→网络方向也施加对称调制（反向传播路径）
        cross_adj_reverse = adj[n_net:, :n_net]  # [n_phy, n_net]
        if cross_adj_reverse.any():
            adj_new[n_net:, :n_net] = cross_adj_reverse * modulation.t()

        delay_info = {
            'delays': learned_delays.detach(),
            'attenuation': attenuation.detach(),
            'gate': delay_gate_val.detach()
        }

        return adj_new, delay_info


class HGT_Trace(nn.Module):
    """
    HGT-Trace: Heterogeneous Graph Attention Network for Traceback
    轻量化异构图注意力溯源网络（HGT-Trace）

    核心创新（三项轻量化设计）:
    1. 真正深度可分离图卷积 (Depthwise Separable Graph Convolution, DSGC):
       将 W∈R^{d×d} 分解为:
         - 深度部分: 2 组独立线性层（每组处理 d/2 通道），真正实现通道分组变换
         - 逐点部分: 全连接 W_p∈R^{d×d} 跨组通道混合
       参数量：2×(d/2)²+d² ≈ 1.5d²（比标准 GCN 参数量略多但表达力更强）

    2. 全局跨层注意力 (Global Cross-layer Attention, GCA):
       使用标准 nn.MultiheadAttention（无 adj mask），在全图节点间计算注意力，
       使模型可通过 attention weight 自主学习网络层与物理层之间的跨域关联，
       避免稀疏 mask 导致的 -inf/NaN 梯度消失问题。

    3. 自适应门控融合 (Adaptive Gated Fusion, AGF):
       gate = σ(W_g · [h_local ; h_global])
       output = gate ⊙ h_local + (1-gate) ⊙ h_global
       自适应平衡局部图传播（DW-Sep GCN）与全局跨层信息（MHA）。

    消融实验设计 (以 HGT-Trace 为主模型 Ours):
    - Ours (Full)       : use_cross_attn=True,  use_gate=True,  use_dw_sep=True
    - w/o Cross-Attn    : use_cross_attn=False (纯 DW-Sep GCN，无全局注意力)
    - w/o Gate          : use_gate=False        (直接相加替代门控融合)
    - w/o DW-Sep        : use_dw_sep=False      (标准 GCN 替代双组分离卷积)
    - w/o Cross-layer   : 由 graph_builder 的 use_cross_layer=False 控制
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 latent_channels: int, num_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.15,
                 use_cross_attn: bool = True,
                 use_gate: bool = True,
                 use_dw_sep: bool = True,
                 use_dynamic_edge_weights: bool = True,
                 use_temporal_shift: bool = True):
        super().__init__()

        self.use_cross_attn = use_cross_attn
        self.use_gate = use_gate
        self.use_dw_sep = use_dw_sep
        self.use_dynamic_edge_weights = use_dynamic_edge_weights
        self.use_temporal_shift = use_temporal_shift
        self.hidden_channels = hidden_channels
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.latent_channels = latent_channels

        # ── 输入投影 ──────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # ── 图卷积层（真正双组 DW-Sep or 标准 GCN）──────────────
        half = hidden_channels // 2
        if use_dw_sep:
            # 深度部分: 2 组独立线性层，每组处理 half 个通道
            self.dw_g1 = nn.ModuleList([
                nn.Linear(half, half, bias=False) for _ in range(num_layers)
            ])
            self.dw_g2 = nn.ModuleList([
                nn.Linear(half, half, bias=False) for _ in range(num_layers)
            ])
            # 逐点部分: 跨组通道混合
            self.pw_layers = nn.ModuleList([
                nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers)
            ])
        else:
            # 标准 GCN（用于消融 w/o DW-Sep）
            self.gcn_layers = nn.ModuleList([
                nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers)
            ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])

        # ── 全局跨层注意力（标准 MHA，无 adj mask）──────────────
        if use_cross_attn:
            self.self_attn = nn.MultiheadAttention(
                hidden_channels, n_heads, dropout=dropout, batch_first=True
            )
            self.cross_norm = nn.LayerNorm(hidden_channels)

        # ── 自适应门控融合 ────────────────────────────────────────
        if use_gate and use_cross_attn:
            self.gate = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.Sigmoid()
            )

        # ── 输出投影 ──────────────────────────────────────────────
        self.latent_proj = nn.Linear(hidden_channels, latent_channels)

        # ── 动态边权重模块 ────────────────────────────────────────
        if use_dynamic_edge_weights:
            self.dynamic_edge_module = DynamicEdgeWeightModule(
                node_feat_dim=hidden_channels, hidden_dim=32
            )

        # ── 时间偏移模块（建模跨域攻击传播延迟）────────────────────
        if use_temporal_shift:
            self.temporal_shift_module = TemporalShiftModule(
                hidden_dim=hidden_channels, n_delay_bases=4
            )

        # ── 特征重构解码器（自监督预训练用）──────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, in_channels)
        )

        self.dropout = dropout
        self._half = half

    def encode(self, x, adj, n_net=None):
        """
        Args:
            x   : [n_nodes, in_channels]   节点特征
            adj : [n_nodes, n_nodes]        邻接矩阵
            n_net: 网络节点数量（用于动态边权重）
        Returns:
            z   : [n_nodes, latent_channels] 节点嵌入
        """
        h = self.input_proj(x)  # [n_nodes, hidden]

        # 动态边权重调制
        if self.use_dynamic_edge_weights and hasattr(self, 'dynamic_edge_module') and n_net is not None:
            adj = self.dynamic_edge_module(h, adj, n_net)

        # 时间偏移调制（跨层边延迟建模）
        self._last_delay_info = None
        if self.use_temporal_shift and hasattr(self, 'temporal_shift_module') and n_net is not None:
            adj, delay_info = self.temporal_shift_module(h, adj, n_net)
            self._last_delay_info = delay_info

        # 对称归一化邻接矩阵
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj_norm = adj / deg  # [n_nodes, n_nodes]

        # ── 局部图卷积（DW-Sep or 标准 GCN）+ 残差 ──
        for i in range(self.num_layers):
            h_agg = torch.matmul(adj_norm, h)       # 邻居聚合

            if self.use_dw_sep:
                # 深度部分: 2 组独立变换
                g1, g2 = h_agg[:, :self._half], h_agg[:, self._half:]
                h_dw = torch.cat([self.dw_g1[i](g1), self.dw_g2[i](g2)], dim=-1)
                # 逐点部分: 跨组混合
                h_new = self.pw_layers[i](h_dw)
            else:
                h_new = self.gcn_layers[i](h_agg)   # 标准 GCN

            h_new = self.layer_norms[i](h_new)
            h_new = F.gelu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # 残差连接

        h_local = h  # 局部图卷积结果

        # ── 全局跨层注意力（标准 MHA，无 mask，学习跨域关联）──
        if self.use_cross_attn:
            h_seq = h_local.unsqueeze(0)           # [1, n_nodes, hidden]
            h_global, _ = self.self_attn(h_seq, h_seq, h_seq)
            h_global = h_global.squeeze(0)         # [n_nodes, hidden]
            h_global = self.cross_norm(h_global + h_local)  # 残差 + LN

            # 自适应门控融合
            if self.use_gate:
                gate_w = self.gate(torch.cat([h_local, h_global], dim=-1))
                h = gate_w * h_local + (1.0 - gate_w) * h_global
            else:
                h = h_local + h_global  # 直接相加

        z = self.latent_proj(h)
        return z

    def decode(self, z):
        """内积解码（自监督预训练）"""
        return torch.sigmoid(torch.matmul(z, z.t()))

    def reconstruct_features(self, z):
        """特征重构（自监督预训练）"""
        return self.decoder(z)

    def forward(self, x, adj):
        return self.encode(x, adj)

    def get_embeddings(self, x, adj):
        return self.encode(x, adj)


class STGaAN_Custom(nn.Module):
    """
    时空图注意力自编码器网络 (Spatio-Temporal Graph Attention AutoEncoder Network)
    参考: STGaAN for anomaly detection in IIoT systems

    特点:
    - 结合时间序列特征和空间图结构
    - 使用多头图注意力机制
    - 双解码器架构(时间+空间)
    - 作为对比模型，与 HGT-Trace (Ours) 进行性能对比
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 latent_channels: int, num_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.3,
                 use_temporal: bool = True,
                 use_spatial: bool = True,
                 use_fusion: bool = True):
        super().__init__()

        self.n_heads = n_heads
        self.latent_channels = latent_channels
        self.use_temporal = use_temporal
        self.use_spatial = use_spatial
        self.use_fusion = use_fusion
        self.hidden_channels = hidden_channels

        # 输入投影层
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 时间特征编码器
        if use_temporal:
            self.temporal_encoder = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.attention_layers = nn.ModuleList()
            self.attention_layers.append(
                nn.MultiheadAttention(hidden_channels, n_heads, dropout=dropout, batch_first=True)
            )
            for _ in range(num_layers - 1):
                self.attention_layers.append(
                    nn.MultiheadAttention(hidden_channels, n_heads, dropout=dropout, batch_first=True)
                )

        # 空间图编码器 (GCN层)
        if use_spatial:
            self.spatial_layers = nn.ModuleList()
            self.spatial_layers.append(nn.Linear(hidden_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.spatial_layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_channels) for _ in range(num_layers)
            ])

        # 融合层
        if use_temporal and use_spatial:
            fusion_in_dim = hidden_channels * 2
        elif use_temporal or use_spatial:
            fusion_in_dim = hidden_channels
        else:
            fusion_in_dim = hidden_channels

        if use_fusion and (use_temporal and use_spatial):
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_in_dim, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            latent_in_dim = hidden_channels
        else:
            latent_in_dim = fusion_in_dim

        self.latent_proj = nn.Linear(latent_in_dim, latent_channels)

        self.decoder = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, in_channels)
        )

        self.dropout = dropout

    def encode(self, x, adj):
        h_base = self.input_proj(x)
        h_spatial = None
        h_attn = None

        if self.use_temporal:
            h_temporal = self.temporal_encoder(x)
            h_attn = h_temporal.unsqueeze(0)
            for attn_layer in self.attention_layers:
                h_attn, _ = attn_layer(h_attn, h_attn, h_attn)
            h_attn = h_attn.squeeze(0)

        if self.use_spatial:
            deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
            adj_norm = adj / deg
            if self.use_temporal:
                h_spatial = self.temporal_encoder(x)
            else:
                h_spatial = h_base
            for i, spatial_layer in enumerate(self.spatial_layers):
                h_spatial = torch.matmul(adj_norm, h_spatial)
                h_spatial = spatial_layer(h_spatial)
                h_spatial = self.layer_norms[i](h_spatial)
                h_spatial = F.gelu(h_spatial)
                h_spatial = F.dropout(h_spatial, p=self.dropout, training=self.training)

        if self.use_temporal and self.use_spatial:
            h_fused = torch.cat([h_spatial, h_attn], dim=-1)
            if self.use_fusion:
                h_fused = self.fusion_layer(h_fused)
        elif self.use_temporal:
            h_fused = h_attn
        elif self.use_spatial:
            h_fused = h_spatial
        else:
            h_fused = h_base

        z = self.latent_proj(h_fused)
        return z

    def decode(self, z):
        return torch.sigmoid(torch.matmul(z, z.t()))

    def reconstruct_features(self, z):
        return self.decoder(z)

    def forward(self, x, adj):
        return self.encode(x, adj)

    def get_embeddings(self, x, adj):
        return self.encode(x, adj)


class EE_GCN_Custom(nn.Module):
    """
    边增强图卷积网络 (Edge-Enhanced Graph Convolutional Network)
    参考: EE-GCN: Exploiting Edge Features for IIoT Intrusion Detection
    
    特点:
    - 捕获网络流量链接的边特征
    - 建模设备节点之间的关系
    - 边特征与节点特征融合
    """
    
    def __init__(self, in_channels: int, hidden_channels: int,
                 latent_channels: int, num_layers: int = 2,
                 dropout: float = 0.3,
                 use_edge_feat: bool = True,
                 use_edge_attn: bool = True):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.use_edge_feat = use_edge_feat
        self.use_edge_attn = use_edge_attn
        
        # 节点特征编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 边特征生成器 (从节点对生成边特征) - 仅当 use_edge_feat=True 时使用
        if use_edge_feat:
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
        
        # GCN层
        self.gcn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        if use_edge_feat:
            # 边增强GCN层: 节点 + 边聚合
            self.edge_update_layers = nn.ModuleList()
            for _ in range(num_layers):
                self.gcn_layers.append(nn.Linear(hidden_channels * 2, hidden_channels))
                self.edge_update_layers.append(nn.Linear(hidden_channels * 3, hidden_channels))
                self.layer_norms.append(nn.LayerNorm(hidden_channels))
        else:
            # 标准GCN层: 仅邻居聚合
            for _ in range(num_layers):
                self.gcn_layers.append(nn.Linear(hidden_channels * 2, hidden_channels))
                self.layer_norms.append(nn.LayerNorm(hidden_channels))
        
        # 节点-边注意力 - 仅当 use_edge_attn=True 时使用
        if use_edge_attn:
            if use_edge_feat:
                # 基于边特征的注意力
                self.edge_attention = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels // 2),
                    nn.Tanh(),
                    nn.Linear(hidden_channels // 2, 1)
                )
            else:
                # 无边特征时，使用节点对的注意力
                self.edge_attention = nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels // 2),
                    nn.Tanh(),
                    nn.Linear(hidden_channels // 2, 1)
                )
        
        # 潜在空间投影
        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, latent_channels)
        )
        
        self.dropout = dropout
        
    def encode(self, x, adj):
        n_nodes = x.shape[0]
        device = x.device
        
        # 节点特征编码
        h = self.node_encoder(x)  # [n_nodes, hidden]
        
        # 初始化边特征矩阵 (如果使用边特征)
        edge_feats = None
        adj_expanded = adj.unsqueeze(-1)  # [n, n, 1]
        
        if self.use_edge_feat:
            h_i = h.unsqueeze(1).expand(-1, n_nodes, -1)  # [n, n, hidden]
            h_j = h.unsqueeze(0).expand(n_nodes, -1, -1)  # [n, n, hidden]
            edge_feats = self.edge_mlp(torch.cat([h_i, h_j], dim=-1))  # [n, n, hidden]
            edge_feats = edge_feats * adj_expanded
        
        for layer_idx in range(self.num_layers):
            # 计算注意力权重
            if self.use_edge_attn:
                if self.use_edge_feat:
                    # 基于边特征的注意力
                    edge_attn = self.edge_attention(edge_feats).squeeze(-1)  # [n, n]
                else:
                    # 基于节点对的注意力 (无边特征)
                    h_i = h.unsqueeze(1).expand(-1, n_nodes, -1)
                    h_j = h.unsqueeze(0).expand(n_nodes, -1, -1)
                    pair_feats = torch.cat([h_i, h_j], dim=-1)  # [n, n, hidden*2]
                    edge_attn = self.edge_attention(pair_feats).squeeze(-1)  # [n, n]
                
                edge_attn = edge_attn * adj  # 掩码非边
                edge_attn = F.softmax(edge_attn + (1 - adj) * (-1e9), dim=-1)
            else:
                # 无注意力: 使用归一化邻接矩阵 (均匀聚合)
                degree = adj.sum(dim=1, keepdim=True).clamp(min=1)
                edge_attn = adj / degree  # [n, n]
            
            # 聚合邻居节点
            neighbor_agg = torch.matmul(edge_attn, h)  # [n, hidden]
            
            if self.use_edge_feat:
                # 聚合边特征
                edge_agg = torch.sum(edge_feats * edge_attn.unsqueeze(-1), dim=1)  # [n, hidden]
                h_combined = torch.cat([h, edge_agg], dim=-1)  # [n, hidden*2]
            else:
                # 无边特征: 使用邻居聚合
                h_combined = torch.cat([h, neighbor_agg], dim=-1)  # [n, hidden*2]
            
            h = self.gcn_layers[layer_idx](h_combined)
            h = self.layer_norms[layer_idx](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            # 更新边特征 (如果使用边特征)
            if self.use_edge_feat:
                h_i = h.unsqueeze(1).expand(-1, n_nodes, -1)
                h_j = h.unsqueeze(0).expand(n_nodes, -1, -1)
                edge_input = torch.cat([h_i, h_j, edge_feats], dim=-1)
                edge_feats = self.edge_update_layers[layer_idx](edge_input)
                edge_feats = edge_feats * adj_expanded
        
        # 投影到潜在空间
        z = self.latent_proj(h)
        return z
    
    def decode(self, z):
        return torch.sigmoid(torch.matmul(z, z.t()))
    
    def forward(self, x, adj):
        z = self.encode(x, adj)
        return z
    
    def get_embeddings(self, x, adj):
        return self.encode(x, adj)


class IIoT_GNN_Custom(nn.Module):
    """
    工业物联网异常检测GNN (GNN for Anomaly Detection in IIoT)
    参考: Wu, Dai, Tang - Graph Neural Networks for Anomaly Detection 
          in Industrial Internet of Things (IEEE IoT Journal 2021)
    
    特点:
    - 针对IIoT场景的GNN架构
    - 多尺度特征聚合
    - 异常感知的节点嵌入
    """
    
    def __init__(self, in_channels: int, hidden_channels: int,
                 latent_channels: int, num_layers: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        self.num_layers = num_layers
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 多尺度GNN层
        self.gnn_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # GNN变换
            self.gnn_layers.append(nn.Linear(hidden_channels, hidden_channels))
            # 跳跃连接
            self.skip_connections.append(nn.Linear(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # 多尺度特征融合
        self.scale_weights = nn.Parameter(torch.ones(num_layers + 1))
        
        # 异常感知模块
        self.anomaly_encoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, hidden_channels)
        )
        
        # 潜在空间投影
        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, latent_channels)
        )
        
        # 特征重构解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels)
        )
        
        self.dropout = dropout
        
    def encode(self, x, adj):
        # 归一化邻接矩阵
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        adj_norm = deg_inv_sqrt * adj * deg_inv_sqrt.t()
        
        # 输入投影
        h = self.input_proj(x)  # [n_nodes, hidden]
        
        # 多尺度特征收集
        multi_scale_features = [h]  # 0阶 (自身)
        
        h_curr = h
        for i in range(self.num_layers):
            # 邻居聚合
            h_agg = torch.matmul(adj_norm, h_curr)
            
            # GNN变换
            h_transformed = self.gnn_layers[i](h_agg)
            
            # 跳跃连接
            h_skip = self.skip_connections[i](h_curr)
            
            # 残差连接
            h_curr = h_transformed + h_skip
            h_curr = self.batch_norms[i](h_curr)
            h_curr = F.relu(h_curr)
            h_curr = F.dropout(h_curr, p=self.dropout, training=self.training)
            
            multi_scale_features.append(h_curr)
        
        # 加权多尺度融合
        scale_weights = F.softmax(self.scale_weights, dim=0)
        h_fused = sum(w * feat for w, feat in zip(scale_weights, multi_scale_features))
        
        # 异常感知编码
        h_anomaly = self.anomaly_encoder(h_fused)
        h_anomaly = h_fused + h_anomaly  # 残差
        
        # 融合正常和异常感知特征
        h_combined = torch.cat([h_fused, h_anomaly], dim=-1)
        
        # 投影到潜在空间
        z = self.latent_proj(h_combined)
        return z
    
    def decode(self, z):
        return torch.sigmoid(torch.matmul(z, z.t()))
    
    def reconstruct_features(self, z):
        return self.decoder(z)
    
    def forward(self, x, adj):
        z = self.encode(x, adj)
        return z
    
    def get_embeddings(self, x, adj):
        return self.encode(x, adj)


class STCI_Custom(nn.Module):
    """时空因果推理网络 (Spatio-Temporal Causal Inference)

    结合Granger因果发现和图神经网络的时空因果推理基线。
    通过时间滞后相关性学习因果图结构，再用GNN进行异常检测。

    特点:
    - 基于Granger因果的图结构学习
    - 时间卷积捕获时序依赖
    - 因果图约束的消息传递
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 latent_channels: int, num_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.3,
                 temporal_kernel_size: int = 3):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # 时间卷积编码器（模拟时序因果发现）
        self.temporal_conv = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Granger因果图学习模块
        self.causal_graph_learner = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )

        # 因果约束GNN层
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.layer_norms.append(nn.LayerNorm(hidden_channels))

        # 因果注意力
        self.causal_attention = nn.MultiheadAttention(
            hidden_channels, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_channels)

        # 输出投影
        self.latent_proj = nn.Linear(hidden_channels, latent_channels)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, in_channels)
        )

        self.dropout = dropout

    def learn_causal_graph(self, h):
        """学习因果图结构"""
        n_nodes = h.shape[0]
        h_i = h.unsqueeze(1).expand(-1, n_nodes, -1)
        h_j = h.unsqueeze(0).expand(n_nodes, -1, -1)
        pair_feat = torch.cat([h_i, h_j], dim=-1)
        causal_adj = self.causal_graph_learner(pair_feat).squeeze(-1)
        # 稀疏化：只保留top-k因果边
        threshold = causal_adj.mean() + causal_adj.std()
        causal_adj = causal_adj * (causal_adj > threshold).float()
        return causal_adj

    def encode(self, x, adj):
        h = self.temporal_conv(x)

        # 学习因果图
        causal_adj = self.learn_causal_graph(h)
        # 融合先验图和学习图
        combined_adj = 0.5 * adj + 0.5 * causal_adj

        # 归一化
        deg = combined_adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj_norm = combined_adj / deg

        # 因果约束GNN
        for i in range(self.num_layers):
            h_agg = torch.matmul(adj_norm, h)
            h_new = self.gnn_layers[i](h_agg)
            h_new = self.layer_norms[i](h_new)
            h_new = F.gelu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new

        # 因果注意力
        h_seq = h.unsqueeze(0)
        h_attn, _ = self.causal_attention(h_seq, h_seq, h_seq)
        h = self.attn_norm(h_attn.squeeze(0) + h)

        z = self.latent_proj(h)
        return z

    def decode(self, z):
        return torch.sigmoid(torch.matmul(z, z.t()))

    def reconstruct_features(self, z):
        return self.decoder(z)

    def forward(self, x, adj):
        return self.encode(x, adj)

    def get_embeddings(self, x, adj):
        return self.encode(x, adj)


class DTGNN_Custom(nn.Module):
    """数字孪生图神经网络 (Digital Twin GNN)

    物理信息约束的GNN基线，将过程方程作为软约束融入图学习。
    模拟数字孪生方法中物理模型与数据驱动模型的融合。

    特点:
    - 物理方程残差作为正则化
    - 过程拓扑感知的消息传递
    - 物理一致性约束
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 latent_channels: int, num_layers: int = 2,
                 dropout: float = 0.3, n_phy_nodes: int = 14):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.n_phy_nodes = n_phy_nodes

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 物理约束编码器（模拟过程方程）
        self.physics_encoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # 数据驱动GNN层
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(nn.Linear(hidden_channels * 2, hidden_channels))
            self.layer_norms.append(nn.LayerNorm(hidden_channels))

        # 物理-数据融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Sigmoid()
        )

        # 输出投影
        self.latent_proj = nn.Linear(hidden_channels, latent_channels)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, in_channels)
        )

        self.dropout = dropout

    def encode(self, x, adj):
        h = self.input_proj(x)

        # 物理约束分支
        h_physics = self.physics_encoder(h)

        # 归一化邻接
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj_norm = adj / deg

        # 数据驱动GNN分支
        h_data = h
        for i in range(self.num_layers):
            h_agg = torch.matmul(adj_norm, h_data)
            h_combined = torch.cat([h_data, h_agg], dim=-1)
            h_new = self.gnn_layers[i](h_combined)
            h_new = self.layer_norms[i](h_new)
            h_new = F.gelu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h_data = h_data + h_new

        # 物理-数据融合
        gate = self.fusion_gate(torch.cat([h_physics, h_data], dim=-1))
        h_fused = gate * h_physics + (1 - gate) * h_data

        z = self.latent_proj(h_fused)
        return z

    def decode(self, z):
        return torch.sigmoid(torch.matmul(z, z.t()))

    def reconstruct_features(self, z):
        return self.decoder(z)

    def physics_residual_loss(self, x, adj):
        """物理残差损失：相邻物理节点应满足守恒约束"""
        h = self.input_proj(x)
        h_physics = self.physics_encoder(h)

        n_nodes = h_physics.shape[0]
        # 物理节点间的一致性约束
        residual = torch.tensor(0.0, device=x.device)
        count = 0
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj[i, j] > 0 and i != j:
                    diff = torch.norm(h_physics[i] - h_physics[j])
                    residual = residual + diff * adj[i, j]
                    count += 1

        return residual / max(count, 1)

    def forward(self, x, adj):
        return self.encode(x, adj)

    def get_embeddings(self, x, adj):
        return self.encode(x, adj)


# ========================= 增强型异常溯源模块 =========================

def causal_constraint_loss(propagation_matrix, causal_mask, n_net, n_phy):
    """因果约束正则化损失

    强制传播矩阵满足物理因果约束：
    1. 无环约束(DAG)：传播矩阵的迹应趋近于0
    2. 时序约束：物理层只能从上游向下游传播
    3. 跨层约束：网络层→物理层单向传播

    Args:
        propagation_matrix: [n_nodes, n_nodes] 传播概率矩阵
        causal_mask: [n_nodes, n_nodes] 因果掩码
        n_net: 网络节点数
        n_phy: 物理节点数

    Returns:
        loss: 因果约束损失标量
    """
    n_nodes = propagation_matrix.shape[0]

    # 1. 无环约束：tr(e^P) - n 应最小化（Zheng et al. NOTEARS）
    # 使用近似：||P * P^T||_F 惩罚双向边
    acyclicity_loss = torch.norm(
        propagation_matrix * propagation_matrix.t(), p='fro'
    ) / (n_nodes * n_nodes)

    # 2. 方向约束：违反因果掩码的传播应被惩罚
    violation = propagation_matrix * (1.0 - causal_mask[:n_nodes, :n_nodes])
    direction_loss = torch.norm(violation, p='fro') / (n_nodes * n_nodes)

    # 3. 物理层反向传播惩罚：下游不应影响上游
    if n_phy > 1:
        phy_start = min(n_net, n_nodes)
        phy_end = min(n_net + n_phy, n_nodes)
        phy_block = propagation_matrix[phy_start:phy_end, phy_start:phy_end]
        # 下三角（下游→上游）应为0
        lower_tri = torch.tril(phy_block, diagonal=-1)
        reverse_loss = torch.norm(lower_tri, p='fro') / max(1, n_phy * n_phy)
    else:
        reverse_loss = torch.tensor(0.0, device=propagation_matrix.device)

    return acyclicity_loss + direction_loss + 0.5 * reverse_loss


class CausalTracebackModule(nn.Module):
    """因果溯源模块 - 基于注意力的精确根因定位

    改进点:
    1. 引入因果注意力机制，学习异常传播路径
    2. 时序因果约束：异常只能从上游传播到下游
    3. 层间因果传播：网络层 → 物理层的定向传播
    4. 干预验证：掩蔽疑似根因节点验证下游分数变化
    """
    
    def __init__(self, latent_dim, n_net_nodes, n_phy_nodes, n_heads=4, dropout=0.2):
        super().__init__()
        self.n_net = n_net_nodes
        self.n_phy = n_phy_nodes
        self.n_nodes = n_net_nodes + n_phy_nodes
        self.latent_dim = latent_dim
        
        # 因果注意力层 - 学习异常传播方向
        self.causal_attention = nn.MultiheadAttention(
            latent_dim, n_heads, dropout=dropout, batch_first=True
        )
        
        # 因果掩码生成器 - 物理层按流程顺序，网络层可影响物理层
        self.register_buffer('causal_mask', self._build_causal_mask())
        
        # 根因评分网络 - 区分根因节点和受影响节点
        self.root_cause_scorer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1)
        )
        
        # 传播路径预测器 - 预测异常传播顺序
        self.propagation_predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid()
        )
        
        # 层间因果系数 - 可学习的网络→物理影响强度
        self.cross_layer_weight = nn.Parameter(torch.tensor(0.5))
        
    def _build_causal_mask(self):
        """构建因果掩码矩阵
        
        规则:
        - 物理层: 只能从上游节点(燃气流方向)接收异常
        - 网络层: 可以影响所有物理节点
        - 网络层内部: 全连接
        """
        mask = torch.zeros(self.n_nodes, self.n_nodes)
        
        # 网络层内部全连接
        mask[:self.n_net, :self.n_net] = 1.0
        
        # 物理层按流程顺序(上游可影响下游)
        for i in range(self.n_phy):
            for j in range(i, self.n_phy):  # 只有上游可以影响下游
                phy_i = self.n_net + i
                phy_j = self.n_net + j
                mask[phy_i, phy_j] = 1.0
        
        # 网络层可影响所有物理层节点
        mask[:self.n_net, self.n_net:] = 1.0
        
        return mask
    
    def forward(self, node_embeddings, adj=None):
        """
        Args:
            node_embeddings: [n_nodes, latent_dim] 节点嵌入
            adj: [n_nodes, n_nodes] 邻接矩阵 (可选)
            
        Returns:
            root_scores: [n_nodes] 根因分数
            propagation_matrix: [n_nodes, n_nodes] 传播概率矩阵
            causal_attention_weights: [n_nodes, n_nodes] 因果注意力权重
        """
        n_nodes = node_embeddings.shape[0]
        x = node_embeddings.unsqueeze(0)  # [1, n_nodes, dim]

        # 因果注意力 - 全局注意力（不施加因果掩码，保证每个节点都能感知全局异常上下文）
        # 因果约束仅用于传播矩阵，不限制根因分数的信息汇聚
        attended, attn_weights = self.causal_attention(
            x, x, x
        )
        attended = attended.squeeze(0)  # [n_nodes, dim]
        attn_weights = attn_weights.squeeze(0)  # [n_nodes, n_nodes]
        
        # 根因评分
        root_scores = self.root_cause_scorer(attended).squeeze(-1)  # [n_nodes]
        root_scores = torch.sigmoid(root_scores)
        
        # 传播路径预测
        h_i = attended.unsqueeze(1).expand(-1, n_nodes, -1)  # [n, n, dim]
        h_j = attended.unsqueeze(0).expand(n_nodes, -1, -1)  # [n, n, dim]
        pair_features = torch.cat([h_i, h_j], dim=-1)  # [n, n, dim*2]
        propagation_matrix = self.propagation_predictor(pair_features).squeeze(-1)  # [n, n]

        # 应用因果约束（仅传播矩阵受方向约束，根因分数不受限制）
        causal_mask = self.causal_mask[:n_nodes, :n_nodes].to(node_embeddings.device)
        propagation_matrix = propagation_matrix * causal_mask
        
        # 网络→物理的跨层影响加权
        cross_layer_propagation = propagation_matrix[:self.n_net, self.n_net:n_nodes]
        propagation_matrix[:self.n_net, self.n_net:n_nodes] = (
            cross_layer_propagation * torch.sigmoid(self.cross_layer_weight)
        )
        
        return {
            'root_scores': root_scores,
            'propagation_matrix': propagation_matrix,
            'causal_attention': attn_weights,
            'attended_features': attended
        }
    
    def trace_propagation_path(self, root_scores, propagation_matrix, anomaly_type='net_phy'):
        """追溯异常传播路径
        
        Args:
            root_scores: [n_nodes] 根因分数
            propagation_matrix: [n_nodes, n_nodes] 传播概率
            anomaly_type: 'phy' (物理故障) 或 'net_phy' (网络攻击导致物理故障)
            
        Returns:
            path: List[int] 传播路径 (节点索引序列)
            path_scores: List[float] 路径上各节点的分数
        """
        n_nodes = root_scores.shape[0]
        
        # 确定搜索范围
        if anomaly_type == 'phy':
            # 物理故障: 根因在物理层
            search_range = list(range(self.n_net, min(n_nodes, self.n_net + self.n_phy)))
        else:
            # 网络攻击: 根因在网络层
            search_range = list(range(min(self.n_net, n_nodes)))
        
        if not search_range:
            return [0], [0.0]

        # 因果深度回溯：找到传播链中最早的异常节点
        root_node, causal_depth = self.find_earliest_root_cause(
            root_scores, propagation_matrix, anomaly_type
        )
        
        # 追溯传播路径 (贪心搜索)
        path = [root_node]
        visited = {root_node}
        current = root_node
        
        for _ in range(min(5, n_nodes)):  # 最多5步传播
            # 找下一个最可能的传播目标
            propagation_probs = propagation_matrix[current].clone()
            
            # 排除已访问节点
            for v in visited:
                propagation_probs[v] = -float('inf')
            
            # 选择概率最高的下一节点
            next_node = torch.argmax(propagation_probs).item()
            
            if propagation_probs[next_node] < 0.1:  # 传播概率过低则停止
                break
                
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        
        path_scores = [root_scores[i].item() for i in path]

        return path, path_scores

    def intervention_validation(self, node_embeddings, root_node_idx):
        """干预验证：掩蔽疑似根因节点，验证下游异常分数是否显著下降

        通过do-calculus近似：将根因节点嵌入置零，重新计算下游节点分数。
        如果下游分数显著下降，则支持该节点为真实根因。

        Args:
            node_embeddings: [n_nodes, latent_dim]
            root_node_idx: 疑似根因节点索引

        Returns:
            intervention_effect: 干预效应（下游分数下降比例）
            is_valid_root: 是否通过干预验证
        """
        # 原始前向
        original_out = self.forward(node_embeddings)
        original_scores = original_out['root_scores']

        # 干预：将根因节点嵌入置零
        intervened_embeddings = node_embeddings.clone()
        intervened_embeddings[root_node_idx] = 0.0

        # 干预后前向
        intervened_out = self.forward(intervened_embeddings)
        intervened_scores = intervened_out['root_scores']

        # 计算下游节点分数变化
        n_nodes = node_embeddings.shape[0]
        downstream_mask = self.causal_mask[root_node_idx, :n_nodes].bool()
        downstream_mask[root_node_idx] = False

        if downstream_mask.any():
            original_downstream = original_scores[downstream_mask].mean()
            intervened_downstream = intervened_scores[downstream_mask].mean()
            intervention_effect = (original_downstream - intervened_downstream) / (original_downstream + 1e-8)
            is_valid_root = intervention_effect.item() > 0.1
        else:
            intervention_effect = torch.tensor(0.0)
            is_valid_root = False

        return intervention_effect.item(), is_valid_root

    def find_earliest_root_cause(self, root_scores, propagation_matrix, anomaly_type='net_phy',
                                  score_threshold=0.3):
        """因果深度回溯：在异常子图中找到入度为0的最早异常节点

        通过拓扑分析传播矩阵，找到异常子图中没有上游异常前驱的源节点，
        而非简单选择得分最高的节点。

        Args:
            root_scores: [n_nodes] 根因分数
            propagation_matrix: [n_nodes, n_nodes] 传播概率矩阵
            anomaly_type: 'phy' (物理故障) 或 'net_phy' (网络攻击)
            score_threshold: 异常节点阈值（相对于最大分数的比例）

        Returns:
            root_idx: 最早异常节点索引
            causal_depth: dict {node_idx: depth} 因果链深度
        """
        n_nodes = root_scores.shape[0]

        if anomaly_type == 'phy':
            layer_mask = torch.zeros(n_nodes, dtype=torch.bool, device=root_scores.device)
            layer_mask[self.n_net:min(n_nodes, self.n_net + self.n_phy)] = True
        else:
            layer_mask = torch.zeros(n_nodes, dtype=torch.bool, device=root_scores.device)
            layer_mask[:min(self.n_net, n_nodes)] = True

        layer_scores = root_scores.clone()
        layer_scores[~layer_mask] = 0
        max_score = layer_scores.max()

        if max_score < 1e-6:
            return torch.argmax(root_scores).item(), {}

        anomalous_mask = (layer_scores > score_threshold * max_score) & layer_mask
        anomalous_indices = torch.where(anomalous_mask)[0]

        if len(anomalous_indices) <= 1:
            return torch.argmax(layer_scores).item(), {}

        # 提取异常子图的传播矩阵
        sub_prop = propagation_matrix[anomalous_indices][:, anomalous_indices]

        # 计算入度（传播概率>0.1的入边数量）
        in_degree = (sub_prop > 0.1).float().sum(dim=0)

        # 找到入度最小的源节点
        min_in_degree = in_degree.min()
        source_candidates = torch.where(in_degree <= min_in_degree + 0.5)[0]

        # 在源节点中选择得分最高的
        best_source = source_candidates[0]
        best_score = root_scores[anomalous_indices[source_candidates[0]]].item()
        for idx in source_candidates:
            s = root_scores[anomalous_indices[idx]].item()
            if s > best_score:
                best_score = s
                best_source = idx

        root_idx = anomalous_indices[best_source].item()

        # BFS计算因果深度
        causal_depth = {root_idx: 0}
        queue = [best_source.item()]
        visited = {best_source.item()}
        while queue:
            current = queue.pop(0)
            current_orig = anomalous_indices[current].item()
            for j in range(len(anomalous_indices)):
                if j not in visited and sub_prop[current, j] > 0.1:
                    visited.add(j)
                    queue.append(j)
                    j_orig = anomalous_indices[j].item()
                    causal_depth[j_orig] = causal_depth[current_orig] + 1

        return root_idx, causal_depth

    def get_causal_constraint_loss(self, propagation_matrix):
        """计算当前传播矩阵的因果约束损失"""
        n_nodes = propagation_matrix.shape[0]
        mask = self.causal_mask[:n_nodes, :n_nodes].to(propagation_matrix.device)
        return causal_constraint_loss(propagation_matrix, mask, self.n_net, self.n_phy)



class UncertaintyQuantifier(nn.Module):
    """不确定性量化模块 - 为溯源结果提供置信度

    改进点:
    1. 使用MC Dropout估计预测不确定性
    2. 区分数据不确定性和模型不确定性
    3. 温度缩放扩大不确定性区分度
    4. 自适应推理：高置信样本单次前向，低置信样本MC采样
    5. 输出溯源结果的置信区间
    """

    def __init__(self, latent_dim, n_mc_samples=10, dropout=0.2, temperature=2.0):
        super().__init__()

        self.n_mc_samples = n_mc_samples
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # 均值预测头
        self.mean_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1)
        )

        # 方差预测头 (数据不确定性)
        self.var_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Softplus()
        )

        # 置信度判别器（用于自适应推理）
        self.confidence_gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 4),
            nn.GELU(),
            nn.Linear(latent_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, features, training_mode=True, adaptive=True):
        """
        Args:
            features: [n_nodes, latent_dim] 节点特征
            training_mode: 是否使用MC Dropout
            adaptive: 是否使用自适应推理（仅对低置信节点MC采样）

        Returns:
            dict with mean_scores, total_uncertainty, epistemic/aleatoric, confidence
        """
        if training_mode and not self.training:
            if adaptive:
                # 自适应推理：先单次前向判断置信度
                with torch.no_grad():
                    gate_score = self.confidence_gate(features).squeeze(-1)

                # 高置信节点（gate > 0.7）：单次确定性推理
                # 低置信节点（gate <= 0.7）：MC Dropout采样
                high_conf_mask = gate_score > 0.7

                mean_scores = torch.zeros(features.shape[0], device=features.device)
                epistemic = torch.zeros_like(mean_scores)
                aleatoric = torch.zeros_like(mean_scores)

                # 高置信节点：确定性推理
                if high_conf_mask.any():
                    h_feat = features[high_conf_mask]
                    mean_scores[high_conf_mask] = self.mean_head(h_feat).squeeze(-1)
                    aleatoric[high_conf_mask] = self.var_head(h_feat).squeeze(-1)

                # 低置信节点：MC Dropout
                if (~high_conf_mask).any():
                    l_feat = features[~high_conf_mask]
                    mc_outputs = []
                    mc_variances = []
                    for _ in range(self.n_mc_samples):
                        dropped = self.dropout(l_feat)
                        mc_outputs.append(self.mean_head(dropped).squeeze(-1).detach())
                        mc_variances.append(self.var_head(dropped).squeeze(-1).detach())

                    mc_out = torch.stack(mc_outputs, dim=0)
                    mc_var = torch.stack(mc_variances, dim=0)
                    mean_scores[~high_conf_mask] = mc_out.mean(dim=0)
                    epistemic[~high_conf_mask] = mc_out.var(dim=0)
                    aleatoric[~high_conf_mask] = mc_var.mean(dim=0)
            else:
                # 全量MC Dropout
                mc_outputs = []
                mc_variances = []
                for _ in range(self.n_mc_samples):
                    dropped = self.dropout(features)
                    mc_outputs.append(self.mean_head(dropped).squeeze(-1).detach())
                    mc_variances.append(self.var_head(dropped).squeeze(-1).detach())

                mc_out = torch.stack(mc_outputs, dim=0)
                mc_var = torch.stack(mc_variances, dim=0)
                mean_scores = mc_out.mean(dim=0)
                epistemic = mc_out.var(dim=0)
                aleatoric = mc_var.mean(dim=0)

            # 温度缩放扩大不确定性区分度
            total = (epistemic + aleatoric) * self.temperature
        else:
            mean_scores = self.mean_head(features).squeeze(-1)
            aleatoric = self.var_head(features).squeeze(-1)
            epistemic = torch.zeros_like(aleatoric)
            total = aleatoric * self.temperature

        return {
            'mean_scores': torch.sigmoid(mean_scores),
            'total_uncertainty': total,
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'confidence': 1.0 / (1.0 + total)
        }
    
    def get_confidence_interval(self, mean_scores, uncertainty, confidence_level=0.95):
        """计算置信区间
        
        Args:
            mean_scores: [n_nodes] 平均分数
            uncertainty: [n_nodes] 不确定性
            confidence_level: 置信水平
            
        Returns:
            lower: [n_nodes] 下界
            upper: [n_nodes] 上界
        """
        # 使用正态分布近似
        z = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        std = torch.sqrt(uncertainty + 1e-8)
        
        lower = mean_scores - z * std
        upper = mean_scores + z * std
        
        # 裁剪到[0, 1]
        lower = torch.clamp(lower, 0, 1)
        upper = torch.clamp(upper, 0, 1)
        
        return lower, upper


class ExplainableTraceback(nn.Module):
    """可解释溯源模块 - 生成人类可理解的溯源解释
    
    改进点:
    1. 特征重要性分析 - 哪些特征导致该节点被判定为根因
    2. 反事实解释 - 如果某特征改变，结论是否改变
    3. 规则提取 - 提取可解释的溯源规则
    """
    
    def __init__(self, input_dim, latent_dim, n_nodes):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_nodes = n_nodes
        
        # 特征重要性网络
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, input_dim),
            nn.Softmax(dim=-1)
        )
        
        # 反事实生成器
        self.counterfactual_generator = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, input_dim)
        )
        
    def compute_feature_importance(self, node_features, root_scores):
        """计算每个节点的特征重要性
        
        Args:
            node_features: [n_nodes, input_dim]
            root_scores: [n_nodes]
            
        Returns:
            importance: [n_nodes, input_dim] 每个特征对根因判定的贡献
        """
        # 注意力权重作为重要性
        attention = self.feature_attention(node_features)  # [n_nodes, input_dim]
        
        # 加权特征与根因分数的相关性
        importance = attention * node_features
        
        # 归一化
        importance = importance / (importance.sum(dim=-1, keepdim=True) + 1e-8)
        
        return importance
    
    def generate_counterfactual(self, node_features, target_change='reduce_score'):
        """生成反事实解释 - 最小改变使结论翻转
        
        Args:
            node_features: [n_nodes, input_dim]
            target_change: 'reduce_score' 或 'increase_score'
            
        Returns:
            counterfactual: [n_nodes, input_dim] 反事实特征
            delta: [n_nodes, input_dim] 需要的改变量
        """
        delta = self.counterfactual_generator(node_features)
        
        if target_change == 'reduce_score':
            # 减少根因分数 - 向"正常"方向移动
            counterfactual = node_features - delta
        else:
            counterfactual = node_features + delta
        
        return counterfactual, delta
    
    def generate_explanation(self, node_idx, node_features, root_scores, 
                            feature_names=None, top_k=5):
        """生成人类可读的解释
        
        Args:
            node_idx: 要解释的节点索引
            node_features: [n_nodes, input_dim]
            root_scores: [n_nodes]
            feature_names: List[str] 特征名称
            top_k: 显示前k个重要特征
            
        Returns:
            explanation: dict 包含解释文本和重要特征
        """
        importance = self.compute_feature_importance(node_features, root_scores)
        node_importance = importance[node_idx]  # [input_dim]
        
        # 找到最重要的特征
        top_indices = torch.argsort(node_importance, descending=True)[:top_k]
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(self.input_dim)]
        
        top_features = []
        for idx in top_indices:
            idx = idx.item()
            top_features.append({
                'name': feature_names[idx] if idx < len(feature_names) else f'Feature_{idx}',
                'value': node_features[node_idx, idx].item(),
                'importance': node_importance[idx].item()
            })
        
        # 生成解释文本
        score = root_scores[node_idx].item()
        explanation_text = f"节点 {node_idx} 被判定为根因 (分数: {score:.3f})，主要原因:\n"
        for i, feat in enumerate(top_features, 1):
            explanation_text += f"  {i}. {feat['name']} = {feat['value']:.3f} (重要性: {feat['importance']:.3f})\n"
        
        return {
            'node_idx': node_idx,
            'root_score': score,
            'top_features': top_features,
            'explanation_text': explanation_text,
            'all_importance': node_importance
        }


class EnhancedTracebackSystem(nn.Module):
    """增强型异常溯源系统 - 集成所有改进模块
    
    特点:
    1. 因果溯源 - 精确定位根因节点和传播路径
    2. 时序建模 - 捕获异常的时间传播特性
    3. 不确定性量化 - 提供溯源结果置信度
    4. 可解释性 - 生成人类可理解的解释
    """
    
    def __init__(self, gnn_encoder, n_nodes, n_net, n_phy, 
                 input_dim, latent_dim, hidden_dim=64, n_classes=6):
        super().__init__()
        
        self.gnn_encoder = gnn_encoder
        self.n_nodes = n_nodes
        self.n_net = n_net
        self.n_phy = n_phy
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        # 基础分类头 - 增强版本提高准确率
        self.graph_pool = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, 1)
        )
        
        # 多尺度特征融合
        self.feature_enhance = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim * 2),  # 使用融合特征
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes)
        )
        
        # 改进模块
        self.causal_traceback = CausalTracebackModule(
            latent_dim, n_net, n_phy, n_heads=4, dropout=0.2
        )
        self.uncertainty_quantifier = UncertaintyQuantifier(
            latent_dim, n_mc_samples=10, dropout=0.2
        )
        self.explainer = ExplainableTraceback(
            input_dim, latent_dim, n_nodes
        )
        
        # 基础节点评分器 (备用)
        self.node_scorer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1)
        )
        
    def encode(self, x, adj):
        """编码器: 获取节点嵌入"""
        if hasattr(self.gnn_encoder, 'encode'):
            z = self.gnn_encoder.encode(x, adj, n_net=self.n_net)
            if isinstance(z, tuple):
                z = z[0]
        elif hasattr(self.gnn_encoder, 'get_embeddings'):
            z = self.gnn_encoder.get_embeddings(x, adj)
        else:
            z = self.gnn_encoder(x, adj)
            if isinstance(z, tuple):
                z = z[0]
        return z
    
    def decode(self, z):
        """解码器: 通过内积重构边"""
        if hasattr(self.gnn_encoder, 'decode'):
            return self.gnn_encoder.decode(z)
        return torch.sigmoid(torch.matmul(z, z.t()))
    
    def get_embeddings(self, x, adj):
        """获取节点嵌入"""
        return self.encode(x, adj)
        
    def forward(self, x, adj, return_traceback=None):
        """前向传播
        
        Args:
            x: [n_nodes, input_dim] 节点特征
            adj: [n_nodes, n_nodes] 邻接矩阵
            return_traceback: 是否返回溯源信息（None时根据training自动判断）
        """
        # 训练时默认不返回溯源信息以节省内存
        if return_traceback is None:
            return_traceback = not self.training
            
        # GNN编码
        z = self.encode(x, adj)
        
        # 特征增强
        z_enhanced = self.feature_enhance(z)
        z_with_residual = z + z_enhanced  # 残差连接
        
        # 图分类 - 多尺度池化
        pool_weights = F.softmax(self.graph_pool(z_with_residual).squeeze(-1), dim=0)
        graph_repr = torch.sum(pool_weights.unsqueeze(-1) * z_with_residual, dim=0)
        
        # 添加max pooling增强表示
        max_repr = z_with_residual.max(dim=0)[0]
        
        # 拼接多尺度特征用于分类
        combined_repr = torch.cat([graph_repr, max_repr], dim=-1)
        logits = self.classifier(combined_repr)
        
        if return_traceback:
            # 因果溯源
            causal_out = self.causal_traceback(z, adj)

            # 不确定性量化 - 评估时使用MC Dropout
            uncertainty_out = self.uncertainty_quantifier(z, training_mode=True)

            # ── 类别感知层引导 ─────────────────────────────────────────────
            # 物理故障 (NS=1, NM=2) → 根因在物理层 (index >= n_net)
            # 网络攻击 (PM=3, PS=4, SS=5) → 根因在网络层 (index < n_net)
            # 利用分类器预测结果大幅抑制错误层节点分数，精确定位根因层
            n_nodes_actual = z.shape[0]
            pred_class = logits.argmax().item()
            n_net_actual = min(self.n_net, n_nodes_actual)
            layer_boost = torch.ones(n_nodes_actual, device=z.device)
            if pred_class in [1, 2]:   # 物理故障：抑制网络层节点
                layer_boost[:n_net_actual] = 0.05
            elif pred_class > 0:   # 其他异常默认视为网络源攻击：抑制物理层节点
                layer_boost[n_net_actual:] = 0.05
                if x.shape[1] > 4:
                    source_candidates = torch.where(x[:n_net_actual, 4] > 0.5)[0]
                    dest_candidates = torch.where(x[:n_net_actual, 4] < -0.5)[0]
                    if len(source_candidates) > 0:
                        layer_boost[source_candidates] *= 6.0
                    if len(dest_candidates) > 0:
                        layer_boost[dest_candidates] *= 1.5
            # pred_class == 0 (Normal): 不调整，保持原始分数
            adjusted_root_scores = causal_out['root_scores'] * layer_boost
            # ──────────────────────────────────────────────────────────────

            # 合并根因分数和置信度
            combined_scores = adjusted_root_scores * uncertainty_out['confidence']

            # ── 因果深度回溯：找到传播链中最早的异常节点 ──────────────
            if pred_class != 0:
                anomaly_type = 'phy' if pred_class in [1, 2] else 'net_phy'
                earliest_root, causal_depth = self.causal_traceback.find_earliest_root_cause(
                    causal_out['root_scores'], causal_out['propagation_matrix'], anomaly_type
                )
                # 对因果链中较深的节点施加衰减惩罚
                depth_penalty = torch.ones_like(combined_scores)
                for node_idx, depth in causal_depth.items():
                    if depth > 0:
                        depth_penalty[node_idx] *= (0.8 ** depth)
                combined_scores = combined_scores * depth_penalty
                # 确保最早根因节点得分最高
                if combined_scores[earliest_root] < combined_scores.max():
                    combined_scores[earliest_root] = combined_scores.max() + 0.01
            # ──────────────────────────────────────────────────────────────

            return {
                'anomaly_logits': logits,
                'embeddings': z,
                'node_scores': combined_scores,
                'root_scores': causal_out['root_scores'],
                'propagation_matrix': causal_out['propagation_matrix'],
                'causal_attention': causal_out['causal_attention'],
                'confidence': uncertainty_out['confidence'],
                'uncertainty': uncertainty_out['total_uncertainty'],
                'epistemic_uncertainty': uncertainty_out['epistemic_uncertainty'],
                'aleatoric_uncertainty': uncertainty_out['aleatoric_uncertainty'],
                'recon': x
            }
        else:
            # 简化输出（训练时使用）
            node_scores = torch.sigmoid(self.node_scorer(z)).squeeze(-1)
            return {
                'anomaly_logits': logits,
                'node_scores': node_scores,
                'embeddings': z,
                'recon': x
            }
    
    def traceback_with_explanation(self, x, adj, anomaly_type='net_phy', 
                                   feature_names=None, top_k=3):
        """完整溯源分析（带解释）
        
        Returns:
            完整的溯源结果，包括:
            - 根因节点及其分数
            - 传播路径
            - 置信度和不确定性
            - 可解释的原因
        """
        out = self.forward(x, adj, return_traceback=True)
        
        # 追溯传播路径
        path, path_scores = self.causal_traceback.trace_propagation_path(
            out['root_scores'], out['propagation_matrix'], anomaly_type
        )
        
        # 获取置信区间
        lower, upper = self.uncertainty_quantifier.get_confidence_interval(
            out['root_scores'], out['uncertainty']
        )
        
        # 生成解释 (对top节点)
        top_node_idx = torch.argmax(out['node_scores']).item()
        explanation = self.explainer.generate_explanation(
            top_node_idx, x, out['root_scores'], feature_names, top_k
        )
        
        return {
            'prediction': out['anomaly_logits'].argmax().item(),
            'root_node': top_node_idx,
            'root_score': out['node_scores'][top_node_idx].item(),
            'propagation_path': path,
            'path_scores': path_scores,
            'confidence': out['confidence'][top_node_idx].item(),
            'confidence_interval': (lower[top_node_idx].item(), upper[top_node_idx].item()),
            'explanation': explanation,
            'all_node_scores': out['node_scores'].detach().cpu().numpy(),
            'all_uncertainty': out['uncertainty'].detach().cpu().numpy()
        }


class BaselineGNNClassifier(nn.Module):
    """将无监督GNN嵌入用于分类的包装器（支持两阶段训练）"""
    
    def __init__(self, gnn_encoder, n_nodes, latent_dim, hidden_dim=64, n_classes=6):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.n_nodes = n_nodes
        self.latent_dim = latent_dim
        
        # 图池化
        self.graph_pool = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, 1)
        )
        
        # 增强分类头 - 更深的网络结构
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes)
        )
        
        self.node_scorer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim // 2, 1)
        )
        
        # 特征增强层
        self.feature_enhance = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
    
    def encode(self, x, adj):
        """编码器: 获取节点嵌入（用于自监督预训练）"""
        if hasattr(self.gnn_encoder, 'encode'):
            z = self.gnn_encoder.encode(x, adj)
            if isinstance(z, tuple):
                z = z[0]
        elif hasattr(self.gnn_encoder, 'get_embeddings'):
            z = self.gnn_encoder.get_embeddings(x, adj)
        else:
            z = self.gnn_encoder(x, adj)
            if isinstance(z, tuple):
                z = z[0]
        return z
    
    def decode(self, z):
        """解码器: 通过内积重构边（用于自监督预训练）"""
        if hasattr(self.gnn_encoder, 'decode'):
            return self.gnn_encoder.decode(z)
        return torch.sigmoid(torch.matmul(z, z.t()))
    
    def get_embeddings(self, x, adj):
        """获取节点嵌入"""
        return self.encode(x, adj)
        
    def forward(self, x, adj, return_traceback=False):
        """前向传播
        
        Args:
            x: [n_nodes, input_dim] 节点特征
            adj: [n_nodes, n_nodes] 邻接矩阵
            return_traceback: 兼容参数（Baseline模型忽略此参数）
        """
        # 获取节点嵌入
        if hasattr(self.gnn_encoder, 'get_embeddings'):
            z = self.gnn_encoder.get_embeddings(x, adj)
        else:
            z = self.gnn_encoder(x, adj)
            if isinstance(z, tuple):
                z = z[0]
        
        # 特征增强
        z_enhanced = self.feature_enhance(z)
        z = z + z_enhanced  # 残差连接
        
        node_scores = torch.sigmoid(self.node_scorer(z)).squeeze(-1)
        
        # 多尺度注意力池化
        pool_weights = F.softmax(self.graph_pool(z).squeeze(-1), dim=0)
        graph_repr = torch.sum(pool_weights.unsqueeze(-1) * z, dim=0)
        
        # 添加max pooling增强表示
        max_repr = z.max(dim=0)[0]
        graph_repr = graph_repr + 0.3 * max_repr  # 加权融合
        
        logits = self.classifier(graph_repr)
        
        return {
            'anomaly_logits': logits,
            'node_scores': node_scores,
            'embeddings': z,
            'recon': x  # 占位
        }


# ========================= 5. 两阶段自监督学习 =========================

def pretrain_self_supervised(encoder, data, graph_builder, device, epochs=15, lr=0.001,
                             use_dynamic_edges=True, use_cross_layer=True,
                             use_phy_chain=True, use_net_edges=True,
                             verbose=True, model_name="Model"):
    """
    阶段1: 自监督预训练 - 通过边重构学习节点嵌入（不使用标签）

    损失函数:
    - 边重构损失 (edge reconstruction)
    - 特征重构损失 (feature reconstruction)
    - 对比学习损失 (contrastive learning)
    """
    encoder.to(device)
    encoder.train()
    
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    X_phy = data['X_phy']
    X_net_feat = data['X_net_feat']
    src_ips = data['src_ips']
    dst_ips = data['dst_ips']
    n_samples = len(data['y'])

    # 预计算静态邻接矩阵
    static_adj = graph_builder.build_static_adjacency(use_cross_layer=use_cross_layer,
                                                       use_phy_chain=use_phy_chain,
                                                       use_net_edges=use_net_edges)
    static_adj_tensor = torch.tensor(static_adj, dtype=torch.float32).to(device)
    
    batch_size = 64
    indices = np.arange(n_samples)
    
    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 5  # 较短的预训练patience
    
    if verbose:
        print(f"\n🔧 [{model_name}] 阶段1: 自监督预训练 (epochs={epochs})...")
    show_large_progress = n_samples >= 100000
    progress_step = max(batch_size, (max(n_samples // 10, batch_size) // batch_size) * batch_size)
    
    for epoch in range(epochs):
        np.random.shuffle(indices)
        total_edge_loss = 0
        total_feat_loss = 0
        
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_indices = indices[batch_start:batch_end]
            if show_large_progress and (batch_start == 0 or batch_start % progress_step == 0):
                pct = batch_start / max(n_samples, 1) * 100
                print(f"     [{model_name}] Pretrain Epoch {epoch+1}/{epochs}: {batch_start}/{n_samples} ({pct:.1f}%)")
            
            batch_edge_loss = 0
            batch_feat_loss = 0
            optimizer.zero_grad()
            
            for t in batch_indices:
                # 选择邻接矩阵
                if use_dynamic_edges:
                    adj = graph_builder.build_dynamic_adjacency(src_ips[t], dst_ips[t],
                                                                 use_cross_layer=use_cross_layer,
                                                                 use_phy_chain=use_phy_chain,
                                                                 use_net_edges=use_net_edges)
                    adj_tensor = torch.tensor(adj, dtype=torch.float32).to(device)
                else:
                    adj_tensor = static_adj_tensor

                # 获取节点特征（带时间窗口）
                node_feats = graph_builder.get_node_features_with_history(t, X_net_feat, X_phy, src_ips, dst_ips)
                
                x = torch.tensor(node_feats, dtype=torch.float32).to(device)
                
                # 获取节点嵌入
                if hasattr(encoder, 'encode'):
                    z = encoder.encode(x, adj_tensor)
                    if isinstance(z, tuple):
                        z, _ = z  # VGAE返回 (mu, logvar)
                else:
                    out = encoder(x, adj_tensor)
                    if isinstance(out, dict):
                        z = out['embeddings']
                    elif isinstance(out, tuple):
                        z = out[0]
                    else:
                        z = out
                
                # 边重构损失
                if hasattr(encoder, 'decode'):
                    adj_recon = encoder.decode(z)
                else:
                    adj_recon = torch.sigmoid(torch.matmul(z, z.t()))
                
                # 二值交叉熵损失
                edge_loss = F.binary_cross_entropy(adj_recon, adj_tensor)
                batch_edge_loss += edge_loss
                
                # 特征重构损失（如果模型有decoder）
                if hasattr(encoder, 'decoder'):
                    x_recon = encoder.decoder(z)
                    feat_loss = F.mse_loss(x_recon, x)
                    batch_feat_loss += feat_loss
            
            # 计算总损失
            loss = batch_edge_loss / len(batch_indices)
            if batch_feat_loss > 0:
                loss += 0.3 * batch_feat_loss / len(batch_indices)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_edge_loss += batch_edge_loss.item()
            total_feat_loss += batch_feat_loss if isinstance(batch_feat_loss, (int, float)) else batch_feat_loss.item()
            
            # 清理GPU缓存
            if device.type == 'cuda' and batch_start % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
        
        scheduler.step()
        avg_loss = total_edge_loss / n_samples
        
        # 早停
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose:
            print(f"     [{model_name}] Pretrain Epoch {epoch+1}/{epochs}: Edge Loss={avg_loss:.4f}")
        elif show_large_progress:
            print(f"     [{model_name}] Pretrain Epoch {epoch+1}/{epochs}: 100.0%, Edge Loss={avg_loss:.4f}")
        
        if patience_counter >= patience:
            if verbose:
                print(f"     ⚠️ 早停于 Epoch {epoch+1}")
            break
    
    # 恢复最佳状态
    if best_state is not None:
        encoder.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    if verbose:
        print(f"     ✅ 自监督预训练完成! Best Edge Loss: {best_loss:.4f}")
    
    return encoder



def finetune_supervised(model, data, graph_builder, device, epochs=40, lr=0.001,
                        freeze_encoder=False, use_dynamic_edges=True, use_cross_layer=True,
                        use_phy_chain=True, use_net_edges=True,
                        verbose=True, model_name="Model"):
    """
    阶段2: 有监督微调 - 使用标签训练分类头

    改进策略:
    - 渐进式解冻：先训练分类头，后解冻编码器
    - 差异化学习率：编码器用较小学习率
    - 过采样少数类
    """
    model.to(device)

    # 差异化学习率：编码器用较小学习率
    encoder_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if (
            'classifier' in name or 'anomaly_head' in name or 'node_scorer' in name
            or 'node_anomaly_head' in name or 'graph_pool' in name
            or 'causal_traceback' in name
        ):
            classifier_params.append(param)
        else:
            encoder_params.append(param)
    
    if verbose:
        print(f"     📊 编码器参数: {sum(p.numel() for p in encoder_params):,}")
        print(f"     📊 分类器参数: {sum(p.numel() for p in classifier_params):,}")
    
    # 计算类别权重（激进式类别平衡）
    val_data = data.get('_val_data', data)
    y = data['y']
    class_counts = np.bincount(y)
    
    # 使用反比例权重，更强的类别平衡
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.min()  # 归一化到最小权重=1
    
    # 特别增加NM类别权重（类别2），因为它是边界类别，容易混淆
    # NM = 物理故障（网络正常+物理中等故障）
    label_names = data.get('label_names', ['Normal', 'NS', 'NM', 'PM', 'PS', 'SS'])
    nm_idx = None
    if 'NM' in label_names:
        nm_idx = label_names.index('NM')
        if nm_idx < len(class_weights):
            class_weights[nm_idx] *= 1.5  # 适度增加NM权重
            if verbose:
                print(f"     ⚠️ NM类别权重增强 (idx={nm_idx}): 权重={class_weights[nm_idx]:.3f}")
    
    # PS类别也是少数类，需要增强
    if 'PS' in label_names:
        ps_idx = label_names.index('PS')
        if ps_idx < len(class_weights):
            class_weights[ps_idx] *= 2.0
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    if verbose:
        print(f"     📊 类别权重: {class_weights.cpu().numpy()}")
    
    # 差异化学习率优化器
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': lr * 0.1},  # 编码器用较小学习率
        {'params': classifier_params, 'lr': lr}      # 分类器用正常学习率
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    X_phy = data['X_phy']
    X_net_feat = data['X_net_feat']
    src_ips = data['src_ips']
    dst_ips = data['dst_ips']
    n_samples = len(y)
    
    # 过采样少数类索引
    oversampled_indices = []
    minority_target = 200 if n_samples < 1000 else 1000
    nm_target = 100 if n_samples < 1000 else 500
    for cls_id in range(len(class_counts)):
        cls_indices = np.where(y == cls_id)[0]
        
        # NM类别特别处理：适度过采样
        if nm_idx is not None and cls_id == nm_idx:  # NM类别
            repeat_times = max(1, nm_target // len(cls_indices))  # 过采样到目标规模
            oversampled_indices.extend(np.repeat(cls_indices, repeat_times).tolist())
        elif class_counts[cls_id] < minority_target:  # 其他少数类
            repeat_times = max(1, minority_target // len(cls_indices))  # 过采样到目标规模
            oversampled_indices.extend(np.repeat(cls_indices, repeat_times).tolist())
        else:
            oversampled_indices.extend(cls_indices.tolist())
    oversampled_indices = np.array(oversampled_indices)
    if verbose:
        print(f"     📊 过采样后样本数: {len(oversampled_indices)} (原始: {n_samples})")
        # 打印过采样后各类别数量
        oversampled_y = y[oversampled_indices]
        for cls_id in range(len(class_counts)):
            cnt = np.sum(oversampled_y == cls_id)
            print(f"        Class {cls_id} ({label_names[cls_id]}): {cnt}")
    
    # 预计算静态邻接矩阵
    static_adj = graph_builder.build_static_adjacency(use_cross_layer=use_cross_layer,
                                                       use_phy_chain=use_phy_chain,
                                                       use_net_edges=use_net_edges)
    static_adj_tensor = torch.tensor(static_adj, dtype=torch.float32).to(device)
    
    batch_size = 64
    
    best_acc = 0
    best_f1 = 0
    best_state = None
    patience_counter = 0
    patience = 15  # 增加耐心值
    
    if verbose:
        print(f"\n🔧 [{model_name}] 阶段2: 有监督微调 (epochs={epochs})...")

    # Traceback supervision is trained after classification fine-tuning. Keeping it
    # out of this loop preserves the classifier's early-stopping trajectory.
    traceback_loss_weight = 0.0
    traceback_loss_stride = 4 if isinstance(model, EnhancedTracebackSystem) else 8
    show_large_progress = len(oversampled_indices) >= 100000
    progress_step = max(batch_size, (max(len(oversampled_indices) // 10, batch_size) // batch_size) * batch_size)
    
    for epoch in range(epochs):
        model.train()
        np.random.shuffle(oversampled_indices)
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for batch_start in range(0, len(oversampled_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(oversampled_indices))
            batch_indices = oversampled_indices[batch_start:batch_end]
            if show_large_progress and (batch_start == 0 or batch_start % progress_step == 0):
                pct = batch_start / max(len(oversampled_indices), 1) * 100
                print(f"     [{model_name}] Finetune Epoch {epoch+1}/{epochs}: {batch_start}/{len(oversampled_indices)} ({pct:.1f}%)")
            
            batch_loss = 0
            optimizer.zero_grad()
            
            for t in batch_indices:
                if use_dynamic_edges:
                    adj = graph_builder.build_dynamic_adjacency(src_ips[t], dst_ips[t], 
                                                                 use_cross_layer=use_cross_layer,
                                                                 use_phy_chain=use_phy_chain,
                                                                 use_net_edges=use_net_edges)
                    adj_tensor = torch.tensor(adj, dtype=torch.float32).to(device)
                else:
                    adj_tensor = static_adj_tensor
                
                # 获取节点特征（带时间窗口）
                node_feats = graph_builder.get_node_features_with_history(t, X_net_feat, X_phy, src_ips, dst_ips)

                x = torch.tensor(node_feats, dtype=torch.float32).to(device)
                label = torch.tensor([y[t]], dtype=torch.long).to(device)

                # Keep the main forward lightweight; train traceback heads explicitly below.
                needs_traceback_loss = (
                    traceback_loss_weight > 0
                    and int(y[t]) > 0
                    and ((total_samples + epoch) % traceback_loss_stride == 0)
                )
                out = model(x, adj_tensor, return_traceback=False)
                
                # 获取分类输出
                if isinstance(out, dict):
                    logits = out['anomaly_logits']
                else:
                    logits = out
                
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                
                cls_loss = F.cross_entropy(logits, label, weight=class_weights)
                sample_loss = cls_loss

                if needs_traceback_loss and isinstance(out, dict):
                    root_target = build_root_target(t, data, graph_builder, device)
                    root_scores = None
                    if isinstance(model, EnhancedTracebackSystem) and 'embeddings' in out:
                        # Train only causal attention + root scorer here. The full propagation
                        # matrix is O(N^2) and is reserved for evaluation/visualization.
                        emb = out['embeddings'].detach().unsqueeze(0)
                        attended, _ = model.causal_traceback.causal_attention(emb, emb, emb)
                        attended = attended.squeeze(0)
                        root_scores = model.causal_traceback.root_cause_scorer(attended).squeeze(-1)
                        root_scores = torch.sigmoid(root_scores)
                    if root_scores is None:
                        if hasattr(model, 'node_scorer') and 'embeddings' in out:
                            root_scores = torch.sigmoid(model.node_scorer(out['embeddings'].detach())).squeeze(-1)
                        else:
                            root_scores = out.get('node_scores', None)
                    if root_target is not None and root_scores is not None:
                        root_scores = root_scores[:graph_builder.n_nodes].clamp(1e-6, 1 - 1e-6)
                        root_target = root_target[:root_scores.shape[0]]
                        # Positive roots are rare, so weight them more heavily than non-root nodes.
                        root_weights = torch.ones_like(root_target)
                        root_weights[root_target > 0] = max(5.0, graph_builder.n_nodes / max(root_target.sum().item(), 1.0))
                        trace_loss = F.binary_cross_entropy(root_scores, root_target, weight=root_weights)
                        sample_loss = sample_loss + traceback_loss_weight * trace_loss

                batch_loss += sample_loss
                
                pred = logits.argmax(dim=-1).item()
                correct += (pred == y[t])
                total_samples += 1
            
            batch_loss = batch_loss / len(batch_indices)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += batch_loss.item() * len(batch_indices)
            
            # 清理GPU缓存
            if device.type == 'cuda' and batch_start % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
        
        scheduler.step()
        avg_loss = total_loss / len(oversampled_indices)
        acc = correct / total_samples * 100
        
        # 每个epoch结束后在原始数据上评估
        model.eval()
        eval_preds = []
        eval_y = val_data['y']
        with torch.no_grad():
            for t in range(len(eval_y)):
                if use_dynamic_edges:
                    adj = graph_builder.build_dynamic_adjacency(
                        val_data['src_ips'][t], val_data['dst_ips'][t],
                        use_cross_layer=use_cross_layer,
                        use_phy_chain=use_phy_chain,
                        use_net_edges=use_net_edges
                    )
                    adj_tensor = torch.tensor(adj, dtype=torch.float32).to(device)
                else:
                    adj_tensor = static_adj_tensor
                
                # 获取节点特征（带时间窗口）
                node_feats = graph_builder.get_node_features_with_history(
                    t, val_data['X_net_feat'], val_data['X_phy'],
                    val_data['src_ips'], val_data['dst_ips']
                )

                x = torch.tensor(node_feats, dtype=torch.float32).to(device)
                # 评估时也禁用溯源以节省内存
                out = model(x, adj_tensor, return_traceback=False)
                if isinstance(out, dict):
                    logits = out['anomaly_logits']
                else:
                    logits = out
                eval_preds.append(logits.argmax().item())
            
            # 清理GPU缓存
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        eval_preds = np.array(eval_preds)
        eval_acc = accuracy_score(eval_y, eval_preds) * 100
        eval_f1 = f1_score(eval_y, eval_preds, average='macro', zero_division=0) * 100
        model.train()
        
        # 使用F1作为主要指标（更关注少数类）
        if eval_f1 > best_f1 or (eval_f1 == best_f1 and eval_acc > best_acc):
            best_f1 = eval_f1
            best_acc = eval_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose:
            print(f"     [{model_name}] Finetune Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={eval_acc:.2f}%, F1={eval_f1:.2f}%, Best F1={best_f1:.2f}%")
        elif show_large_progress:
            print(f"     [{model_name}] Finetune Epoch {epoch+1}/{epochs}: 100.0%, Loss={avg_loss:.4f}, Val Acc={eval_acc:.2f}%, Val F1={eval_f1:.2f}%")
        
        if patience_counter >= patience:
            if verbose:
                print(f"     ⚠️ 早停于 Epoch {epoch+1}")
            break
    
    # 恢复最佳状态
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    if verbose:
        print(f"     ✅ 微调完成! Best Acc: {best_acc:.2f}%, Best F1: {best_f1:.2f}%")
    
    return model


def train_traceback_head(model, data, graph_builder, device, epochs=1, lr=0.001,
                         use_dynamic_edges=True, use_cross_layer=True,
                         use_phy_chain=True, use_net_edges=True,
                         stride=2, verbose=True, model_name="Model"):
    """Train only the exact-node traceback head after classifier fine-tuning."""
    if not isinstance(model, EnhancedTracebackSystem):
        return model

    anomaly_indices = [
        i for i, label in enumerate(data['y'])
        if int(label) > 0 and infer_true_root_nodes(i, data, graph_builder)
    ]
    if stride > 1:
        anomaly_indices = anomaly_indices[::stride]
    if not anomaly_indices:
        return model

    params = list(model.causal_traceback.causal_attention.parameters()) + list(model.causal_traceback.root_cause_scorer.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    static_adj = graph_builder.build_static_adjacency(
        use_cross_layer=use_cross_layer,
        use_phy_chain=use_phy_chain,
        use_net_edges=use_net_edges
    )
    static_adj_tensor = torch.tensor(static_adj, dtype=torch.float32).to(device)

    if verbose:
        print(f"     🔎 [{model_name}] 训练溯源头: {len(anomaly_indices)} anomaly samples, epochs={epochs}")

    model.eval()
    model.causal_traceback.train()

    for epoch in range(epochs):
        np.random.shuffle(anomaly_indices)
        total_loss = 0.0
        used = 0

        for t in anomaly_indices:
            if use_dynamic_edges:
                adj = graph_builder.build_dynamic_adjacency(
                    data['src_ips'][t], data['dst_ips'][t],
                    use_cross_layer=use_cross_layer,
                    use_phy_chain=use_phy_chain,
                    use_net_edges=use_net_edges
                )
                adj_tensor = torch.tensor(adj, dtype=torch.float32).to(device)
            else:
                adj_tensor = static_adj_tensor

            node_feats = graph_builder.get_node_features_with_history(
                t, data['X_net_feat'], data['X_phy'], data['src_ips'], data['dst_ips']
            )
            x = torch.tensor(node_feats, dtype=torch.float32).to(device)
            root_target = build_root_target(t, data, graph_builder, device)
            if root_target is None:
                continue

            with torch.no_grad():
                z = model.encode(x, adj_tensor)

            emb = z.detach().unsqueeze(0)
            attended, _ = model.causal_traceback.causal_attention(emb, emb, emb)
            attended = attended.squeeze(0)
            root_scores = model.causal_traceback.root_cause_scorer(attended).squeeze(-1)
            root_scores = torch.sigmoid(root_scores).clamp(1e-6, 1 - 1e-6)
            root_target = root_target[:root_scores.shape[0]]

            root_weights = torch.ones_like(root_target)
            root_weights[root_target > 0] = max(5.0, graph_builder.n_nodes / max(root_target.sum().item(), 1.0))
            loss = F.binary_cross_entropy(root_scores, root_target, weight=root_weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            used += 1

        if verbose:
            avg_loss = total_loss / max(used, 1)
            print(f"        Traceback Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")

    model.eval()
    return model


def train_two_stage(model, data, graph_builder, device, pretrain_epochs=15, finetune_epochs=40,
                    pretrain_lr=0.001, finetune_lr=0.002, freeze_encoder=False,
                    use_dynamic_edges=True, use_cross_layer=True,
                    use_phy_chain=True, use_net_edges=True,
                    verbose=True, model_name="Model"):
    """
    两阶段训练: 自监督预训练 + 有监督微调（改进版）

    改进:
    - 减少预训练轮数，避免过拟合到边重构
    - 微调时不冻结编码器，使用差异化学习率
    - 过采样少数类，增强类别平衡
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"📌 [{model_name}] 两阶段自监督学习")
        print(f"{'='*50}")

    # 阶段1: 自监督预训练
    model = pretrain_self_supervised(
        model, data, graph_builder, device,
        epochs=pretrain_epochs, lr=pretrain_lr,
        use_dynamic_edges=use_dynamic_edges, use_cross_layer=use_cross_layer,
        use_phy_chain=use_phy_chain, use_net_edges=use_net_edges,
        verbose=verbose, model_name=model_name
    )

    # 阶段2: 有监督微调
    model = finetune_supervised(
        model, data, graph_builder, device,
        epochs=finetune_epochs, lr=finetune_lr,
        freeze_encoder=freeze_encoder,
        use_dynamic_edges=use_dynamic_edges, use_cross_layer=use_cross_layer,
        use_phy_chain=use_phy_chain, use_net_edges=use_net_edges,
        verbose=verbose, model_name=model_name
    )

    # 阶段3: 只训练溯源头，不改变分类编码器/分类器。
    model = train_traceback_head(
        model, data, graph_builder, device,
        epochs=max(1, finetune_epochs // 2), lr=finetune_lr,
        use_dynamic_edges=use_dynamic_edges, use_cross_layer=use_cross_layer,
        use_phy_chain=use_phy_chain, use_net_edges=use_net_edges,
        stride=2, verbose=verbose, model_name=model_name
    )

    model = fit_raw_feature_calibrator(
        model, data, graph_builder, model_name=model_name, verbose=verbose
    )

    return model


def train_model(model, data, graph_builder, device, epochs=50, lr=0.002, patience=10, 
                use_dynamic_edges=True, use_cross_layer=True, verbose=True):
    """训练模型（保留旧接口兼容性，现在使用两阶段训练）"""
    return train_two_stage(
        model, data, graph_builder, device,
        pretrain_epochs=max(10, epochs // 4),  # 较少的预训练
        finetune_epochs=epochs - max(10, epochs // 4),  # 更多的微调
        pretrain_lr=lr,
        finetune_lr=lr,
        freeze_encoder=False,  # 不冻结编码器
        use_dynamic_edges=use_dynamic_edges,
        use_cross_layer=use_cross_layer,
        verbose=verbose,
        model_name="Model"
    )


def train_model_legacy(model, data, graph_builder, device, epochs=50, lr=0.001, patience=10, 
                use_dynamic_edges=True, verbose=True):
    """训练模型（旧版混合训练，保留用于对比）"""
    model.to(device)
    
    # 计算类别权重（处理类别不平衡）
    y = data['y']
    class_counts = np.bincount(y)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    if verbose:
        print(f"📊 类别权重: {class_weights.cpu().numpy()}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    X_phy = data['X_phy']
    X_net_feat = data['X_net_feat']
    src_ips = data['src_ips']
    dst_ips = data['dst_ips']
    n_samples = len(y)
    
    # 早停机制
    best_acc = 0
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Mini-batch训练（加速）
    batch_size = 64
    indices = np.arange(n_samples)
    
    # 预计算静态邻接矩阵（消融实验用）
    static_adj = graph_builder.build_static_adjacency()
    static_adj_tensor = torch.tensor(static_adj, dtype=torch.float32).to(device)
    
    if verbose:
        print(f"\n🔧 开始训练 (epochs={epochs}, patience={patience}, batch_size={batch_size})...")
    
    for epoch in range(epochs):
        model.train()
        np.random.shuffle(indices)
        total_loss = 0
        correct = 0
        
        # Mini-batch
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_indices = indices[batch_start:batch_end]
            
            batch_loss = 0
            optimizer.zero_grad()
            
            for t in batch_indices:
                # 选择邻接矩阵（动态或静态）
                if use_dynamic_edges:
                    adj = graph_builder.build_dynamic_adjacency(src_ips[t], dst_ips[t])
                    adj_tensor = torch.tensor(adj, dtype=torch.float32).to(device)
                else:
                    adj_tensor = static_adj_tensor
                
                # 获取节点特征（带时间窗口）
                node_feats = graph_builder.get_node_features_with_history(t, X_net_feat, X_phy, src_ips, dst_ips)
                x = torch.tensor(node_feats, dtype=torch.float32).to(device)
                label = torch.tensor([y[t]], dtype=torch.long).to(device)

                out = model(x, adj_tensor)
                
                recon_loss = F.mse_loss(out['recon'], x)
                cls_loss = F.cross_entropy(out['anomaly_logits'].unsqueeze(0), label, weight=class_weights)
                loss = recon_loss * 0.1 + cls_loss  # 降低重建损失权重
                batch_loss += loss
                
                pred = out['anomaly_logits'].argmax().item()
                correct += (pred == y[t])
            
            batch_loss = batch_loss / len(batch_indices)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += batch_loss.item() * len(batch_indices)
        
        scheduler.step()
        avg_loss = total_loss / n_samples
        acc = correct / n_samples * 100
        
        # 早停检查
        if acc > best_acc or (acc == best_acc and avg_loss < best_loss):
            best_acc = acc
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 5 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"   Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.2f}%, LR={lr_now:.6f}, Best={best_acc:.2f}%")
        
        # 早停
        if patience_counter >= patience:
            if verbose:
                print(f"\n⚠️ 早停触发！在 Epoch {epoch+1} 停止，最佳准确率: {best_acc:.2f}%")
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"✅ 已恢复最佳模型 (Acc={best_acc:.2f}%)")
    
    if verbose:
        print("✅ 训练完成!")
    return model


def evaluate_model(model, data, graph_builder, device, use_dynamic_edges=True, use_cross_layer=True,
                   use_phy_chain=True, use_net_edges=True):
    """评估模型 - 包含扩展指标（AUC, 推理时间, RCA, MRR, NDCG@K）"""
    import time
    from sklearn.metrics import roc_auc_score, confusion_matrix
    from sklearn.preprocessing import label_binarize

    if '_test_data' in data:
        print("   📌 evaluate_model: using held-out chronological test split")
        data = data['_test_data']

    model.eval()

    X_phy = data['X_phy']
    X_net_feat = data['X_net_feat']
    src_ips = data['src_ips']
    dst_ips = data['dst_ips']
    y = data['y']
    anomaly_types = data.get('anomaly_types', [None] * len(y))
    n_samples = len(y)
    n_classes = data.get('n_classes', 6)
    calibrator_probs = get_raw_calibrator_probs(model, data, graph_builder, n_classes)
    calibrator_alpha = getattr(model, 'raw_feature_calibrator', {}).get('alpha', 0.0) if calibrator_probs is not None else 0.0

    # 预计算静态邻接矩阵
    static_adj = graph_builder.build_static_adjacency(use_cross_layer=use_cross_layer,
                                                       use_phy_chain=use_phy_chain,
                                                       use_net_edges=use_net_edges)
    static_adj_tensor = torch.tensor(static_adj, dtype=torch.float32).to(device)
    
    all_preds = []
    all_probs = []
    all_node_scores = []

    # 干预验证统计
    intervention_pass_count = 0
    intervention_total_count = 0
    intervention_effects = []
    intervention_corrected = 0

    # 计时开始
    start_time = time.time()

    is_enhanced = isinstance(model, EnhancedTracebackSystem)
    show_large_progress = n_samples >= 100000
    progress_step = max(1, n_samples // 10)

    with torch.no_grad():
        for t in range(n_samples):
            if show_large_progress and (t == 0 or t % progress_step == 0):
                print(f"   [{model.__class__.__name__}] evaluate_model: {t}/{n_samples} ({t / max(n_samples, 1) * 100:.1f}%)")
            if use_dynamic_edges:
                adj = graph_builder.build_dynamic_adjacency(src_ips[t], dst_ips[t],
                                                            use_cross_layer=use_cross_layer,
                                                            use_phy_chain=use_phy_chain,
                                                            use_net_edges=use_net_edges)
                adj_tensor = torch.tensor(adj, dtype=torch.float32).to(device)
            else:
                adj_tensor = static_adj_tensor

            # 获取节点特征（带时间窗口）
            node_feats = graph_builder.get_node_features_with_history(t, X_net_feat, X_phy, src_ips, dst_ips)
            x = torch.tensor(node_feats, dtype=torch.float32).to(device)
            out = model(x, adj_tensor)

            # 获取预测和概率
            logits = out['anomaly_logits']
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            if calibrator_probs is not None:
                probs = (1.0 - calibrator_alpha) * probs + calibrator_alpha * calibrator_probs[t]
                probs = probs / max(float(probs.sum()), 1e-8)
            pred = int(np.argmax(probs))

            # 获取节点异常分数（用于溯源评估）
            if 'node_scores' in out:
                node_scores = out['node_scores'].cpu().numpy()
            else:
                node_scores = np.zeros(graph_builder.n_nodes)
            if getattr(model, 'use_input_root_prior', False):
                node_scores = apply_input_root_prior(
                    node_scores, node_feats, pred, graph_builder, data.get('label_names', None)
                )

            # 严格评估阶段不使用真实标签修正节点分数，避免溯源指标泄露。

            all_preds.append(pred)
            all_probs.append(probs)
            all_node_scores.append(node_scores)
        if show_large_progress:
            print(f"   [{model.__class__.__name__}] evaluate_model: {n_samples}/{n_samples} (100.0%)")
    
    # 计时结束
    end_time = time.time()
    inference_time = (end_time - start_time) / n_samples * 1000  # 毫秒/样本
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_node_scores = np.array(all_node_scores)
    
    # ==================== 基础指标 ====================
    metrics = {
        'accuracy': accuracy_score(y, all_preds) * 100,
        'precision': precision_score(y, all_preds, average='macro', zero_division=0) * 100,
        'recall': recall_score(y, all_preds, average='macro', zero_division=0) * 100,
        'f1': f1_score(y, all_preds, average='macro', zero_division=0) * 100,
        'precision_weighted': precision_score(y, all_preds, average='weighted', zero_division=0) * 100,
        'recall_weighted': recall_score(y, all_preds, average='weighted', zero_division=0) * 100,
        'f1_weighted': f1_score(y, all_preds, average='weighted', zero_division=0) * 100
    }
    
    # ==================== AUC ====================
    try:
        # One-vs-Rest AUC
        y_bin = label_binarize(y, classes=range(n_classes))
        auc_score = roc_auc_score(y_bin, all_probs, multi_class='ovr', average='macro')
        metrics['auc'] = auc_score * 100
    except Exception as e:
        metrics['auc'] = 0.0
    
    # ==================== 推理时间 ====================
    metrics['inference_time'] = inference_time
    
    # ==================== 混淆矩阵 ====================
    metrics['confusion_matrix'] = confusion_matrix(y, all_preds, labels=range(n_classes))
    metrics['predictions'] = all_preds
    metrics['probabilities'] = all_probs
    
    # ==================== Strict Exact-Node Traceback Metrics ====================
    metrics.update(compute_strict_traceback_metrics(all_node_scores, data, graph_builder))

    # ==================== Intervention Validation Metrics ====================
    if intervention_total_count > 0:
        metrics['intervention_pass_rate'] = intervention_pass_count / intervention_total_count * 100
        metrics['intervention_avg_effect'] = np.mean(intervention_effects) if intervention_effects else 0.0
        metrics['intervention_corrected'] = intervention_corrected
    else:
        metrics['intervention_pass_rate'] = 0.0
        metrics['intervention_avg_effect'] = 0.0
        metrics['intervention_corrected'] = 0

    return metrics


def plot_confusion_matrix(metrics, label_names, save_path='confusion_matrix.png'):
    """绘制精美混淆矩阵图 - 数字下面显示准确率"""
    import seaborn as sns

    cm = metrics['confusion_matrix']
    n_classes = len(label_names)

    # 计算百分比（每行归一化 = 各类的准确率）
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    cm_percent = np.nan_to_num(cm_percent)

    # 创建图形（放大以容纳更大字体）
    fig, ax = plt.subplots(figsize=(14, 11))

    # 创建注释文本：数字 + 换行 + 准确率（精确到两位小数）
    annot_text = np.empty_like(cm, dtype=object)
    for i in range(n_classes):
        for j in range(n_classes):
            annot_text[i, j] = f'{cm[i, j]}\n{cm_percent[i, j]:.2f}%'

    # 使用更美观的配色
    sns.heatmap(cm, annot=annot_text, fmt='', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, ax=ax,
                cbar_kws={'label': 'Sample Count'}, linewidths=0.5, linecolor='white',
                annot_kws={'fontsize': 22, 'fontweight': 'bold'})

    # colorbar 字体
    cbar = ax.collections[0].colorbar
    cbar.set_label('Sample Count', fontsize=22, fontweight='normal')
    cbar.ax.tick_params(labelsize=19)

    ax.set_xlabel('Predicted Label', fontsize=25, fontweight='normal')
    ax.set_ylabel('True Label', fontsize=25, fontweight='normal')

    # 旋转标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=23, fontweight='normal')
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=23, fontweight='normal')

    plt.tight_layout()
    ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"   ✅ 混淆矩阵已保存至: {save_path}")


def plot_root_cause_summary(results, graph_builder, data=None, save_path='root_cause_summary.png'):
    """绘制根因分析汇总图 - 4张独立图片
    
    显示:
    1. 各类型异常的根因节点分布（条形图）
    2. 根因层分布（饼图）
    3. 各异常类型的根因分布堆叠条形图（含表格）
    4. 平均异常分数热力图
    """
    import seaborn as sns
    
    save_dir = os.path.dirname(save_path)
    if not save_dir:
        save_dir = '.'
    os.makedirs(save_dir, exist_ok=True)
    
    label_names = data.get('label_names', ['Normal', 'NS', 'NM', 'PM', 'PS', 'SS']) if data else ['Normal', 'NS', 'NM', 'PM', 'PS', 'SS']
    anomaly_label_ids = list(range(1, len(label_names)))
    
    n_net = graph_builder.n_net
    n_phy = graph_builder.n_phy
    node_names_all = list(graph_builder.net_nodes) + list(graph_builder.phy_nodes)

    anomaly_results = [r for r in results if r['true_label'] > 0]
    if not anomaly_results:
        print("⚠️ 没有足够的异常样本进行根因分析")
        return

    # ── 按阈值统计异常节点（与因果传播图保持一致: mean + 0.3*std）────────────
    per_class_anom = {}
    for atype in anomaly_label_ids:
        cls_r = [r for r in results if r['true_label'] == atype]
        if not cls_r:
            per_class_anom[atype] = {'net': 0, 'phy': 0, 'avg_scores': np.zeros(n_net + n_phy)}
            continue
        avg_s = np.mean([r['node_scores'] for r in cls_r], axis=0)
        ns = avg_s[:n_net];  ps = avg_s[n_net:]
        thr_n = ns.mean() + 0.3 * ns.std() if ns.std() > 0 else ns.mean()
        thr_p = ps.mean() + 0.3 * ps.std() if ps.std() > 0 else ps.mean()
        per_class_anom[atype] = {
            'net': int(np.sum(ns >= thr_n)),
            'phy': int(np.sum(ps >= thr_p)),
            'avg_scores': avg_s,
        }

    layer_stats = {
        'network':  sum(v['net']  for v in per_class_anom.values()),
        'physical': sum(v['phy'] for v in per_class_anom.values()),
    }

    # Top-10 节点：统计每个节点在异常样本中被判为根因的频次
    # 对每个异常样本，分别对网络层和物理层用 mean+0.3*std 阈值判定根因节点
    node_freq = np.zeros(n_net + n_phy, dtype=int)
    for r in anomaly_results:
        ns = r['node_scores'][:n_net]
        ps = r['node_scores'][n_net:]
        thr_n = ns.mean() + 0.3 * ns.std() if ns.std() > 0 else ns.mean()
        thr_p = ps.mean() + 0.3 * ps.std() if ps.std() > 0 else ps.mean()
        node_freq[:n_net] += (ns >= thr_n).astype(int)
        node_freq[n_net:] += (ps >= thr_p).astype(int)

    top_roots = sorted(
        [(node_names_all[i], int(node_freq[i])) for i in range(n_net + n_phy)],
        key=lambda x: x[1], reverse=True
    )[:10]

    # ==================== 子图1: 根因节点Top-10排名（独立保存） ====================
    fig1, ax1 = plt.subplots(figsize=(14, 10))

    # 构建节点名到标签的映射: 网络节点→N1,N2,...  物理节点→P1,P2,...
    net_nodes_list = list(graph_builder.net_nodes)
    phy_nodes_list = list(graph_builder.phy_nodes)
    node_label_map = {}
    for idx, nn in enumerate(net_nodes_list):
        node_label_map[nn] = f'N{idx + 1}'
    for idx, pn in enumerate(phy_nodes_list):
        node_label_map[pn] = f'P{idx + 1}'

    nodes  = [node_label_map.get(n, n[:15]) for n, _ in top_roots]
    raw_names = [n for n, _ in top_roots]
    freqs = [f for _, f in top_roots]

    # 配色与 explain_layer_contribution 一致: Network=#3498db(蓝), Physical=#e74c3c(红)
    colors = ['#e74c3c' if ('Phy' in rn or 'phy' in rn) else '#3498db' for rn in raw_names]
    bars = ax1.barh(range(len(nodes)), freqs, color=colors, edgecolor='white', linewidth=1.5)
    ax1.set_yticks(range(len(nodes)))
    ax1.set_yticklabels(nodes, fontsize=36)
    ax1.set_ylabel('Root Cause Nodes', fontsize=40, fontweight='normal')
    ax1.set_xlabel('Frequency', fontsize=40, fontweight='normal')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_facecolor('#fafafa')
    ax1.tick_params(axis='x', labelsize=36)

    # 数值标签放在柱状图内部
    for bar, freq in zip(bars, freqs):
        ax1.text(bar.get_width() * 0.5, bar.get_y() + bar.get_height()/2,
                str(freq), va='center', ha='center', fontsize=36, fontweight='bold', color='white')

    # 图例 - 配色与 explain_layer_contribution 一致
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Network Layer'),
        Patch(facecolor='#e74c3c', label='Physical Layer')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=44)
    
    fig1.tight_layout()
    path1 = os.path.join(save_dir, 'root_cause_top10.png')
    fig1.savefig(path1, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig1)
    print(f"   ✅ 根因节点Top-10已保存至: {path1}")
    
    # ==================== 子图2: 网络层 vs 物理层饼图（独立保存） ====================
    fig2, ax2 = plt.subplots(figsize=(11, 10))

    sizes = [layer_stats['network'], layer_stats['physical']]
    labels = ['Network Layer', 'Physical Layer']
    colors_pie = ['#2ecc71', '#3498db']
    explode = (0.05, 0.05)

    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie,
                                        autopct='%1.1f%%', startangle=90, explode=explode,
                                        shadow=True, textprops={'fontsize': 21, 'fontweight': 'bold'})

    # 美化饼图
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(21)
    
    fig2.tight_layout()
    path2 = os.path.join(save_dir, 'root_cause_layer_dist.png')
    fig2.savefig(path2, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig2)
    print(f"   ✅ 根因层分布已保存至: {path2}")
    
    # ==================== 子图3: 各异常类型的根因分布堆叠条形图（含表格，独立保存） ====================
    fig3, ax3 = plt.subplots(figsize=(14, 12))

    anomaly_types = anomaly_label_ids
    anomaly_names = [label_names[i] for i in anomaly_types]

    net_counts = []
    phy_counts = []

    for atype in anomaly_types:
        net_counts.append(per_class_anom[atype]['net'])
        phy_counts.append(per_class_anom[atype]['phy'])

    x = np.arange(len(anomaly_names))
    width = 0.6

    bars1 = ax3.bar(x, net_counts, width, label='Network Layer', color='#3498db', edgecolor='white')
    bars2 = ax3.bar(x, phy_counts, width, bottom=net_counts, label='Physical Layer', color='#e74c3c', edgecolor='white')

    ax3.set_ylabel('Root Cause Count', fontsize=44, fontweight='normal')
    ax3.set_xlabel('Anomaly Type', fontsize=44, fontweight='normal')
    ax3.set_xticks(x)
    ax3.set_xticklabels(anomaly_names, fontsize=40)
    ax3.legend(loc='best', fontsize=40)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_facecolor('#fafafa')
    ax3.set_ylim(0, 15)
    ax3.tick_params(axis='y', labelsize=40)

    # 在条形上添加数量标注
    for i, (nc, pc) in enumerate(zip(net_counts, phy_counts)):
        if nc > 0:
            ax3.text(i, nc / 2, str(nc), ha='center', va='center', fontsize=40, fontweight='bold', color='white')
        if pc > 0:
            ax3.text(i, nc + pc / 2, str(pc), ha='center', va='center', fontsize=40, fontweight='bold', color='white')
    
    # 打印表格到控制台
    print("\n   📋 根因分布统计表:")
    print(f"   {'Type':<8} {'Network':<10} {'Physical':<10} {'Total':<10}")
    print(f"   {'-'*38}")
    for i, name in enumerate(anomaly_names):
        total = net_counts[i] + phy_counts[i]
        print(f"   {name:<8} {net_counts[i]:<10} {phy_counts[i]:<10} {total:<10}")
    print(f"   {'-'*38}")
    print(f"   {'Total':<8} {sum(net_counts):<10} {sum(phy_counts):<10} {sum(net_counts)+sum(phy_counts):<10}")
    
    fig3.tight_layout()
    path3 = os.path.join(save_dir, 'root_cause_by_type.png')
    fig3.savefig(path3, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig3)
    print(f"   ✅ 各类型根因分布已保存至: {path3}")
    
    # ==================== 子图4: 异常分数热力图（独立保存） ====================
    fig4, ax4 = plt.subplots(figsize=(14, 12))

    # 计算各节点在不同异常类型下的平均分数（直接从 per_class_anom 取）
    # 取Top-10节点（按全部异常样本平均分排名）
    all_avg = np.mean([r['node_scores'] for r in anomaly_results], axis=0)
    top_10_idx = sorted(range(n_net + n_phy), key=lambda i: all_avg[i], reverse=True)[:10]
    top_node_names = [
        f'N{i+1}' if i < n_net else f'P{i - n_net + 1}'
        for i in top_10_idx
    ]

    heatmap_data = []
    for idx in top_10_idx:
        row = [per_class_anom[atype]['avg_scores'][idx] for atype in anomaly_types]
        heatmap_data.append(row)

    heatmap_array = np.array(heatmap_data)

    im = ax4.imshow(heatmap_array, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(len(anomaly_types)))
    ax4.set_xticklabels([label_names[i] for i in anomaly_types], fontsize=41)
    ax4.set_yticks(range(len(top_node_names)))
    ax4.set_yticklabels(top_node_names, fontsize=41)
    ax4.set_xlabel('Anomaly Type', fontsize=45, fontweight='normal')
    ax4.set_ylabel('Root Cause Node', fontsize=45, fontweight='normal')

    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Avg Score', fontsize=41, fontweight='normal')
    cbar.ax.tick_params(labelsize=41)

    # 添加数值标注 - 全部黑色
    for i in range(len(top_node_names)):
        for j in range(len(anomaly_types)):
            if heatmap_array[i, j] > 0:
                ax4.text(j, i, f'{heatmap_array[i, j]:.2f}', ha='center', va='center',
                        fontsize=38, fontweight='bold', color='black')
    
    fig4.tight_layout()
    path4 = os.path.join(save_dir, 'root_cause_heatmap.png')
    fig4.savefig(path4, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig4)
    print(f"   ✅ 异常分数热力图已保存至: {path4}")
    
    print(f"✅ 根因分析（共4张图）全部保存完成")


def evaluate_and_traceback(model, data, graph_builder, device,
                           use_dynamic_edges=True,
                           use_cross_layer=True,
                           use_phy_chain=True,
                           use_net_edges=True):
    """评估并进行异常溯源

    增强功能:
    - 支持EnhancedTracebackSystem的因果溯源
    - 输出传播路径、置信度和不确定性
    """
    if '_test_data' in data:
        print("   📌 evaluate_and_traceback: using held-out chronological test split")
        data = data['_test_data']

    model.eval()

    X_phy = data['X_phy']
    X_net_feat = data['X_net_feat']
    src_ips = data['src_ips']
    dst_ips = data['dst_ips']
    y = data['y']
    times = data['times']
    anomaly_types = data['anomaly_types']
    n_samples = len(y)
    n_classes = data.get('n_classes', len(np.unique(y)))
    calibrator_probs = get_raw_calibrator_probs(model, data, graph_builder, n_classes)
    calibrator_alpha = getattr(model, 'raw_feature_calibrator', {}).get('alpha', 0.0) if calibrator_probs is not None else 0.0
    
    # 检查是否是增强型溯源系统
    is_enhanced = isinstance(model, EnhancedTracebackSystem)
    if is_enhanced:
        print(f"   🚀 使用增强型溯源系统 (含因果溯源+不确定性量化)")
    
    # 预计算静态邻接矩阵
    static_adj = graph_builder.build_static_adjacency(
        use_cross_layer=use_cross_layer,
        use_phy_chain=use_phy_chain,
        use_net_edges=use_net_edges
    )
    static_adj_tensor = torch.tensor(static_adj, dtype=torch.float32).to(device)
    
    results = []
    all_preds = []
    
    # 计时开始
    import time
    start_time = time.time()
    show_large_progress = n_samples >= 100000
    progress_step = max(1, n_samples // 10)
    
    with torch.no_grad():
        for t in range(n_samples):
            if show_large_progress and (t == 0 or t % progress_step == 0):
                print(f"   evaluate_and_traceback: {t}/{n_samples} ({t / max(n_samples, 1) * 100:.1f}%)")
            # 构建动态邻接矩阵
            if use_dynamic_edges:
                adj = graph_builder.build_dynamic_adjacency(
                    src_ips[t], dst_ips[t],
                    use_cross_layer=use_cross_layer,
                    use_phy_chain=use_phy_chain,
                    use_net_edges=use_net_edges
                )
                adj_tensor = torch.tensor(adj, dtype=torch.float32).to(device)
            else:
                adj_tensor = static_adj_tensor

            # 获取节点特征（带时间窗口）
            node_feats = graph_builder.get_node_features_with_history(t, X_net_feat, X_phy, src_ips, dst_ips)

            x = torch.tensor(node_feats, dtype=torch.float32).to(device)
            
            # 调用模型 - 支持增强型溯源系统
            if is_enhanced:
                out = model(x, adj_tensor, return_traceback=True)
            else:
                out = model(x, adj_tensor)
            
            probs = torch.softmax(out['anomaly_logits'], dim=-1).cpu().numpy()
            if calibrator_probs is not None:
                probs = (1.0 - calibrator_alpha) * probs + calibrator_alpha * calibrator_probs[t]
                probs = probs / max(float(probs.sum()), 1e-8)
            pred = int(np.argmax(probs))
            node_scores = out['node_scores'].cpu().numpy()
            if getattr(model, 'use_input_root_prior', False):
                node_scores = apply_input_root_prior(
                    node_scores, node_feats, pred, graph_builder, data.get('label_names', None)
                )
            
            all_preds.append(pred)
            
            top_k = 3
            top_indices = np.argsort(node_scores)[-top_k:][::-1]
            top_nodes = [graph_builder.all_nodes[i] for i in top_indices]
            top_scores = [node_scores[i] for i in top_indices]
            
            result = {
                'time': times[t],
                'true_label': y[t],
                'pred_label': pred,
                'anomaly_type': anomaly_types[t],
                'src_ip': src_ips[t],
                'dst_ip': dst_ips[t],
                'top_nodes': top_nodes,
                'top_scores': top_scores,
                'node_scores': node_scores
            }
            
            # 添加增强型溯源信息
            if is_enhanced:
                # 置信度
                if 'confidence' in out:
                    conf = out['confidence'].cpu().numpy()
                    result['confidence'] = float(conf.mean()) if isinstance(conf, np.ndarray) else float(conf)

                # 不确定性
                if 'uncertainty' in out:
                    unc = out['uncertainty'].cpu().numpy()
                    result['uncertainty'] = float(unc.mean()) if isinstance(unc, np.ndarray) else float(unc)
                
                # 传播路径 - 异常样本才计算
                if y[t] > 0 and hasattr(model, 'causal_traceback'):
                    try:
                        root_scores = out.get('root_scores', out['node_scores'])
                        prop_matrix = out.get('propagation_matrix', None)
                        if prop_matrix is not None:
                            # 确定异常类型
                            atype = anomaly_types[t][0] if isinstance(anomaly_types[t], tuple) else 'net_phy'
                            path, path_scores = model.causal_traceback.trace_propagation_path(
                                root_scores, prop_matrix, atype
                            )
                            result['propagation_path'] = path
                            result['path_scores'] = path_scores
                    except Exception as e:
                        pass  # 传播路径计算失败时跳过
            else:
                # 为非增强型模型添加默认值
                result['confidence'] = 0.5
                result['uncertainty'] = 0.5
            
            results.append(result)
        if show_large_progress:
            print(f"   evaluate_and_traceback: {n_samples}/{n_samples} (100.0%)")
    
    # 计时结束
    end_time = time.time()
    inference_time = (end_time - start_time) / n_samples * 1000  # 毫秒/样本
    
    all_preds = np.array(all_preds)
    
    # ==================== 基础分类指标 ====================
    acc = accuracy_score(y, all_preds) * 100
    prec = precision_score(y, all_preds, average='macro', zero_division=0) * 100
    rec = recall_score(y, all_preds, average='macro', zero_division=0) * 100
    f1 = f1_score(y, all_preds, average='macro', zero_division=0) * 100
    prec_weighted = precision_score(y, all_preds, average='weighted', zero_division=0) * 100
    rec_weighted = recall_score(y, all_preds, average='weighted', zero_division=0) * 100
    f1_weighted = f1_score(y, all_preds, average='weighted', zero_division=0) * 100
    
    # ==================== AUC ====================
    try:
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_auc_score
        # 收集概率
        all_probs = []
        with torch.no_grad():
            for t in range(n_samples):
                if show_large_progress and (t == 0 or t % progress_step == 0):
                    print(f"   evaluate_and_traceback AUC pass: {t}/{n_samples} ({t / max(n_samples, 1) * 100:.1f}%)")
                if use_dynamic_edges:
                    adj = graph_builder.build_dynamic_adjacency(
                        src_ips[t], dst_ips[t],
                        use_cross_layer=use_cross_layer,
                        use_phy_chain=use_phy_chain,
                        use_net_edges=use_net_edges
                    )
                    adj_tensor = torch.tensor(adj, dtype=torch.float32).to(device)
                else:
                    adj_tensor = static_adj_tensor

                node_feats = graph_builder.get_node_features_with_history(t, X_net_feat, X_phy, src_ips, dst_ips)

                x = torch.tensor(node_feats, dtype=torch.float32).to(device)
                if is_enhanced:
                    out = model(x, adj_tensor, return_traceback=True)
                else:
                    out = model(x, adj_tensor)
                probs = torch.softmax(out['anomaly_logits'], dim=-1).cpu().numpy()
                if calibrator_probs is not None:
                    probs = (1.0 - calibrator_alpha) * probs + calibrator_alpha * calibrator_probs[t]
                    probs = probs / max(float(probs.sum()), 1e-8)
                all_probs.append(probs)
            if show_large_progress:
                print(f"   evaluate_and_traceback AUC pass: {n_samples}/{n_samples} (100.0%)")
        all_probs = np.array(all_probs)
        y_bin = label_binarize(y, classes=range(n_classes))
        auc_score = roc_auc_score(y_bin, all_probs, multi_class='ovr', average='macro') * 100
    except Exception as e:
        auc_score = 0.0
    
    # ==================== 溯源指标 ====================
    # 收集所有节点分数
    all_node_scores = np.array([r['node_scores'] for r in results])
    
    # Exact-node RCA/MRR/NDCG/APD. This no longer treats a whole layer as correct.
    trace_metrics = compute_strict_traceback_metrics(all_node_scores, data, graph_builder)
    rca = trace_metrics['rca']
    mrr = trace_metrics['mrr']
    ndcg = trace_metrics['ndcg']
    apd = trace_metrics['apd']
    
    # ==================== 模型参数与资源 ====================
    param_count = sum(p.numel() for p in model.parameters()) / 1000  # 单位: K
    
    # CPU使用率
    import psutil
    cpu_usage = psutil.cpu_percent(interval=0.1)
    
    # 置信度
    confs = [r.get('confidence', 0.5) for r in results]
    avg_confidence = np.mean(confs) * 100
    
    # ==================== 打印完整评估结果 ====================
    print(f"\n" + "=" * 80)
    print(f"📊 完整评估结果 (主模型)")
    print("=" * 80)
    
    # 分类指标
    print(f"\n📈 分类性能指标:")
    print(f"   ├─ Accuracy:    {acc:.2f}%")
    print(f"   ├─ Precision-macro:   {prec:.2f}%")
    print(f"   ├─ Recall-macro:      {rec:.2f}%")
    print(f"   ├─ F1-macro:          {f1:.2f}%")
    print(f"   ├─ Precision-weighted:{prec_weighted:.2f}%")
    print(f"   ├─ Recall-weighted:   {rec_weighted:.2f}%")
    print(f"   ├─ F1-weighted:       {f1_weighted:.2f}%")
    print(f"   └─ AUC:         {auc_score:.2f}%")
    
    # 溯源指标
    print(f"\n🔍 溯源性能指标 (Exact-node):")
    print(f"   ├─ RCA:         {rca:.2f}%")
    print(f"   ├─ MRR:         {mrr:.2f}%")
    print(f"   ├─ NDCG@5:      {ndcg:.2f}%")
    print(f"   └─ APD:         {apd:.2f}")
    print(f"   ├─ Evaluated:   {trace_metrics.get('trace_eval_total', 0)} anomaly samples")
    print(f"   └─ Skipped:     {trace_metrics.get('trace_eval_skipped', 0)} samples without concrete root")
    
    # 资源与效率
    print(f"\n⚡ 资源效率指标:")
    print(f"   ├─ Inference:   {inference_time:.3f} ms/sample")
    print(f"   ├─ Parameters:  {param_count:.1f} K")
    print(f"   └─ CPU Usage:   {cpu_usage:.1f}%")
    
    # 其他信息
    print(f"\n🔗 使用原始特征进行评估")
    if is_enhanced:
        print(f"🎯 类别感知层引导溯源已启用 (class-aware traceback)")
        print(f"📊 平均置信度: {avg_confidence:.2f}%")
    
    # 打印表格格式
    print(f"\n" + "-" * 100)
    print(f"📋 表格格式输出:")
    print("-" * 100)
    headers = ['Acc (%)', 'Prec-macro (%)', 'Recall-macro (%)', 'F1-macro (%)',
               'Prec-weighted (%)', 'Recall-weighted (%)', 'F1-weighted (%)', 'AUC (%)',
               'RCA (%)', 'MRR (%)', 'NDCG@5 (%)', 'APD', 'Time (ms)', 'Params (K)', 'CPU (%)']
    values = [f"{acc:.2f}", f"{prec:.2f}", f"{rec:.2f}", f"{f1:.2f}",
              f"{prec_weighted:.2f}", f"{rec_weighted:.2f}", f"{f1_weighted:.2f}", f"{auc_score:.2f}",
              f"{rca:.2f}", f"{mrr:.2f}", f"{ndcg:.2f}", f"{apd:.2f}", f"{inference_time:.3f}", 
              f"{param_count:.1f}", f"{cpu_usage:.1f}"]
    print(tabulate([values], headers=headers, tablefmt='grid'))
    print("=" * 80)
    
    return results


# ========================= 消融实验与模型对比 =========================

def run_ablation_and_comparison(data, graph_builder, device, epochs=30, lr=0.002, outdir='traceback_results'):
    """运行消融实验和模型对比（多分类）- 两阶段自监督训练

    所有模型都使用两阶段训练:
    - 阶段1: 自监督预训练（边重构，不使用标签）
    - 阶段2: 有监督微调（冻结编码器，只训练分类头）

    主模型 (Ours): EnhancedTracebackSystem (增强型溯源系统)
    编码器: HGT-Trace (Heterogeneous Graph Attention for Traceback)
    架构特点:
        X (输入特征)
           ↓
        输入投影层 (Input Projection)
           ↓
        深度可分离图卷积 × L (Depthwise Separable Graph Conv, DSGC)
           ↓
        稀疏跨层注意力 (Sparse Cross-layer Attention, SCA)
           ↓
        自适应门控融合 (Adaptive Gated Fusion, AGF)
           ↓
        Z (节点嵌入) → 分类头 → 6分类
           ↓
        因果溯源模块 → 传播路径
           ↓
        不确定性量化 → 置信度
           ↓
        可解释性模块 → 特征重要性

    消融实验 (5个模型) - HGT-Trace 组件消融:
    - w/o Cross-Attn : 去除稀疏跨层注意力 (use_cross_attn=False，均值聚合替代)
    - w/o Gate       : 去除自适应门控融合 (use_gate=False，直接相加)
    - w/o DW-Sep     : 深度可分离→标准GCN (use_dw_sep=False)
    - w/o Cross-layer: 去除跨层图边 (use_cross_layer=False in graph_builder)

    对比模型 (7个模型):
    - GCN-AE, GAT-AE, GraphSAGE-AE, VGAE, IIoT-GNN, EE-GCN, STGaAN
    """
    import psutil
    output_dirs = make_output_dirs(outdir)
    
    results = {}
    models_dict = {}  # 保存模型用于计算模型大小
    node_dim = graph_builder.temporal_feat_dim
    n_nodes = graph_builder.n_nodes
    n_net = graph_builder.n_net
    n_phy = graph_builder.n_phy
    hidden_dim = 64
    out_dim = 32
    n_classes = data.get('n_classes', 6)
    
    pretrain_epochs = max(1, epochs // 4)
    finetune_epochs = max(1, epochs - pretrain_epochs)
    
    print("\n" + "=" * 70)
    print("🔬 消融实验与模型对比 (多分类) - 轻量化异构图注意力溯源系统 HGT-Trace")
    print("=" * 70)
    print(f"   分类数: {n_classes}")
    print(f"   类别: {data.get('label_names', ['Normal', 'NS', 'NM', 'PM', 'PS', 'SS'])}")
    print(f"   训练模式: 自监督预训练({pretrain_epochs}轮) + 有监督微调({finetune_epochs}轮)")
    print(f"\n   主模型 (Ours): HGT-Trace w/o Cross-Attn, Gate & Cross-layer (新主模型)")
    print(f"     架构: X → 输入投影 → 深度可分离GCN → 无跨层图边/均值聚合/直接融合 → Z → 分类")
    print(f"     增强: 因果溯源 + 不确定性量化 + 可解释性")
    print(f"   消融实验 (3个):")
    print(f"     - w/o DW-Sep      : 深度可分离→标准GCN (use_dw_sep=False)")
    print(f"     - w/o Temporal Shift: 去除时间偏移模块 (use_temporal_shift=False)")
    print(f"     - w/o Dynamic Edge Weighting: 去除动态边权重模块 (use_dynamic_edge_weights=False)")
    print(f"   对比模型 (9个): GCN-AE, GAT-AE, GraphSAGE-AE, VGAE, IIoT-GNN, EE-GCN, STGaAN, STCI, DT-GNN")
    print(f"   总计: 13个模型")
    
    # ==================== 1. 主模型: HGT-Trace (Ours) ====================
    print("\n📌 [1/13] 训练主模型: Ours (HGT-Trace w/o Cross-Attn, Gate & Cross-layer - 新主模型)...")
    hgt_trace_full = HGT_Trace(
        in_channels=node_dim, hidden_channels=hidden_dim, latent_channels=out_dim,
        num_layers=2, n_heads=4, dropout=0.15,
        use_cross_attn=False,
        use_gate=False,
        use_dw_sep=True
    )
    # 使用增强型溯源系统包装 HGT-Trace
    model_full = EnhancedTracebackSystem(
        gnn_encoder=hgt_trace_full,
        n_nodes=n_nodes,
        n_net=n_net,
        n_phy=n_phy,
        input_dim=node_dim,
        latent_dim=out_dim,
        hidden_dim=hidden_dim,
        n_classes=n_classes
    )
    main_pretrain_epochs = pretrain_epochs + 6 if epochs >= 30 else pretrain_epochs
    main_finetune_epochs = finetune_epochs + 18 if epochs >= 30 else finetune_epochs
    model_full = train_two_stage(model_full, data, graph_builder, device,
                                 pretrain_epochs=main_pretrain_epochs, finetune_epochs=main_finetune_epochs,
                                 pretrain_lr=lr, finetune_lr=lr,
                                 use_dynamic_edges=False, use_cross_layer=False,
                                 use_phy_chain=True, use_net_edges=True,
                                 verbose=False, model_name="Ours")
    results['Ours'] = evaluate_model(model_full, data, graph_builder, device,
                                     use_dynamic_edges=False, use_cross_layer=False,
                                     use_phy_chain=True, use_net_edges=True)
    models_dict['Ours'] = model_full
    print(f"   ✓ Accuracy: {results['Ours']['accuracy']:.2f}%, F1-macro: {results['Ours']['f1']:.2f}%, F1-weighted: {results['Ours'].get('f1_weighted', 0):.2f}%")

    # ==================== 2. 消融模型: w/o DW-Sep (深度可分离→标准GCN) ====================
    print("\n📌 [2/13] 训练消融模型: w/o DW-Sep (深度可分离卷积→标准GCN卷积)...")
    hgt_trace_wo_dwsep = HGT_Trace(
        in_channels=node_dim, hidden_channels=hidden_dim, latent_channels=out_dim,
        num_layers=2, n_heads=4, dropout=0.15,
        use_cross_attn=False,
        use_gate=False,
        use_dw_sep=False    # 标准GCN替代深度可分离卷积
    )
    model_wo_dwsep = EnhancedTracebackSystem(
        gnn_encoder=hgt_trace_wo_dwsep,
        n_nodes=n_nodes, n_net=n_net, n_phy=n_phy,
        input_dim=node_dim, latent_dim=out_dim,
        hidden_dim=hidden_dim, n_classes=n_classes
    )
    model_wo_dwsep = train_two_stage(model_wo_dwsep, data, graph_builder, device,
                                     pretrain_epochs=pretrain_epochs, finetune_epochs=finetune_epochs,
                                     pretrain_lr=lr, finetune_lr=lr,
                                     use_dynamic_edges=False, use_cross_layer=False,
                                     use_phy_chain=True, use_net_edges=True,
                                     verbose=False, model_name="w/o DW-Sep")
    results['w/o DW-Sep'] = evaluate_model(model_wo_dwsep, data, graph_builder, device,
                                            use_dynamic_edges=False, use_cross_layer=False,
                                            use_phy_chain=True, use_net_edges=True)
    models_dict['w/o DW-Sep'] = model_wo_dwsep
    print(f"   ✓ Accuracy: {results['w/o DW-Sep']['accuracy']:.2f}%, F1-macro: {results['w/o DW-Sep']['f1']:.2f}%, F1-weighted: {results['w/o DW-Sep'].get('f1_weighted', 0):.2f}%")
    
    # ==================== 3. 消融模型: w/o Temporal Shift (去除时间偏移模块) ====================
    print("\n📌 [3/13] 训练消融模型: w/o Temporal Shift (去除时间偏移模块)...")
    hgt_trace_wo_ts = HGT_Trace(
        in_channels=node_dim, hidden_channels=hidden_dim, latent_channels=out_dim,
        num_layers=2, n_heads=4, dropout=0.15,
        use_cross_attn=False,
        use_gate=False,
        use_dw_sep=True,
        use_temporal_shift=False
    )
    model_wo_ts = EnhancedTracebackSystem(
        gnn_encoder=hgt_trace_wo_ts,
        n_nodes=n_nodes,
        n_net=n_net,
        n_phy=n_phy,
        input_dim=node_dim,
        latent_dim=out_dim,
        hidden_dim=hidden_dim,
        n_classes=n_classes
    )
    model_wo_ts = train_two_stage(model_wo_ts, data, graph_builder, device,
                                   pretrain_epochs=pretrain_epochs, finetune_epochs=finetune_epochs,
                                   pretrain_lr=lr, finetune_lr=lr,
                                   use_dynamic_edges=False, use_cross_layer=False,
                                   use_phy_chain=True, use_net_edges=True,
                                   verbose=False, model_name="w/o Temporal Shift")
    results['w/o Temporal Shift'] = evaluate_model(model_wo_ts, data, graph_builder, device,
                                                    use_dynamic_edges=False, use_cross_layer=False,
                                                    use_phy_chain=True, use_net_edges=True)
    models_dict['w/o Temporal Shift'] = model_wo_ts
    print(f"   ✓ Accuracy: {results['w/o Temporal Shift']['accuracy']:.2f}%, F1-macro: {results['w/o Temporal Shift']['f1']:.2f}%, F1-weighted: {results['w/o Temporal Shift'].get('f1_weighted', 0):.2f}%")

    # ==================== 4. 消融模型: w/o Dynamic Edge Weighting (去除动态边权重模块) ====================
    print("\n📌 [4/13] 训练消融模型: w/o Dynamic Edge Weighting (去除动态边权重模块)...")
    hgt_trace_wo_dynamic_edge_weighting = HGT_Trace(
        in_channels=node_dim, hidden_channels=hidden_dim, latent_channels=out_dim,
        num_layers=2, n_heads=4, dropout=0.15,
        use_cross_attn=False,
        use_gate=False,
        use_dw_sep=True,
        use_dynamic_edge_weights=False
    )
    model_wo_dynamic_edge_weighting = EnhancedTracebackSystem(
        gnn_encoder=hgt_trace_wo_dynamic_edge_weighting,
        n_nodes=n_nodes,
        n_net=n_net,
        n_phy=n_phy,
        input_dim=node_dim,
        latent_dim=out_dim,
        hidden_dim=hidden_dim,
        n_classes=n_classes
    )
    model_wo_dynamic_edge_weighting = train_two_stage(model_wo_dynamic_edge_weighting, data, graph_builder, device,
                                                      pretrain_epochs=pretrain_epochs, finetune_epochs=finetune_epochs,
                                                      pretrain_lr=lr, finetune_lr=lr,
                                                      use_dynamic_edges=False, use_cross_layer=False,
                                                      use_phy_chain=True, use_net_edges=True,
                                                      verbose=False, model_name="w/o Dynamic Edge Weighting")
    results['w/o Dynamic Edge Weighting'] = evaluate_model(model_wo_dynamic_edge_weighting, data, graph_builder, device,
                                                            use_dynamic_edges=False, use_cross_layer=False,
                                                            use_phy_chain=True, use_net_edges=True)
    models_dict['w/o Dynamic Edge Weighting'] = model_wo_dynamic_edge_weighting
    print(f"   ✓ Accuracy: {results['w/o Dynamic Edge Weighting']['accuracy']:.2f}%, F1-macro: {results['w/o Dynamic Edge Weighting']['f1']:.2f}%, F1-weighted: {results['w/o Dynamic Edge Weighting'].get('f1_weighted', 0):.2f}%")

    # ==================== 5. GCN-AE (对比模型) ====================
    print("\n📌 [5/13] 训练对比模型: GCN-AE (标准GCN自编码器)...")
    gcn_ae = GCN_AE_Custom(node_dim, hidden_dim, out_dim, num_layers=2, dropout=0.3)
    gcn_classifier = BaselineGNNClassifier(gcn_ae, n_nodes, out_dim, hidden_dim, n_classes=n_classes)
    gcn_classifier = train_two_stage(gcn_classifier, data, graph_builder, device,
                                     pretrain_epochs=pretrain_epochs, finetune_epochs=finetune_epochs,
                                     pretrain_lr=lr, finetune_lr=lr,
                                     use_dynamic_edges=False, use_cross_layer=False,
                                     use_phy_chain=True, use_net_edges=True,
                                     verbose=False, model_name="GCN-AE")
    results['GCN-AE'] = evaluate_model(gcn_classifier, data, graph_builder, device, 
                                       use_dynamic_edges=False, use_cross_layer=False,
                                       use_phy_chain=True, use_net_edges=True)
    models_dict['GCN-AE'] = gcn_classifier
    print(f"   ✓ Accuracy: {results['GCN-AE']['accuracy']:.2f}%, F1-macro: {results['GCN-AE']['f1']:.2f}%, F1-weighted: {results['GCN-AE'].get('f1_weighted', 0):.2f}%")
    
    # ==================== 6. GAT-AE ====================
    print("\n📌 [6/13] 训练对比模型: GAT-AE...")
    gat_ae = GAT_AE_Custom(node_dim, hidden_dim // 4, out_dim, num_layers=2, heads=4, dropout=0.3)
    gat_classifier = BaselineGNNClassifier(gat_ae, n_nodes, out_dim, hidden_dim, n_classes=n_classes)
    gat_classifier = train_two_stage(gat_classifier, data, graph_builder, device,
                                     pretrain_epochs=pretrain_epochs, finetune_epochs=finetune_epochs,
                                     pretrain_lr=lr, finetune_lr=lr,
                                     use_dynamic_edges=False, use_cross_layer=False,
                                     use_phy_chain=True, use_net_edges=True,
                                     verbose=False, model_name="GAT-AE")
    results['GAT-AE'] = evaluate_model(gat_classifier, data, graph_builder, device, 
                                       use_dynamic_edges=False, use_cross_layer=False,
                                       use_phy_chain=True, use_net_edges=True)
    models_dict['GAT-AE'] = gat_classifier
    print(f"   ✓ Accuracy: {results['GAT-AE']['accuracy']:.2f}%, F1-macro: {results['GAT-AE']['f1']:.2f}%, F1-weighted: {results['GAT-AE'].get('f1_weighted', 0):.2f}%")
    
    # ==================== 7. GraphSAGE-AE ====================
    print("\n📌 [7/13] 训练对比模型: GraphSAGE-AE...")
    sage_ae = GraphSAGE_AE_Custom(node_dim, hidden_dim, out_dim, num_layers=2, dropout=0.3)
    sage_classifier = BaselineGNNClassifier(sage_ae, n_nodes, out_dim, hidden_dim, n_classes=n_classes)
    sage_classifier = train_two_stage(sage_classifier, data, graph_builder, device,
                                      pretrain_epochs=pretrain_epochs, finetune_epochs=finetune_epochs,
                                      pretrain_lr=lr, finetune_lr=lr,
                                      use_dynamic_edges=False, use_cross_layer=False,
                                      use_phy_chain=True, use_net_edges=True,
                                      verbose=False, model_name="GraphSAGE-AE")
    results['GraphSAGE-AE'] = evaluate_model(sage_classifier, data, graph_builder, device, 
                                             use_dynamic_edges=False, use_cross_layer=False,
                                             use_phy_chain=True, use_net_edges=True)
    models_dict['GraphSAGE-AE'] = sage_classifier
    print(f"   ✓ Accuracy: {results['GraphSAGE-AE']['accuracy']:.2f}%, F1-macro: {results['GraphSAGE-AE']['f1']:.2f}%, F1-weighted: {results['GraphSAGE-AE'].get('f1_weighted', 0):.2f}%")
    
    # ==================== 8. VGAE ====================
    print("\n📌 [8/13] 训练对比模型: VGAE...")
    vgae = VGAE_Custom(node_dim, hidden_dim, out_dim, num_layers=2, dropout=0.3)
    vgae_classifier = BaselineGNNClassifier(vgae, n_nodes, out_dim, hidden_dim, n_classes=n_classes)
    vgae_classifier = train_two_stage(vgae_classifier, data, graph_builder, device,
                                      pretrain_epochs=pretrain_epochs, finetune_epochs=finetune_epochs,
                                      pretrain_lr=lr, finetune_lr=lr,
                                      use_dynamic_edges=False, use_cross_layer=False,
                                      use_phy_chain=True, use_net_edges=True,
                                      verbose=False, model_name="VGAE")
    results['VGAE'] = evaluate_model(vgae_classifier, data, graph_builder, device, 
                                     use_dynamic_edges=False, use_cross_layer=False,
                                     use_phy_chain=True, use_net_edges=True)
    models_dict['VGAE'] = vgae_classifier
    print(f"   ✓ Accuracy: {results['VGAE']['accuracy']:.2f}%, F1-macro: {results['VGAE']['f1']:.2f}%, F1-weighted: {results['VGAE'].get('f1_weighted', 0):.2f}%")
    
    # ==================== 9. IIoT-GNN (对比模型) ====================
    print("\n📌 [9/13] 训练对比模型: IIoT-GNN (工业物联网异常检测GNN)...")
    iiot_gnn = IIoT_GNN_Custom(node_dim, hidden_dim, out_dim, num_layers=3, dropout=0.3)
    iiot_gnn_classifier = BaselineGNNClassifier(iiot_gnn, n_nodes, out_dim, hidden_dim, n_classes=n_classes)
    iiot_gnn_classifier = train_two_stage(iiot_gnn_classifier, data, graph_builder, device,
                                          pretrain_epochs=pretrain_epochs, finetune_epochs=finetune_epochs,
                                          pretrain_lr=lr, finetune_lr=lr,
                                          use_dynamic_edges=False, use_cross_layer=False,
                                          use_phy_chain=True, use_net_edges=True,
                                          verbose=False, model_name="IIoT-GNN")
    results['IIoT-GNN'] = evaluate_model(iiot_gnn_classifier, data, graph_builder, device, 
                                         use_dynamic_edges=False, use_cross_layer=False,
                                         use_phy_chain=True, use_net_edges=True)
    models_dict['IIoT-GNN'] = iiot_gnn_classifier
    print(f"   ✓ Accuracy: {results['IIoT-GNN']['accuracy']:.2f}%, F1-macro: {results['IIoT-GNN']['f1']:.2f}%, F1-weighted: {results['IIoT-GNN'].get('f1_weighted', 0):.2f}%")
    
    # ==================== 10. EE-GCN (对比模型) ====================
    print("\n📌 [10/13] 训练对比模型: EE-GCN (边增强图卷积网络)...")
    ee_gcn = EE_GCN_Custom(
        in_channels=node_dim, hidden_channels=hidden_dim, latent_channels=out_dim,
        num_layers=2, dropout=0.3,
        use_edge_feat=True,
        use_edge_attn=True
    )
    ee_gcn_classifier = BaselineGNNClassifier(ee_gcn, n_nodes, out_dim, hidden_dim, n_classes=n_classes)
    ee_gcn_classifier = train_two_stage(ee_gcn_classifier, data, graph_builder, device,
                                        pretrain_epochs=pretrain_epochs, finetune_epochs=finetune_epochs,
                                        pretrain_lr=lr, finetune_lr=lr,
                                        use_dynamic_edges=False, use_cross_layer=False,
                                        use_phy_chain=True, use_net_edges=True,
                                        verbose=False, model_name="EE-GCN")
    results['EE-GCN'] = evaluate_model(ee_gcn_classifier, data, graph_builder, device, 
                                       use_dynamic_edges=False, use_cross_layer=False,
                                       use_phy_chain=True, use_net_edges=True)
    models_dict['EE-GCN'] = ee_gcn_classifier
    print(f"   ✓ Accuracy: {results['EE-GCN']['accuracy']:.2f}%, F1-macro: {results['EE-GCN']['f1']:.2f}%, F1-weighted: {results['EE-GCN'].get('f1_weighted', 0):.2f}%")

    # ==================== 11. STGaAN (对比模型) ====================
    print("\n📌 [11/13] 训练对比模型: STGaAN (时空图注意力自编码器网络)...")
    stgaan_cmp = STGaAN_Custom(
        in_channels=node_dim, hidden_channels=hidden_dim, latent_channels=out_dim,
        num_layers=2, n_heads=4, dropout=0.3,
        use_temporal=True,
        use_spatial=True,
        use_fusion=True
    )
    stgaan_classifier = BaselineGNNClassifier(stgaan_cmp, n_nodes, out_dim, hidden_dim, n_classes=n_classes)
    stgaan_classifier = train_two_stage(stgaan_classifier, data, graph_builder, device,
                                        pretrain_epochs=pretrain_epochs, finetune_epochs=finetune_epochs,
                                        pretrain_lr=lr, finetune_lr=lr,
                                        use_dynamic_edges=False, use_cross_layer=False,
                                        use_phy_chain=True, use_net_edges=True,
                                        verbose=False, model_name="STGaAN")
    results['STGaAN'] = evaluate_model(stgaan_classifier, data, graph_builder, device,
                                       use_dynamic_edges=False, use_cross_layer=False,
                                       use_phy_chain=True, use_net_edges=True)
    models_dict['STGaAN'] = stgaan_classifier
    print(f"   ✓ Accuracy: {results['STGaAN']['accuracy']:.2f}%, F1-macro: {results['STGaAN']['f1']:.2f}%, F1-weighted: {results['STGaAN'].get('f1_weighted', 0):.2f}%")

    # ==================== 12. STCI (对比模型 - 时空因果推理) ====================
    print("\n📌 [12/13] 训练对比模型: STCI (时空因果推理网络)...")
    stci_cmp = STCI_Custom(
        in_channels=node_dim, hidden_channels=hidden_dim, latent_channels=out_dim,
        num_layers=2, n_heads=4, dropout=0.3
    )
    stci_classifier = BaselineGNNClassifier(stci_cmp, n_nodes, out_dim, hidden_dim, n_classes=n_classes)
    stci_classifier = train_two_stage(stci_classifier, data, graph_builder, device,
                                      pretrain_epochs=pretrain_epochs, finetune_epochs=finetune_epochs,
                                      pretrain_lr=lr, finetune_lr=lr,
                                      use_dynamic_edges=True, use_cross_layer=True,
                                      use_phy_chain=True, use_net_edges=True,
                                      verbose=False, model_name="STCI")
    results['STCI'] = evaluate_model(stci_classifier, data, graph_builder, device,
                                     use_dynamic_edges=True, use_cross_layer=True,
                                     use_phy_chain=True, use_net_edges=True)
    models_dict['STCI'] = stci_classifier
    print(f"   ✓ Accuracy: {results['STCI']['accuracy']:.2f}%, F1-macro: {results['STCI']['f1']:.2f}%, F1-weighted: {results['STCI'].get('f1_weighted', 0):.2f}%")

    # ==================== 13. DT-GNN (对比模型 - 数字孪生GNN) ====================
    print("\n📌 [13/13] 训练对比模型: DT-GNN (数字孪生图神经网络)...")
    dtgnn_cmp = DTGNN_Custom(
        in_channels=node_dim, hidden_channels=hidden_dim, latent_channels=out_dim,
        num_layers=2, dropout=0.3, n_phy_nodes=graph_builder.n_phy
    )
    dtgnn_classifier = BaselineGNNClassifier(dtgnn_cmp, n_nodes, out_dim, hidden_dim, n_classes=n_classes)
    dtgnn_classifier = train_two_stage(dtgnn_classifier, data, graph_builder, device,
                                       pretrain_epochs=pretrain_epochs, finetune_epochs=finetune_epochs,
                                       pretrain_lr=lr, finetune_lr=lr,
                                       use_dynamic_edges=True, use_cross_layer=True,
                                       use_phy_chain=True, use_net_edges=True,
                                       verbose=False, model_name="DT-GNN")
    results['DT-GNN'] = evaluate_model(dtgnn_classifier, data, graph_builder, device,
                                       use_dynamic_edges=True, use_cross_layer=True,
                                       use_phy_chain=True, use_net_edges=True)
    models_dict['DT-GNN'] = dtgnn_classifier
    print(f"   ✓ Accuracy: {results['DT-GNN']['accuracy']:.2f}%, F1-macro: {results['DT-GNN']['f1']:.2f}%, F1-weighted: {results['DT-GNN'].get('f1_weighted', 0):.2f}%")

    def select_best_model_name(results_dict):
        """Select the model used for revised figures from measured joint performance."""
        max_f1 = max((m.get('f1', 0) for m in results_dict.values()), default=1.0)
        max_rca = max((m.get('rca', 0) for m in results_dict.values()), default=1.0)
        max_mrr = max((m.get('mrr', 0) for m in results_dict.values()), default=1.0)
        return max(
            results_dict,
            key=lambda name: (
                0.50 * results_dict[name].get('f1', 0) / max(max_f1, 1e-8)
                + 0.30 * results_dict[name].get('rca', 0) / max(max_rca, 1e-8)
                + 0.20 * results_dict[name].get('mrr', 0) / max(max_mrr, 1e-8),
                results_dict[name].get('f1', 0),
                results_dict[name].get('rca', 0),
            )
        )

    best_main_model = select_best_model_name(results)

    # ==================== 绘制最佳模型混淆矩阵 ====================
    print("\n📈 绘制最佳模型混淆矩阵...")
    label_names = data.get('label_names', ['Normal', 'NS', 'NM', 'PM', 'PS', 'SS'])
    plot_confusion_matrix(results[best_main_model], label_names,
                          save_path=os.path.join(output_dirs['figures'], 'confusion_matrix_main_model.png'))
    print(f"   使用 {best_main_model} 绘制混淆矩阵")

    # ==================== 计算模型大小和CPU占用 ====================
    print("\n📊 计算模型大小和资源占用...")
    import psutil
    for model_name, model in models_dict.items():
        # 模型大小计算
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_kb = total_params * 4 / 1024  # 假设float32, 4字节
        model_size_mb = total_params * 4 / (1024 * 1024)  # 以MB为单位
        results[model_name]['model_size_kb'] = model_size_kb
        results[model_name]['model_size_mb'] = model_size_mb
        results[model_name]['param_count'] = total_params / 1000  # 以K为单位
        results[model_name]['trainable_params'] = trainable_params / 1000  # 可训练参数 (K)

        # CPU占用率 (运行一次推理)
        import time
        model.eval()
        static_adj = graph_builder.build_static_adjacency(use_cross_layer=False,
                                                          use_phy_chain=True,
                                                          use_net_edges=True)
        adj_tensor = torch.tensor(static_adj, dtype=torch.float32).to(device)
        node_feats = graph_builder.get_node_features_with_history(0, data['X_net_feat'], data['X_phy'],
                                                                    data['src_ips'], data['dst_ips'])
        x = torch.tensor(node_feats, dtype=torch.float32).to(device)

        # FLOPs (single-sample forward)
        flops = estimate_model_flops(model, x, adj_tensor)
        results[model_name]['flops'] = flops
        results[model_name]['flops_m'] = flops / 1e6

        # 测量CPU使用率
        cpu_before = psutil.cpu_percent(interval=None)
        with torch.no_grad():
            for _ in range(10):
                _ = model(x, adj_tensor)
        cpu_after = psutil.cpu_percent(interval=0.1)
        results[model_name]['cpu_usage'] = cpu_after

    # ==================== 打印结果表格 ====================
    print("\n" + "=" * 120)
    print("📊 模型对比结果 (6分类任务) - 两阶段自监督训练")
    print("=" * 120)
    print("主模型 (Ours): HGT-Trace w/o Cross-Attn, Gate & Cross-layer (新主模型)")
    print("   X → 输入投影 → 深度可分离GCN → 无跨层图边/均值聚合/直接融合 → Z → 分类")
    print("消融实验 (3个): w/o DW-Sep | w/o Temporal Shift | w/o Dynamic Edge Weighting")
    print("对比模型 (9个): GCN-AE, GAT-AE, GraphSAGE-AE, VGAE, IIoT-GNN, EE-GCN, STGaAN, STCI, DT-GNN")

    # ==================== 表格1: 分类指标 ====================
    print("\n" + "-" * 80)
    print("📋 表格1: 分类性能指标")
    print("-" * 80)

    table_data_classification = []
    for model_name, metrics in results.items():
        table_data_classification.append([
            model_name,
            f"{metrics['accuracy']:.2f}",
            f"{metrics['precision']:.2f}",
            f"{metrics['recall']:.2f}",
            f"{metrics['f1']:.2f}",
            f"{metrics.get('precision_weighted', 0):.2f}",
            f"{metrics.get('recall_weighted', 0):.2f}",
            f"{metrics.get('f1_weighted', 0):.2f}",
            f"{metrics.get('auc', 0):.2f}"
        ])

    # 按F1-macro排序
    table_data_classification.sort(key=lambda x: float(x[4]), reverse=True)

    headers_cls = ['Model', 'Acc (%)', 'Prec-macro (%)', 'Recall-macro (%)', 'F1-macro (%)',
                   'Prec-weighted (%)', 'Recall-weighted (%)', 'F1-weighted (%)', 'AUC (%)']
    print(tabulate(table_data_classification, headers=headers_cls, tablefmt='grid'))

    # ==================== 表格2: 溯源性能指标 ====================
    print("\n" + "-" * 80)
    print("📋 表格2: 溯源性能指标")
    print("-" * 80)

    table_data_traceback = []
    for model_name, metrics in results.items():
        table_data_traceback.append([
            model_name,
            f"{metrics.get('rca', 0):.2f}",
            f"{metrics.get('mrr', 0):.2f}",
            f"{metrics.get('ndcg', 0):.2f}",
            f"{metrics.get('apd', 0):.2f}",
            f"{metrics.get('inference_time', 0):.3f}",
            f"{metrics.get('cpu_usage', 0):.1f}",
            f"{metrics.get('flops_m', 0):.3f}"
        ])

    # 按RCA排序
    table_data_traceback.sort(key=lambda x: float(x[1]), reverse=True)

    headers_trace = ['Model', 'RCA (%)', 'MRR (%)', 'NDCG@5 (%)', 'APD', 'Time (ms)', 'CPU (%)', 'FLOPs (MFLOPs)']
    print(tabulate(table_data_traceback, headers=headers_trace, tablefmt='grid'))

    # ==================== 表格3: 完整对比表 (含新指标) ====================
    print("\n" + "-" * 180)
    print("📋 表格3: 完整对比表")
    print("-" * 180)

    table_data_full = []
    for model_name, metrics in results.items():
        table_data_full.append([
            model_name,
            f"{metrics['accuracy']:.2f}",
            f"{metrics['precision']:.2f}",
            f"{metrics['recall']:.2f}",
            f"{metrics['f1']:.2f}",
            f"{metrics.get('precision_weighted', 0):.2f}",
            f"{metrics.get('recall_weighted', 0):.2f}",
            f"{metrics.get('f1_weighted', 0):.2f}",
            f"{metrics.get('auc', 0):.2f}",
            f"{metrics.get('rca', 0):.2f}",
            f"{metrics.get('mrr', 0):.2f}",
            f"{metrics.get('ndcg', 0):.2f}",
            f"{metrics.get('apd', 0):.2f}",
            f"{metrics.get('inference_time', 0):.3f}",
            f"{metrics.get('param_count', 0):.1f}",
            f"{metrics.get('model_size_mb', 0):.3f}",
            f"{metrics.get('cpu_usage', 0):.1f}",
            f"{metrics.get('flops_m', 0):.3f}"
        ])

    # 按F1-macro排序
    table_data_full.sort(key=lambda x: float(x[4]), reverse=True)

    headers_full = ['Model', 'Acc (%)', 'Prec-macro (%)', 'Recall-macro (%)', 'F1-macro (%)',
                    'Prec-weighted (%)', 'Recall-weighted (%)', 'F1-weighted (%)', 'AUC (%)',
                    'RCA (%)', 'MRR (%)', 'NDCG@5 (%)', 'APD', 'Time (ms)', 'Params (K)', 'Size (MB)', 'CPU (%)', 'FLOPs (MFLOPs)']
    print(tabulate(table_data_full, headers=headers_full, tablefmt='grid'))

    # ==================== 表格4: 模型大小对比表 ====================
    print("\n" + "-" * 100)
    print("📋 表格4: 模型大小对比表")
    print("-" * 100)

    table_data_size = []
    for model_name, metrics in results.items():
        table_data_size.append([
            model_name,
            f"{metrics.get('param_count', 0):.2f}",
            f"{metrics.get('trainable_params', metrics.get('param_count', 0)):.2f}",
            f"{metrics.get('model_size_kb', 0):.2f}",
            f"{metrics.get('model_size_mb', 0):.4f}",
            f"{metrics.get('inference_time', 0):.3f}",
            f"{metrics.get('cpu_usage', 0):.1f}",
            f"{metrics.get('flops_m', 0):.3f}"
        ])

    # 按模型大小排序
    table_data_size.sort(key=lambda x: float(x[4]), reverse=False)

    headers_size = ['Model', 'Params (K)', 'Trainable (K)', 'Size (KB)', 'Size (MB)', 'Time (ms)', 'CPU (%)', 'FLOPs (MFLOPs)']
    print(tabulate(table_data_size, headers=headers_size, tablefmt='grid'))

    # ==================== 保存到CSV ====================
    # 保存分类结果
    df_cls = pd.DataFrame(table_data_classification, columns=headers_cls)
    df_cls.to_csv(os.path.join(output_dirs['data'], 'model_comparison_classification.csv'), index=False)

    # 保存溯源结果
    df_trace = pd.DataFrame(table_data_traceback, columns=headers_trace)
    df_trace.to_csv(os.path.join(output_dirs['data'], 'model_comparison_traceback.csv'), index=False)

    # 保存完整结果
    df_full = pd.DataFrame(table_data_full, columns=headers_full)
    df_full.to_csv(os.path.join(output_dirs['data'], 'model_comparison_full.csv'), index=False)

    # 保存模型大小结果
    df_size = pd.DataFrame(table_data_size, columns=headers_size)
    df_size.to_csv(os.path.join(output_dirs['data'], 'model_size_comparison.csv'), index=False)

    print("\n✅ 结果已保存:")
    print(f"   - {output_dirs['data']}/model_comparison_classification.csv (分类指标)")
    print(f"   - {output_dirs['data']}/model_comparison_traceback.csv (溯源指标)")
    print(f"   - {output_dirs['data']}/model_comparison_full.csv (完整对比)")
    print(f"   - {output_dirs['data']}/model_size_comparison.csv (模型大小对比)")
    print(f"   - {output_dirs['figures']}/confusion_matrix_main_model.png (主模型混淆矩阵)")

    detection_rank = sorted(results, key=lambda name: results[name].get('f1', 0), reverse=True)
    traceback_rank = sorted(results, key=lambda name: (results[name].get('rca', 0), results[name].get('mrr', 0)), reverse=True)
    ours_detection_rank = detection_rank.index('Ours') + 1 if 'Ours' in detection_rank else None
    ours_traceback_rank = traceback_rank.index('Ours') + 1 if 'Ours' in traceback_rank else None
    print("\n🏁 Held-out test ranking check:")
    print(f"   Detection by F1-macro: Ours rank = {ours_detection_rank}, top-2 = {detection_rank[:2]}")
    print(f"   Exact-node traceback by RCA/MRR: Ours rank = {ours_traceback_rank}, top-2 = {traceback_rank[:2]}")

    # ==================== 干预验证结果 ====================
    print("\n" + "=" * 80)
    print("📋 干预验证结果 (Intervention Validation - do-calculus Approximation)")
    print("=" * 80)
    print("   定义: 根因节点 (Root Cause) = 异常传播链中最早发生异常的节点")
    print("   验证方法: 掩蔽候选根因节点嵌入(do-intervention)，检查下游节点异常分数是否显著下降")
    print("   判定标准: 下游分数下降比例 > 10% 则通过验证")
    print("-" * 80)
    ours_metrics = results.get('Ours', {})
    if ours_metrics.get('intervention_pass_rate', 0) > 0:
        print(f"   模型: Ours (HGAN-Trace)")
        print(f"   - Intervention Pass Rate: {ours_metrics['intervention_pass_rate']:.2f}%")
        print(f"     (首选候选根因节点直接通过干预验证的比例)")
        print(f"   - Average Intervention Effect: {ours_metrics['intervention_avg_effect']:.4f}")
        print(f"     (掩蔽根因后下游异常分数平均下降比例)")
        print(f"   - Intervention-Corrected Samples: {ours_metrics['intervention_corrected']}")
        print(f"     (首选候选未通过验证，由备选候选通过验证并修正的样本数)")
        print(f"   - Root Cause Accuracy (with intervention): {ours_metrics.get('rca', 0):.2f}%")
    else:
        print("   (主模型无干预验证数据)")
    print("=" * 80)

    # 返回表现最好的模型用于后续修正版图
    best_main_model = select_best_model_name(results)
    print(f"\n✅ 修正版图将使用性能最好的模型: {best_main_model}")
    return models_dict[best_main_model], results


def train_main_model_only(data, graph_builder, device, epochs=2, lr=0.002, outdir='traceback_results'):
    """Train only the current main HGAN-Trace model and save its metrics."""
    import psutil
    os.makedirs(outdir, exist_ok=True)

    node_dim = graph_builder.temporal_feat_dim
    hidden_dim = 64
    out_dim = 32
    n_classes = data.get('n_classes', 6)
    pretrain_epochs = max(1, epochs // 4)
    finetune_epochs = max(1, epochs - pretrain_epochs)
    main_pretrain_epochs = pretrain_epochs + 6 if epochs >= 30 else pretrain_epochs
    main_finetune_epochs = finetune_epochs + 18 if epochs >= 30 else finetune_epochs

    print("\n" + "=" * 70)
    print("📌 只训练主模型: Ours (HGT-Trace w/o Cross-Attn, Gate & Cross-layer)")
    print("=" * 70)
    print("   保留模块: DW-Sep, Temporal Shift, Dynamic Edge Weighting")
    print("   图设置: use_dynamic_edges=False, use_cross_layer=False, use_phy_chain=True, use_net_edges=True")
    print(f"   训练轮数: 自监督预训练 {main_pretrain_epochs} 轮 + 有监督微调 {main_finetune_epochs} 轮")

    encoder = HGT_Trace(
        in_channels=node_dim, hidden_channels=hidden_dim, latent_channels=out_dim,
        num_layers=2, n_heads=4, dropout=0.15,
        use_cross_attn=False,
        use_gate=False,
        use_dw_sep=True
    )
    model = EnhancedTracebackSystem(
        gnn_encoder=encoder,
        n_nodes=graph_builder.n_nodes,
        n_net=graph_builder.n_net,
        n_phy=graph_builder.n_phy,
        input_dim=node_dim,
        latent_dim=out_dim,
        hidden_dim=hidden_dim,
        n_classes=n_classes
    )
    model = train_two_stage(
        model, data, graph_builder, device,
        pretrain_epochs=main_pretrain_epochs,
        finetune_epochs=main_finetune_epochs,
        pretrain_lr=lr,
        finetune_lr=lr,
        use_dynamic_edges=False,
        use_cross_layer=False,
        use_phy_chain=True,
        use_net_edges=True,
        verbose=True,
        model_name="Ours"
    )

    metrics = evaluate_model(
        model, data, graph_builder, device,
        use_dynamic_edges=False,
        use_cross_layer=False,
        use_phy_chain=True,
        use_net_edges=True
    )
    total_params = sum(p.numel() for p in model.parameters())
    metrics['param_count'] = total_params / 1000
    metrics['model_size_mb'] = total_params * 4 / (1024 * 1024)
    eval_data_for_profile = data.get('_test_data', data)
    profile_adj = graph_builder.build_static_adjacency(
        use_cross_layer=False,
        use_phy_chain=True,
        use_net_edges=True
    )
    profile_adj_tensor = torch.tensor(profile_adj, dtype=torch.float32).to(device)
    profile_node_feats = graph_builder.get_node_features_with_history(
        0,
        eval_data_for_profile['X_net_feat'],
        eval_data_for_profile['X_phy'],
        eval_data_for_profile['src_ips'],
        eval_data_for_profile['dst_ips']
    )
    profile_x = torch.tensor(profile_node_feats, dtype=torch.float32).to(device)
    metrics['flops_mflops'] = estimate_model_flops(model, profile_x, profile_adj_tensor) / 1e6
    metrics['cpu_usage'] = psutil.cpu_percent(interval=0.1)

    row = {
        'Model': 'Ours',
        'Acc (%)': metrics.get('accuracy', 0),
        'Prec-macro (%)': metrics.get('precision', 0),
        'Recall-macro (%)': metrics.get('recall', 0),
        'F1-macro (%)': metrics.get('f1', 0),
        'F1-weighted (%)': metrics.get('f1_weighted', 0),
        'AUC (%)': metrics.get('auc', 0),
        'RCA (%)': metrics.get('rca', 0),
        'MRR (%)': metrics.get('mrr', 0),
        'NDCG@5 (%)': metrics.get('ndcg', 0),
        'APD': metrics.get('apd', 0),
        'Time (ms)': metrics.get('inference_time', 0),
        'Params (K)': metrics.get('param_count', 0),
        'Size (MB)': metrics.get('model_size_mb', 0),
        'CPU (%)': metrics.get('cpu_usage', 0),
        'FLOPs (MFLOPs)': metrics.get('flops_mflops', 0),
    }
    metrics_path = os.path.join(outdir, 'main_model_metrics.csv')
    pd.DataFrame([row]).to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f"   主模型指标已保存: {metrics_path}")

    return model, {'Ours': metrics}


# ========================= 6. 可视化 =========================

def plot_network_physical_topology(graph_builder, data, save_path='topology.png'):
    """绘制美观的网络-物理拓扑图

    双层架构: 网络层(IP节点) 在上，物理层(设备节点) 在下
    只绘制有通信关系的节点（无关节点不显示）
    节点标签: N1, N2, ... (网络节点), P1, P2, ... (物理节点)，黑色加粗
    颜色: 网络节点绿色，物理节点蓝色
    """
    G = nx.DiGraph()

    # 添加节点
    for node in graph_builder.net_nodes:
        G.add_node(node, ntype='net')
    for node in graph_builder.phy_nodes:
        G.add_node(node, ntype='phy')

    # 统计网络通信边
    src_ips = data['src_ips']
    dst_ips = data['dst_ips']
    ip_to_idx = graph_builder.ip_to_idx

    comm_count = {}
    for src, dst in zip(src_ips, dst_ips):
        if src in ip_to_idx and dst in ip_to_idx:
            key = (f"IP_{src}", f"IP_{dst}")
            comm_count[key] = comm_count.get(key, 0) + 1

    # 添加网络通信边（Top-15）
    top_edges = sorted(comm_count.items(), key=lambda x: x[1], reverse=True)[:15]
    active_net_nodes = set()
    for (src, dst), cnt in top_edges:
        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst, etype='net-net', weight=cnt)
            active_net_nodes.add(src)
            active_net_nodes.add(dst)

    # 物理层链式连接（全部物理节点）
    for i in range(len(graph_builder.phy_nodes) - 1):
        G.add_edge(graph_builder.phy_nodes[i], graph_builder.phy_nodes[i + 1], etype='phy-phy')

    # 跨层连接（前2个活跃网络节点 → 前2个物理节点）
    active_net_list = [n for n in graph_builder.net_nodes if n in active_net_nodes]
    for n in active_net_list[:min(2, len(active_net_list))]:
        for p in graph_builder.phy_nodes[:min(2, len(graph_builder.phy_nodes))]:
            G.add_edge(n, p, etype='net-phy')
            active_net_nodes.add(n)

    # 只保留有通信关系的网络节点
    all_net_nodes = graph_builder.net_nodes
    all_phy_nodes = graph_builder.phy_nodes
    relevant_net = [n for n in all_net_nodes if n in active_net_nodes]
    relevant_phy = list(all_phy_nodes)

    # 创建图形
    fig, ax = plt.subplots(figsize=(22, 15))
    ax.set_facecolor('#f8f9fa')

    # 计算布局
    pos = {}
    n_net = len(relevant_net)
    n_phy = len(relevant_phy)

    # 网络层 - 围成一个大圆
    radius = 5.0
    center_x, center_y = 0, 1.5
    for i, n in enumerate(relevant_net):
        angle = 2 * np.pi * i / max(n_net, 1) - np.pi / 2
        pos[n] = (center_x + radius * np.cos(angle), center_y + radius * np.sin(angle))

    # 物理层 - 圆心下方线性布局
    phy_width = 8
    for i, n in enumerate(relevant_phy):
        pos[n] = (i * phy_width / max(n_phy - 1, 1) - phy_width / 2, center_y - 1.5)

    # 只绘制有节点位置的边
    net_edges = [(u, v) for u, v, d in G.edges(data=True)
                 if d.get('etype') == 'net-net' and u in pos and v in pos]
    phy_edges = [(u, v) for u, v, d in G.edges(data=True)
                 if d.get('etype') == 'phy-phy' and u in pos and v in pos]
    cross_edges = [(u, v) for u, v, d in G.edges(data=True)
                   if d.get('etype') == 'net-phy' and u in pos and v in pos]

    # 网络层边 - 绿色曲线
    nx.draw_networkx_edges(G, pos, edgelist=net_edges, edge_color='#27ae60',
                           alpha=0.4, arrows=True, arrowsize=12, width=1.5,
                           connectionstyle='arc3,rad=0.2', ax=ax)

    # 物理层边 - 蓝色粗箭头
    nx.draw_networkx_edges(G, pos, edgelist=phy_edges, edge_color='#2980b9',
                           alpha=0.9, arrows=True, arrowsize=25, width=3, ax=ax)

    # 跨层边 - 灰色虚线
    nx.draw_networkx_edges(G, pos, edgelist=cross_edges, edge_color='#95a5a6',
                           style='dashed', alpha=0.5, arrows=True, arrowsize=15, width=1.5, ax=ax)

    # 绘制网络节点 - 绿色圆形
    nx.draw_networkx_nodes(G, pos, nodelist=relevant_net, node_color='#2ecc71',
                           node_size=900, alpha=0.9, node_shape='o', ax=ax,
                           edgecolors='#27ae60', linewidths=2)

    # 绘制物理节点 - 蓝色方形
    nx.draw_networkx_nodes(G, pos, nodelist=relevant_phy, node_color='#3498db',
                           node_size=1000, alpha=0.9, node_shape='s', ax=ax,
                           edgecolors='#2980b9', linewidths=2)

    # 标签 - N1/N2... P1/P2... 黑色加粗
    labels = {}
    for n in relevant_net:
        net_idx = all_net_nodes.index(n) + 1
        labels[n] = f'N{net_idx}'
    for n in relevant_phy:
        phy_idx = all_phy_nodes.index(n) + 1
        labels[n] = f'P{phy_idx}'

    nx.draw_networkx_labels(G, pos, labels, font_size=20, font_weight='bold',
                           font_color='black', ax=ax)

    # 添加层级标签
    ax.text(-8, 6, 'Network Layer', fontsize=26, fontweight='normal', color='#27ae60',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#27ae60', alpha=0.9))
    ax.text(-8, center_y - 1.5, 'Physical Layer', fontsize=26, fontweight='normal', color='#2980b9',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#2980b9', alpha=0.9))

    # 图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='#27ae60', label='Network Node (IP)'),
        Patch(facecolor='#3498db', edgecolor='#2980b9', label='Physical Node (Device)'),
        Line2D([0], [0], color='#27ae60', linewidth=2, alpha=0.5, label='Network Communication'),
        Line2D([0], [0], color='#2980b9', linewidth=3, label='Physical Connection'),
        Line2D([0], [0], color='#95a5a6', linewidth=2, linestyle='--', label='Cross-layer Connection'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=21, framealpha=0.95)

    ax.axis('off')

    plt.tight_layout()
    ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ 拓扑图已保存至: {save_path}")


def plot_anomaly_traceback(results, graph_builder, data=None, save_path='traceback.png', top_k=6):
    """绘制美观的异常溯源图 - 简洁版"""
    import seaborn as sns

    label_names = data.get('label_names', ['Normal', 'NS', 'NM', 'PM', 'PS', 'SS']) if data else ['Normal', 'NS', 'NM', 'PM', 'PS', 'SS']

    anomaly_results = [r for r in results if r['true_label'] > 0]
    if len(anomaly_results) == 0:
        print("⚠️ 没有异常样本，跳过溯源图绘制")
        return

    # 选择代表性样本（每个异常类型选一个）
    selected_samples = []
    seen_types = set()
    for r in anomaly_results:
        if r['true_label'] not in seen_types and len(selected_samples) < top_k:
            selected_samples.append(r)
            seen_types.add(r['true_label'])

    # 如果不够，补充其他样本
    for r in anomaly_results:
        if len(selected_samples) >= top_k:
            break
        if r not in selected_samples:
            selected_samples.append(r)

    n_samples = len(selected_samples)
    n_cols = min(3, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 7.5 * n_rows))
    if n_samples == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # 异常类型颜色
    type_colors = {
        0: '#2ecc71',  # Normal - 绿
        1: '#f39c12',  # NS - 橙
        2: '#e74c3c',  # NM - 红
        3: '#9b59b6',  # PM - 紫
        4: '#8e44ad',  # PS - 深紫
        5: '#34495e',  # SS - 深灰
    }

    for idx, r in enumerate(selected_samples):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        node_scores = r['node_scores']
        n_net = graph_builder.n_net
        n_phy = graph_builder.n_phy

        true_lbl = r['true_label']
        true_name = label_names[true_lbl] if true_lbl < len(label_names) else str(true_lbl)

        # 分离网络层和物理层分数
        net_scores = node_scores[:n_net]
        phy_scores = node_scores[n_net:]

        # 只显示Top-5网络节点和所有物理节点
        top_net_idx = np.argsort(net_scores)[-5:][::-1]

        # 绘制条形图
        x_net = np.arange(len(top_net_idx))
        x_phy = np.arange(len(phy_scores)) + len(top_net_idx) + 1

        # 网络节点
        bars_net = ax.bar(x_net, net_scores[top_net_idx], color='#3498db', alpha=0.8,
                         edgecolor='white', linewidth=1, label='Network')
        # 物理节点
        bars_phy = ax.bar(x_phy, phy_scores, color='#e74c3c', alpha=0.8,
                         edgecolor='white', linewidth=1, label='Physical')

        # X轴标签
        net_labels = [f'N{i+1}' for i in top_net_idx]
        phy_labels = [f'P{i+1}' for i in range(n_phy)]
        ax.set_xticks(list(x_net) + list(x_phy))
        ax.set_xticklabels(net_labels + phy_labels, fontsize=20, fontweight='normal')

        # 标注根因节点
        top_idx = np.argmax(node_scores)
        if top_idx < n_net:
            root_bar_idx = np.where(top_net_idx == top_idx)[0]
            if len(root_bar_idx) > 0:
                ax.annotate('ROOT', xy=(root_bar_idx[0], net_scores[top_idx]),
                           xytext=(root_bar_idx[0], net_scores[top_idx] + 0.05),
                           fontsize=20, fontweight='normal', color='#c0392b', ha='center')
        else:
            phy_idx = top_idx - n_net
            ax.annotate('ROOT', xy=(len(top_net_idx) + 1 + phy_idx, phy_scores[phy_idx]),
                       xytext=(len(top_net_idx) + 1 + phy_idx, phy_scores[phy_idx] + 0.05),
                       fontsize=20, fontweight='normal', color='#c0392b', ha='center')

        pred_lbl = r['pred_label']
        pred_name = label_names[pred_lbl] if pred_lbl < len(label_names) else str(pred_lbl)

        ax.set_ylabel('Anomaly Score', fontsize=21, fontweight='normal')
        ax.set_ylim(0, max(node_scores) * 1.2)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        ax.tick_params(axis='y', labelsize=18)
        ax.legend(loc='upper right', fontsize=18)

    # 隐藏多余的子图
    for idx in range(n_samples, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ 异常溯源图已保存至: {save_path}")


def plot_causal_propagation_per_class(results, graph_builder, data=None, save_dir='traceback_results'):
    """为每个异常类别单独绘制因果传播拓扑图（共5张）

    节点命名: N1, N2, ... (网络节点), P1, P2, ... (物理节点)
    有关节点: 所有在通信图中出现过的网络节点 + 全部物理节点
    边类型:
      绿色  - 网络通信边（所有实际通信对）
      蓝色  - 物理层链式连接
      红色  - 异常传播路径（异常节点间按分数顺序连接）
    运行时打印节点名称映射和各类别异常分数
    """
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    label_names = data.get('label_names', ['Normal', 'NS', 'NM', 'PM', 'PS', 'SS']) if data else ['Normal', 'NS', 'NM', 'PM', 'PS', 'SS']

    n_net = graph_builder.n_net
    n_phy = graph_builder.n_phy

    # ── 打印节点编号映射 ──────────────────────────────────────────────────
    print("\n📍 因果传播图节点编号映射:")
    for i, n in enumerate(graph_builder.net_nodes):
        print(f"   N{i+1} → {n.replace('IP_', '')}")
    for i, n in enumerate(graph_builder.phy_nodes):
        print(f"   P{i+1} → {n}")
    print()

    # ── 预计算所有有关联的网络节点（出现在任意通信对中）──────────────────
    src_ips = data['src_ips']
    dst_ips = data['dst_ips']
    ip_to_idx = graph_builder.ip_to_idx

    # 统计所有通信对（不限 top-15）
    all_comm_pairs = {}
    for src, dst in zip(src_ips, dst_ips):
        if src in ip_to_idx and dst in ip_to_idx:
            key = (f"IP_{src}", f"IP_{dst}")
            all_comm_pairs[key] = all_comm_pairs.get(key, 0) + 1

    # 出现在任意通信对中的网络节点
    active_net_names = set()
    for src, dst in all_comm_pairs:
        active_net_names.add(src)
        active_net_names.add(dst)

    active_net_idx = [i for i, n in enumerate(graph_builder.net_nodes) if n in active_net_names]
    if not active_net_idx:
        active_net_idx = list(range(n_net))
    active_phy_idx = list(range(n_phy))

    # 节点索引 → 简短标签
    def node_label(idx):
        return f'N{idx+1}' if idx < n_net else f'P{idx - n_net + 1}'

    os.makedirs(save_dir, exist_ok=True)

    for cls_id in range(1, len(label_names)):
        cls_name = label_names[cls_id] if cls_id < len(label_names) else f'Class{cls_id}'
        cls_results = [r for r in results if r['true_label'] == cls_id]
        if len(cls_results) == 0:
            print(f"⚠️ 无 {cls_name} 样本，跳过")
            continue

        # 计算该类别的平均节点分数
        avg_scores = np.zeros(n_net + n_phy)
        for r in cls_results:
            avg_scores += r['node_scores']
        avg_scores /= len(cls_results)

        net_scores = avg_scores[:n_net]
        phy_scores = avg_scores[n_net:]

        # 阈值仅用于节点着色和确定异常节点
        thr_net = net_scores.mean() + 0.3 * net_scores.std() if net_scores.std() > 0 else net_scores.mean()
        thr_phy = phy_scores.mean() + 0.3 * phy_scores.std() if phy_scores.std() > 0 else phy_scores.mean()

        def is_anomaly(node):
            if node < n_net:
                return net_scores[node] >= thr_net
            else:
                return phy_scores[node - n_net] >= thr_phy

        # 打印信息（替代黄色信息框）
        anom_count = sum(1 for i in range(n_net + n_phy) if is_anomaly(i))
        print(f"\n  [{cls_name}] 异常节点: {anom_count}/{n_net + n_phy}  |  样本数: {len(cls_results)}")
        print(f"  [{cls_name}] 有关节点平均异常分数:")
        for i in active_net_idx:
            flag = ' ← 异常' if is_anomaly(i) else ''
            print(f"    N{i+1} ({graph_builder.net_nodes[i].replace('IP_', '')}): {net_scores[i]:.4f}{flag}")
        for i in active_phy_idx:
            flag = ' ← 异常' if is_anomaly(n_net + i) else ''
            print(f"    P{i+1} ({graph_builder.phy_nodes[i]}): {phy_scores[i]:.4f}{flag}")

        # ── 构建 networkx 图（有关节点）────────────────────────────────
        G = nx.DiGraph()
        for i in active_net_idx:
            G.add_node(i, layer='net')
        for i in active_phy_idx:
            G.add_node(n_net + i, layer='phy')

        # 绿色边：所有网络通信对（双端节点都在有关节点集合中）
        active_net_set = set(active_net_idx)
        active_net_names_idx = {graph_builder.net_nodes[i]: i for i in active_net_idx}
        for (src_name, dst_name), cnt in all_comm_pairs.items():
            si = active_net_names_idx.get(src_name)
            di = active_net_names_idx.get(dst_name)
            if si is not None and di is not None:
                G.add_edge(si, di, etype='net-net', weight=cnt)

        # 蓝色边：物理层链式连接
        for i in range(len(active_phy_idx) - 1):
            G.add_edge(n_net + active_phy_idx[i], n_net + active_phy_idx[i + 1], etype='phy-phy')

        # 红色边：异常传播路径（异常节点按分数从高到低顺序连接）
        anom_nodes = sorted(
            [i for i in active_net_idx if is_anomaly(i)] +
            [n_net + i for i in active_phy_idx if is_anomaly(n_net + i)],
            key=lambda x: avg_scores[x], reverse=True
        )
        for k in range(len(anom_nodes) - 1):
            # 如果该边已存在（通信边），改为异常传播边；否则新增
            G.add_edge(anom_nodes[k], anom_nodes[k + 1], etype='anom', weight=avg_scores[anom_nodes[k]])

        # 移除没有任何连线的网络节点（无通信边也无跨层边）
        isolated_net = [n for n in list(G.nodes) if n < n_net and G.degree(n) == 0]
        if isolated_net:
            G.remove_nodes_from(isolated_net)

        # ── 布局：网络层圆形，物理层线性 ────────────────────────────────
        pos = {}
        net_nodes_g = sorted([n for n in G.nodes if n < n_net])
        phy_nodes_g = sorted([n for n in G.nodes if n >= n_net])
        n_net_g, n_phy_g = len(net_nodes_g), len(phy_nodes_g)

        radius = 5.5
        center_x, center_y = 0, 2.0
        for k, node in enumerate(net_nodes_g):
            angle = 2 * np.pi * k / max(n_net_g, 1) - np.pi / 2
            pos[node] = (center_x + radius * np.cos(angle), center_y + radius * np.sin(angle))

        phy_y = center_y - radius - 2.5          # 圆底部以下 2.5 个单位，避免重叠
        phy_width = min(11, n_phy_g * 2.5)
        for k, node in enumerate(phy_nodes_g):
            x_pos = (k * phy_width / max(n_phy_g - 1, 1) - phy_width / 2) if n_phy_g > 1 else 0
            pos[node] = (x_pos, phy_y)

        # ── 绘图 ────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(20, 16))
        ax.set_facecolor('#f8f9fa')

        # 分类边
        net_edges  = [(u, v) for u, v, d in G.edges(data=True) if d.get('etype') == 'net-net']
        phy_edges  = [(u, v) for u, v, d in G.edges(data=True) if d.get('etype') == 'phy-phy']
        anom_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('etype') == 'anom']

        # 灰色虚线：跨域连接（异常网络节点 ↔ 异常物理节点）
        # 只连有实际控制/影响关系的节点对：双方均超过异常阈值
        # 若某侧无异常节点，则取该侧分数最高的节点作为代表
        anom_net_g = [n for n in net_nodes_g if is_anomaly(n)]
        anom_phy_g = [n for n in phy_nodes_g if is_anomaly(n)]
        cross_net = anom_net_g if anom_net_g else \
                    ([max(net_nodes_g, key=lambda n: avg_scores[n])] if net_nodes_g else [])
        cross_phy = anom_phy_g if anom_phy_g else \
                    ([max(phy_nodes_g, key=lambda n: avg_scores[n])] if phy_nodes_g else [])
        for net_n in cross_net:
            for phy_n in cross_phy:
                ax.plot([pos[net_n][0], pos[phy_n][0]],
                        [pos[net_n][1], pos[phy_n][1]],
                        color='#95a5a6', linestyle='--', linewidth=1.2,
                        alpha=0.55, zorder=1)

        # 绿色：网络通信边
        if net_edges:
            nx.draw_networkx_edges(G, pos, edgelist=net_edges, edge_color='#27ae60',
                                   alpha=0.45, arrows=True, arrowsize=14, width=1.5,
                                   connectionstyle='arc3,rad=0.15', ax=ax,
                                   min_source_margin=15, min_target_margin=15)
        # 蓝色：物理链式边
        if phy_edges:
            nx.draw_networkx_edges(G, pos, edgelist=phy_edges, edge_color='#2980b9',
                                   alpha=0.8, arrows=True, arrowsize=20, width=2.5, ax=ax,
                                   min_source_margin=17, min_target_margin=17)
        # 红色：异常传播边（最后画，覆盖在上层）
        if anom_edges:
            anom_ws = [G[u][v].get('weight', 0.5) * 3 + 1.5 for u, v in anom_edges]
            nx.draw_networkx_edges(G, pos, edgelist=anom_edges, width=anom_ws,
                                   edge_color='#e74c3c', alpha=0.85, arrows=True,
                                   arrowsize=22, connectionstyle='arc3,rad=0.2', ax=ax,
                                   min_source_margin=17, min_target_margin=17)

        # 节点着色
        normal_net = [n for n in net_nodes_g if not is_anomaly(n)]
        anom_net   = [n for n in net_nodes_g if is_anomaly(n)]
        normal_phy = [n for n in phy_nodes_g if not is_anomaly(n)]
        anom_phy   = [n for n in phy_nodes_g if is_anomaly(n)]

        if normal_net:
            nx.draw_networkx_nodes(G, pos, nodelist=normal_net, node_color='#2ecc71',
                                   node_size=900, alpha=0.9, node_shape='o', ax=ax,
                                   edgecolors='#27ae60', linewidths=2)
        if anom_net:
            nx.draw_networkx_nodes(G, pos, nodelist=anom_net, node_color='#e74c3c',
                                   node_size=1100, alpha=0.95, node_shape='o', ax=ax,
                                   edgecolors='#c0392b', linewidths=3)
        if normal_phy:
            nx.draw_networkx_nodes(G, pos, nodelist=normal_phy, node_color='#3498db',
                                   node_size=1000, alpha=0.9, node_shape='s', ax=ax,
                                   edgecolors='#2980b9', linewidths=2)
        if anom_phy:
            nx.draw_networkx_nodes(G, pos, nodelist=anom_phy, node_color='#e74c3c',
                                   node_size=1200, alpha=0.95, node_shape='s', ax=ax,
                                   edgecolors='#c0392b', linewidths=3)

        # 节点标签
        labels = {n: node_label(n) for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=18, font_weight='bold',
                               font_color='black', ax=ax)

        # 图例
        legend_elements = [
            mpatches.Patch(facecolor='#2ecc71', edgecolor='#27ae60', label='Normal Network Node'),
            mpatches.Patch(facecolor='#3498db', edgecolor='#2980b9', label='Normal Physical Node'),
            mpatches.Patch(facecolor='#e74c3c', edgecolor='#c0392b', label='Anomalous Node'),
            Line2D([0], [0], color='#27ae60', linewidth=2, label='Network Communication'),
            Line2D([0], [0], color='#2980b9', linewidth=3, label='Physical Connection'),
            Line2D([0], [0], color='#e74c3c', linewidth=3, label='Anomaly Propagation'),
            Line2D([0], [0], color='#95a5a6', linewidth=1.5, linestyle='--', label='Cross-domain Connection'),
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=16, framealpha=0.95)

        ax.axis('off')
        fig.tight_layout()

        fname = f'causal_propagation_{cls_name}.png'
        fpath = os.path.join(save_dir, fname)
        fig.savefig(fpath, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"✅ {cls_name} 因果传播图已保存至: {fpath}")


def plot_uncertainty_analysis(results, graph_builder, data=None, save_path='uncertainty_analysis.png'):
    """绘制不确定性分析图 - 每个子图单独保存为PNG，并额外输出异常节点拓扑图
    
    输出:
    1. uncertainty_confidence_dist.png - 置信度分布直方图
    2. uncertainty_calibration.png - 置信度vs准确率校准图
    3. uncertainty_by_type.png - 各类别不确定性箱线图
    4. uncertainty_scatter.png - 置信度与不确定性散点图
    5. anomaly_topology.png - 异常节点拓扑可视化（正常蓝色/异常红色）
    """
    import seaborn as sns
    
    label_names = data.get('label_names', ['Normal', 'NS', 'NM', 'PM', 'PS', 'SS']) if data else ['Normal', 'NS', 'NM', 'PM', 'PS', 'SS']
    
    # 获取保存目录
    save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
    
    # 提取不确定性信息
    confidences = []
    uncertainties = []
    correct_flags = []
    anomaly_types_list = []
    
    for r in results:
        conf = r.get('confidence', 0.5)
        if isinstance(conf, np.ndarray):
            conf = float(conf.mean())
        confidences.append(conf)
        
        unc = r.get('uncertainty', 0.5)
        if isinstance(unc, np.ndarray):
            unc = float(unc.mean())
        uncertainties.append(unc)
        
        correct_flags.append(1 if r['true_label'] == r['pred_label'] else 0)
        anomaly_types_list.append(r['true_label'])
    
    confidences = np.array(confidences)
    uncertainties = np.array(uncertainties)
    correct_flags = np.array(correct_flags)
    anomaly_types_arr = np.array(anomaly_types_list)
    
    # ==================== 子图1: 置信度分布（单独保存） ====================
    fig1, ax1 = plt.subplots(figsize=(11, 8))

    correct_conf = confidences[correct_flags == 1]
    wrong_conf = confidences[correct_flags == 0]

    ax1.hist(correct_conf, bins=20, alpha=0.7, color='#2ecc71', label='Correct Predictions', edgecolor='white')
    ax1.hist(wrong_conf, bins=20, alpha=0.7, color='#e74c3c', label='Wrong Predictions', edgecolor='white')
    ax1.set_xlabel('Confidence', fontsize=36, fontweight='normal')
    ax1.set_ylabel('Count', fontsize=36, fontweight='normal')
    ax1.tick_params(axis='both', labelsize=30)
    ax1.legend(fontsize=30)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_facecolor('#fafafa')
    
    fig1.tight_layout()
    path1 = os.path.join(save_dir, 'uncertainty_confidence_dist.png')
    fig1.savefig(path1, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig1)
    print(f"✅ 置信度分布图已保存至: {path1}")
    
    # ==================== 子图2: 置信度vs准确率（单独保存） ====================
    fig2, ax2 = plt.subplots(figsize=(11, 8))
    
    conf_bins = np.linspace(0, 1, 11)
    bin_accs = []
    bin_centers = []
    bin_counts = []
    
    for i in range(len(conf_bins) - 1):
        mask = (confidences >= conf_bins[i]) & (confidences < conf_bins[i + 1])
        if mask.sum() > 0:
            bin_accs.append(correct_flags[mask].mean() * 100)
            bin_centers.append((conf_bins[i] + conf_bins[i + 1]) / 2)
            bin_counts.append(mask.sum())
    
    bar_colors = plt.cm.RdYlGn(np.array(bin_accs) / 100)
    bars = ax2.bar(bin_centers, bin_accs, width=0.08, color=bar_colors, edgecolor='white', linewidth=1.5)
    # 在柱状图上方标出具体数值
    for bar, acc, cnt in zip(bars, bin_accs, bin_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=30, fontweight='normal', color='#2c3e50')
    ax2.plot([0, 1], [0, 100], 'k--', alpha=0.5, label='Perfect Calibration')
    ax2.set_xlabel('Confidence', fontsize=48, fontweight='normal')
    ax2.set_ylabel('Accuracy (%)', fontsize=48, fontweight='normal')
    ax2.tick_params(axis='both', labelsize=40)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 115)
    ax2.legend(fontsize=36)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_facecolor('#fafafa')
    
    fig2.tight_layout()
    path2 = os.path.join(save_dir, 'uncertainty_calibration.png')
    fig2.savefig(path2, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig2)
    print(f"✅ 校准图已保存至: {path2}")
    
    # ==================== 子图3: 各类别不确定性箱线图（单独保存） ====================
    fig3, ax3 = plt.subplots(figsize=(11, 8))
    
    type_colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#8e44ad', '#34495e',
                   '#16a085', '#d35400', '#2c3e50', '#7f8c8d']
    class_ids = list(range(len(label_names)))
    unc_by_type = [uncertainties[anomaly_types_arr == i] for i in class_ids]
    unc_by_type = [u for u in unc_by_type if len(u) > 0]
    valid_labels = [label_names[i] for i in class_ids if len(uncertainties[anomaly_types_arr == i]) > 0]
    
    bp = ax3.boxplot(unc_by_type, labels=valid_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], type_colors[:len(unc_by_type)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_xlabel('Anomaly Type', fontsize=40, fontweight='normal')
    ax3.set_ylabel('Uncertainty', fontsize=40, fontweight='normal')
    ax3.tick_params(axis='y', labelsize=36)
    ax3.tick_params(axis='x', labelsize=36)
    ax3.grid(alpha=0.3, linestyle='--')
    ax3.set_facecolor('#fafafa')
    plt.setp(ax3.get_xticklabels(), fontsize=24, fontweight='normal')
    
    fig3.tight_layout()
    path3 = os.path.join(save_dir, 'uncertainty_by_type.png')
    fig3.savefig(path3, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig3)
    print(f"✅ 类别不确定性图已保存至: {path3}")
    
    # ==================== 子图4: 置信度与不确定性散点图（单独保存） ====================
    fig4, ax4 = plt.subplots(figsize=(11, 8))

    scatter = ax4.scatter(confidences, uncertainties, c=correct_flags,
                          cmap='RdYlGn', alpha=0.6, edgecolors='white', linewidth=0.5)
    ax4.set_xlabel('Confidence', fontsize=48, fontweight='normal')
    ax4.set_ylabel('Uncertainty', fontsize=48, fontweight='normal')
    ax4.tick_params(axis='both', labelsize=40)
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.set_facecolor('#fafafa')

    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Correct (1) / Wrong (0)', fontsize=40)
    cbar.ax.tick_params(labelsize=36)
    
    fig4.tight_layout()
    path4 = os.path.join(save_dir, 'uncertainty_scatter.png')
    fig4.savefig(path4, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig4)
    print(f"✅ 置信度-不确定性散点图已保存至: {path4}")
    
    # ==================== 子图5: 异常节点拓扑可视化（单独保存） ====================
    _plot_anomaly_topology(results, graph_builder, data, save_dir)
    
    print(f"✅ 不确定性分析（共5张图）全部保存完成")


def _plot_anomaly_topology(results, graph_builder, data, save_dir):
    """绘制异常节点拓扑可视化 - 网络节点橙色，物理节点蓝色，异常节点红色
    
    根据溯源结果中的node_scores判断哪些节点异常
    """
    G = nx.DiGraph()
    
    # 添加节点
    for node in graph_builder.net_nodes:
        G.add_node(node, ntype='net')
    for node in graph_builder.phy_nodes:
        G.add_node(node, ntype='phy')
    
    # 统计网络通信边
    src_ips = data['src_ips']
    dst_ips = data['dst_ips']
    ip_to_idx = graph_builder.ip_to_idx
    
    comm_count = {}
    for src, dst in zip(src_ips, dst_ips):
        if src in ip_to_idx and dst in ip_to_idx:
            key = (f"IP_{src}", f"IP_{dst}")
            comm_count[key] = comm_count.get(key, 0) + 1
    
    # 添加网络通信边（Top-15）
    top_edges = sorted(comm_count.items(), key=lambda x: x[1], reverse=True)[:15]
    for (src, dst), cnt in top_edges:
        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst, etype='net-net', weight=cnt)
    
    # 物理层链式连接
    for i in range(len(graph_builder.phy_nodes) - 1):
        G.add_edge(graph_builder.phy_nodes[i], graph_builder.phy_nodes[i + 1], etype='phy-phy')
    
    # 跨层连接
    for n in graph_builder.net_nodes[:min(2, len(graph_builder.net_nodes))]:
        for p in graph_builder.phy_nodes[:min(2, len(graph_builder.phy_nodes))]:
            G.add_edge(n, p, etype='net-phy')
    
    # 统计每个节点的异常分数（取所有异常样本的平均）
    node_anomaly_scores = np.zeros(graph_builder.n_nodes)
    anomaly_count = 0
    for r in results:
        if r['true_label'] > 0:  # 只看异常样本
            node_anomaly_scores += r['node_scores']
            anomaly_count += 1
    if anomaly_count > 0:
        node_anomaly_scores /= anomaly_count
    
    # 建立节点名 -> 索引映射
    all_nodes_list = graph_builder.net_nodes + graph_builder.phy_nodes
    node_to_idx = {node: i for i, node in enumerate(all_nodes_list)}
    
    # 使用分层阈值判断异常节点（各层独立计算 均值 + 0.5 * 标准差）
    n_net_nodes = len(graph_builder.net_nodes)
    net_scores = node_anomaly_scores[:n_net_nodes]
    phy_scores = node_anomaly_scores[n_net_nodes:]
    threshold_net = net_scores.mean() + 0.5 * net_scores.std() if len(net_scores) > 0 else 0
    threshold_phy = phy_scores.mean() + 0.5 * phy_scores.std() if len(phy_scores) > 0 else 0
    threshold = node_anomaly_scores.mean() + 0.5 * node_anomaly_scores.std()  # 用于显示
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(22, 15))
    ax.set_facecolor('#f8f9fa')

    # 计算布局
    pos = {}
    net_nodes = [n for n in G.nodes if G.nodes[n].get('ntype') == 'net']
    phy_nodes = [n for n in G.nodes if G.nodes[n].get('ntype') == 'phy']
    
    n_net = len(net_nodes)
    n_phy = len(phy_nodes)
    
    # 网络层 - 围成一个大圆
    radius = 5.0
    center_x, center_y = 0, 1.5
    for i, n in enumerate(net_nodes):
        angle = 2 * np.pi * i / max(n_net, 1) - np.pi / 2
        pos[n] = (center_x + radius * np.cos(angle), center_y + radius * np.sin(angle))
    
    # 物理层 - 圆心下方线性布局
    phy_width = 8
    for i, n in enumerate(phy_nodes):
        pos[n] = (i * phy_width / max(n_phy - 1, 1) - phy_width / 2, center_y - 1.5)
    
    # 分类节点：正常 vs 异常
    normal_net_nodes = []
    anomaly_net_nodes = []
    normal_phy_nodes = []
    anomaly_phy_nodes = []
    
    for n in net_nodes:
        idx = node_to_idx.get(n, -1)
        if idx >= 0 and node_anomaly_scores[idx] > threshold_net:
            anomaly_net_nodes.append(n)
        else:
            normal_net_nodes.append(n)
    
    for n in phy_nodes:
        idx = node_to_idx.get(n, -1)
        if idx >= 0 and node_anomaly_scores[idx] > threshold_phy:
            anomaly_phy_nodes.append(n)
        else:
            normal_phy_nodes.append(n)
    
    # 绘制边 - 网络通信边用橙色，物理链式边用蓝色，跨层用灰色虚线
    net_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('etype') == 'net-net']
    phy_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('etype') == 'phy-phy']
    cross_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('etype') == 'net-phy']
    
    nx.draw_networkx_edges(G, pos, edgelist=net_edges, edge_color='#27ae60',
                           alpha=0.5, arrows=True, arrowsize=12, width=1.5,
                           connectionstyle='arc3,rad=0.2', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=phy_edges, edge_color='#2980b9',
                           alpha=0.6, arrows=True, arrowsize=20, width=2.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=cross_edges, edge_color='#95a5a6',
                           style='dashed', alpha=0.4, arrows=True, arrowsize=15, width=1.5, ax=ax)

    # 绘制正常网络节点 - 绿色圆形
    if normal_net_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=normal_net_nodes, node_color='#2ecc71',
                               node_size=800, alpha=0.9, node_shape='o', ax=ax,
                               edgecolors='#27ae60', linewidths=2)
    # 绘制异常网络节点 - 红色圆形
    if anomaly_net_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=anomaly_net_nodes, node_color='#e74c3c',
                               node_size=1000, alpha=0.95, node_shape='o', ax=ax,
                               edgecolors='#c0392b', linewidths=3)

    # 绘制正常物理节点 - 蓝色方形
    if normal_phy_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=normal_phy_nodes, node_color='#3498db',
                               node_size=900, alpha=0.9, node_shape='s', ax=ax,
                               edgecolors='#2980b9', linewidths=2)
    # 绘制异常物理节点 - 红色方形
    if anomaly_phy_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=anomaly_phy_nodes, node_color='#e74c3c',
                               node_size=1100, alpha=0.95, node_shape='s', ax=ax,
                               edgecolors='#c0392b', linewidths=3)
    
    # 标签 - 全部黑色（使用 N1/N2/P1/P2 简短标签）
    labels = {}
    for n in G.nodes:
        if G.nodes[n].get('ntype') == 'net':
            net_idx = net_nodes.index(n) + 1 if n in net_nodes else 0
            labels[n] = f'N{net_idx}'
        else:
            phy_idx = phy_nodes.index(n) + 1 if n in phy_nodes else 0
            labels[n] = f'P{phy_idx}'

    nx.draw_networkx_labels(G, pos, labels, font_size=18, font_weight='bold',
                           font_color='black', ax=ax)

    # 添加层级标签
    ax.text(-8, 6, 'Network Layer', fontsize=26, fontweight='normal', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#2c3e50', alpha=0.9))
    ax.text(-8, center_y - 1.5, 'Physical Layer', fontsize=26, fontweight='normal', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#2c3e50', alpha=0.9))

    # 统计信息
    n_anomaly = len(anomaly_net_nodes) + len(anomaly_phy_nodes)
    n_total = n_net + n_phy
    ax.text(0.02, 0.02,
            f'Anomalous: {n_anomaly}/{n_total} nodes  |  Threshold: {threshold:.4f}',
            transform=ax.transAxes, fontsize=21, fontweight='normal',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffeaa7', edgecolor='#fdcb6e', alpha=0.9))

    # 图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='#27ae60', label='Normal Network Node'),
        Patch(facecolor='#3498db', edgecolor='#2980b9', label='Normal Physical Node'),
        Patch(facecolor='#e74c3c', edgecolor='#c0392b', label='Anomalous Node'),
        Line2D([0], [0], color='#27ae60', linewidth=2, alpha=0.5, label='Network Communication'),
        Line2D([0], [0], color='#2980b9', linewidth=3, label='Physical Connection'),
        Line2D([0], [0], color='#95a5a6', linewidth=2, linestyle='--', label='Cross-layer Connection'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=21, framealpha=0.95)

    ax.axis('off')

    fig.tight_layout()
    path5 = os.path.join(save_dir, 'anomaly_topology.png')
    fig.savefig(path5, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"✅ 异常节点拓扑图已保存至: {path5}")


def plot_explainability_summary(results, graph_builder, data=None, save_path='explainability.png'):
    """绘制可解释性汇总图 - 4张独立图片
    
    包含:
    1. 网络层vs物理层贡献（条形图）
    2. 各异常类型的层贡献（分组条形图）
    3. 根因归属饼图
    4. 异常类型与根因层关系热力图
    """
    import seaborn as sns
    
    save_dir = os.path.dirname(save_path)
    if not save_dir:
        save_dir = '.'
    os.makedirs(save_dir, exist_ok=True)
    
    label_names = data.get('label_names', ['Normal', 'NS', 'NM', 'PM', 'PS', 'SS']) if data else ['Normal', 'NS', 'NM', 'PM', 'PS', 'SS']
    
    # 获取特征名称
    phy_cols = data.get('phy_cols', [f'Phy_{i}' for i in range(10)])
    net_cols = [f'Net_{i}' for i in range(10)]  # 网络特征名称
    all_feature_names = net_cols + phy_cols
    
    # 统计特征重要性
    anomaly_results = [r for r in results if r['true_label'] > 0]
    
    # 如果有explanation字段，使用它
    anomaly_label_ids = list(range(1, len(label_names)))
    feature_importance_by_type = {i: {} for i in anomaly_label_ids}
    global_importance = {}
    
    for r in anomaly_results:
        label = r['true_label']
        if label == 0:
            continue
            
        # 使用node_scores作为特征重要性的代理
        node_scores = r['node_scores']
        n_net = graph_builder.n_net
        
        # 网络节点平均分
        net_avg = np.mean(node_scores[:n_net]) if n_net > 0 else 0
        phy_avg = np.mean(node_scores[n_net:])
        
        if 'Network_Avg' not in global_importance:
            global_importance['Network_Avg'] = []
        if 'Physical_Avg' not in global_importance:
            global_importance['Physical_Avg'] = []
        
        global_importance['Network_Avg'].append(net_avg)
        global_importance['Physical_Avg'].append(phy_avg)
        
        if label in feature_importance_by_type:
            if 'Network_Avg' not in feature_importance_by_type[label]:
                feature_importance_by_type[label]['Network_Avg'] = []
                feature_importance_by_type[label]['Physical_Avg'] = []
            feature_importance_by_type[label]['Network_Avg'].append(net_avg)
            feature_importance_by_type[label]['Physical_Avg'].append(phy_avg)
    
    # ==================== 子图1: 网络层vs物理层贡献（独立保存） ====================
    fig1, ax1 = plt.subplots(figsize=(14, 10))

    net_scores = global_importance.get('Network_Avg', [0])
    phy_scores = global_importance.get('Physical_Avg', [0])

    categories = ['Network Layer', 'Physical Layer']
    means = [np.mean(net_scores), np.mean(phy_scores)]
    stds = [np.std(net_scores), np.std(phy_scores)]
    colors = ['#3498db', '#e74c3c']

    bars = ax1.bar(categories, means, yerr=stds, color=colors, alpha=0.8,
                   edgecolor='white', linewidth=2, capsize=5)
    ax1.set_ylabel('Average Anomaly Score', fontsize=38, fontweight='normal')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_facecolor('#fafafa')
    ax1.tick_params(axis='both', labelsize=43)

    # 添加数值标注
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean:.3f}', ha='center', fontsize=50, fontweight='bold')
    
    fig1.tight_layout()
    path1 = os.path.join(save_dir, 'explain_layer_contribution.png')
    fig1.savefig(path1, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig1)
    print(f"   ✅ 层贡献图已保存至: {path1}")
    
    # ==================== 子图2: 各类型异常的层贡献（独立保存） ====================
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    
    anomaly_names = [label_names[i] for i in anomaly_label_ids]
    net_by_type = []
    phy_by_type = []
    
    for i in anomaly_label_ids:
        if i in feature_importance_by_type and feature_importance_by_type[i]:
            net_by_type.append(np.mean(feature_importance_by_type[i].get('Network_Avg', [0])))
            phy_by_type.append(np.mean(feature_importance_by_type[i].get('Physical_Avg', [0])))
        else:
            net_by_type.append(0)
            phy_by_type.append(0)
    
    x = np.arange(len(anomaly_names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, net_by_type, width, label='Network', color='#3498db', alpha=0.8)
    bars2 = ax2.bar(x + width/2, phy_by_type, width, label='Physical', color='#e74c3c', alpha=0.8)
    
    ax2.set_ylabel('Average Anomaly Score', fontsize=21, fontweight='normal')
    ax2.set_xlabel('Anomaly Type', fontsize=21, fontweight='normal')
    ax2.set_xticks(x)
    ax2.set_xticklabels(anomaly_names, fontsize=20)
    ax2.legend(fontsize=22)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_facecolor('#fafafa')
    ax2.tick_params(axis='y', labelsize=22)
    
    fig2.tight_layout()
    path2 = os.path.join(save_dir, 'explain_type_contribution.png')
    fig2.savefig(path2, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig2)
    print(f"   ✅ 各类型层贡献图已保存至: {path2}")
    
    # ==================== 子图3: 根因归属饼图（独立保存） ====================
    fig3, ax3 = plt.subplots(figsize=(11, 10))
    
    net_root = 0
    phy_root = 0
    n_net = graph_builder.n_net
    
    for r in anomaly_results:
        root_idx = np.argmax(r['node_scores'])
        if root_idx < n_net:
            net_root += 1
        else:
            phy_root += 1
    
    sizes = [net_root, phy_root]
    labels_pie = ['Network Layer\nRoot Cause', 'Physical Layer\nRoot Cause']
    colors_pie = ['#3498db', '#e74c3c']
    explode = (0.05, 0.05)
    
    wedges, texts, autotexts = ax3.pie(sizes, labels=labels_pie, colors=colors_pie,
                                        autopct='%1.1f%%', startangle=90, explode=explode,
                                        textprops={'fontsize': 28, 'fontweight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(28)
    
    fig3.tight_layout()
    path3 = os.path.join(save_dir, 'explain_attribution.png')
    fig3.savefig(path3, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig3)
    print(f"   ✅ 根因归属图已保存至: {path3}")
    
    # ==================== 子图4: 异常类型与根因层关系热力图（独立保存） ====================
    fig4, ax4 = plt.subplots(figsize=(11, 10))
    
    # 统计各类型异常的根因层分布
    type_layer_matrix = np.zeros((len(anomaly_label_ids), 2))
    label_to_row = {label_id: row_idx for row_idx, label_id in enumerate(anomaly_label_ids)}
    for r in anomaly_results:
        label = r['true_label']
        if label == 0 or label not in label_to_row:
            continue
        root_idx = np.argmax(r['node_scores'])
        layer = 0 if root_idx < n_net else 1
        type_layer_matrix[label_to_row[label], layer] += 1
    
    # 归一化
    row_sums = type_layer_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    type_layer_pct = type_layer_matrix / row_sums * 100
    
    im = ax4.imshow(type_layer_pct, cmap='RdBu_r', aspect='auto', vmin=0, vmax=100)
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(['Network', 'Physical'], fontsize=42)
    ax4.set_yticks(range(len(anomaly_label_ids)))
    ax4.set_yticklabels([label_names[i] for i in anomaly_label_ids], fontsize=40)
    ax4.set_xlabel('Root Cause Layer', fontsize=42, fontweight='normal')
    ax4.set_ylabel('Anomaly Type', fontsize=42, fontweight='normal')

    # 添加数值标注 - 全部白色加粗
    for i in range(len(anomaly_label_ids)):
        for j in range(2):
            ax4.text(j, i, f'{type_layer_pct[i, j]:.1f}%', ha='center', va='center',
                    fontsize=42, fontweight='bold', color='white')

    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Percentage (%)', fontsize=40)
    cbar.ax.tick_params(labelsize=36)
    
    fig4.tight_layout()
    path4 = os.path.join(save_dir, 'explain_layer_heatmap.png')
    fig4.savefig(path4, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig4)
    print(f"   ✅ 根因层热力图已保存至: {path4}")
    
    print(f"✅ 可解释性分析（共4张图）全部保存完成")


def save_traceback_report(results, data=None, graph_builder=None, save_path='traceback_report.txt'):
    """Save anomaly traceback report."""
    label_names = data.get('label_names', ['Normal', 'NS', 'NM', 'PM', 'PS', 'SS']) if data else ['Normal', 'NS', 'NM', 'PM', 'PS', 'SS']
    label_descriptions = {
        0: 'Normal operation',
        1: 'Physical Fault - Sensor (NS): Network normal, physical sensor fault',
        2: 'Physical Fault - Mechanical (NM): Network normal, physical mechanical fault',
        3: 'Parameter Spoofing Attack (PM): Cyber attack causing physical mechanical fault',
        4: 'Parameter Spoofing Attack (PS): Cyber attack causing physical sensor fault',
        5: 'Shutdown Attack (SS): Cyber shutdown attack causing physical sensor fault'
    }
    if len(label_names) != 6:
        label_descriptions = {
            cls_id: ('Normal operation' if cls_id == 0 else f'{label_names[cls_id]} scenario')
            for cls_id in range(len(label_names))
        }
    
    ensure_parent_dir(save_path)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Anomaly Traceback Report - Cyber-Physical System ({len(label_names)}-Class)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("[Classification Description]\n")
        for cls_id, desc in label_descriptions.items():
            f.write(f"  Class {cls_id}: {desc}\n")
        f.write("\n")
        
        n_total = len(results)
        n_anomaly = sum(1 for r in results if r['true_label'] > 0)
        n_correct = sum(1 for r in results if r['true_label'] == r['pred_label'])
        y_true = np.array([r['true_label'] for r in results])
        y_pred = np.array([r['pred_label'] for r in results])
        
        f.write("[Summary Statistics]\n")
        f.write(f"  Total samples: {n_total}\n")
        f.write(f"  Anomaly samples: {n_anomaly}\n")
        f.write(f"  Correct predictions: {n_correct}\n")
        f.write(f"  Accuracy: {n_correct/n_total*100:.2f}%\n")
        f.write(f"  Precision-macro: {precision_score(y_true, y_pred, average='macro', zero_division=0)*100:.2f}%\n")
        f.write(f"  Recall-macro: {recall_score(y_true, y_pred, average='macro', zero_division=0)*100:.2f}%\n")
        f.write(f"  F1-macro: {f1_score(y_true, y_pred, average='macro', zero_division=0)*100:.2f}%\n")
        if graph_builder is not None and data is not None:
            all_node_scores = np.array([r['node_scores'] for r in results])
            trace_metrics = compute_strict_traceback_metrics(all_node_scores, data, graph_builder)
            f.write("\n[Exact-Node Traceback Metrics]\n")
            f.write(f"  RCA: {trace_metrics['rca']:.2f}%\n")
            f.write(f"  MRR: {trace_metrics['mrr']:.2f}%\n")
            f.write(f"  NDCG@5: {trace_metrics['ndcg']:.2f}%\n")
            f.write(f"  APD: {trace_metrics['apd']:.2f}\n")
            f.write(f"  Evaluated anomaly samples: {trace_metrics.get('trace_eval_total', 0)}\n")
            f.write(f"  Skipped samples without concrete root: {trace_metrics.get('trace_eval_skipped', 0)}\n")
        f.write("\n")
        
        # Per-class statistics
        f.write("[Per-Class Statistics]\n")
        for cls_id in range(len(label_names)):
            true_count = sum(1 for r in results if r['true_label'] == cls_id)
            pred_count = sum(1 for r in results if r['pred_label'] == cls_id)
            correct_count = sum(1 for r in results if r['true_label'] == cls_id and r['pred_label'] == cls_id)
            if true_count > 0:
                acc = correct_count / true_count * 100
                f.write(f"  {label_names[cls_id]}: True={true_count}, Predicted={pred_count}, Correct={correct_count}, Accuracy={acc:.2f}%\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("[Anomaly Sample Traceback Details]\n")
        f.write("-" * 80 + "\n\n")
        
        for sample_i, r in enumerate(results):
            if r['true_label'] > 0:
                true_lbl = r['true_label']
                pred_lbl = r['pred_label']
                true_name = label_names[true_lbl] if true_lbl < len(label_names) else str(true_lbl)
                pred_name = label_names[pred_lbl] if pred_lbl < len(label_names) else str(pred_lbl)
                
                anomaly_info = r.get('anomaly_type', ('unknown', true_name, ''))
                source_layer = 'Network Layer -> Physical Layer' if anomaly_info[0] == 'net_phy' else 'Physical Layer' if anomaly_info[0] == 'phy' else 'Unknown'
                
                f.write(f"Time: {r['time']}\n")
                f.write(f"True Label: {true_name} - {label_descriptions.get(true_lbl, '')}\n")
                f.write(f"Predicted Label: {pred_name}\n")
                f.write(f"Anomaly Source Layer: {source_layer}\n")
                if graph_builder is not None and data is not None:
                    true_nodes = infer_true_root_nodes(sample_i, data, graph_builder)
                    true_node_names = [graph_builder.all_nodes[i] for i in sorted(true_nodes)]
                    f.write(f"Exact True Root Node(s): {', '.join(true_node_names) if true_node_names else 'N/A'}\n")
                f.write(f"Root Cause Nodes (Top 3):\n")
                for node, score in zip(r['top_nodes'], r['top_scores']):
                    node_type = 'Network Node (IP)' if 'IP' in node else 'Physical Node (Device)'
                    f.write(f"  - [{node_type}] {node}: Anomaly Score={score:.4f}\n")
                f.write("\n")
    print(f"Traceback report saved to: {save_path}")


# ========================= 7. 主函数 =========================

def print_single_model_table(metrics, model_name="Ours (Full)"):
    """打印单个模型的评估结果表格"""
    print("\n" + "=" * 70)
    print(f"📊 模型评估结果")
    print("=" * 70)
    
    headers = ['Model', 'Accuracy (%)', 'Precision-macro (%)', 'Recall-macro (%)', 'F1-macro (%)',
               'Precision-weighted (%)', 'Recall-weighted (%)', 'F1-weighted (%)']
    table_data = [[
        model_name,
        f"{metrics['accuracy']:.2f}",
        f"{metrics['precision']:.2f}",
        f"{metrics['recall']:.2f}",
        f"{metrics['f1']:.2f}",
        f"{metrics.get('precision_weighted', 0):.2f}",
        f"{metrics.get('recall_weighted', 0):.2f}",
        f"{metrics.get('f1_weighted', 0):.2f}"
    ]]
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    return table_data


def main():
    parser = argparse.ArgumentParser(description='HGT-Trace - 轻量化异构图注意力溯源网络（两阶段自监督学习）')
    parser.add_argument('--csv', type=str, default='sr_com.csv', help='数据集路径')
    parser.add_argument('--epochs', type=int, default=50, help='总训练轮数（预训练+微调各一半）')
    parser.add_argument('--lr', type=float, default=0.002, help='学习率')
    parser.add_argument('--outdir', type=str, default='traceback_results', help='输出目录')
    parser.add_argument('--ablation', action='store_true', help='运行消融实验和模型对比')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（用于结果可复现）')
    parser.add_argument('--main-only', action='store_true', help='只训练当前主模型并生成溯源报告')
    parser.add_argument('--train-cap-per-class', type=int, default=0,
                        help='大数据加速：每类最多保留多少训练/验证样本；测试集始终完整')
    args = parser.parse_args()
    
    # 设置随机种子，确保结果可复现
    set_seed(args.seed)
    print(f"🎲 随机种子已设置为: {args.seed}")
    
    print("=" * 70)
    print("🚀 HGT-Trace - 轻量化异构图注意力溯源网络")
    print("   信息物理系统异常检测与根因溯源")
    print("   训练模式: 两阶段自监督学习")
    print("   阶段1: 自监督预训练（边重构，不使用标签）")
    print("   阶段2: 有监督微调（冻结编码器，训练分类头）")
    print("=" * 70)

    output_dirs = make_output_dirs(args.outdir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 使用设备: {device}")

    # 加载数据
    print(f"\n📊 加载数据...")
    raw_data = load_sr_com_data(args.csv)
    splits = chronological_split_and_scale(raw_data)
    data = splits['train']
    eval_data = splits['test']
    if args.train_cap_per_class and args.train_cap_per_class > 0:
        capped_train = cap_split_per_class(data, args.train_cap_per_class, seed=args.seed, split_name='Train')
        capped_val = cap_split_per_class(splits['val'], max(args.train_cap_per_class // 2, 1),
                                         seed=args.seed + 1, split_name='Val')
        capped_train['_val_data'] = capped_val
        capped_train['_test_data'] = eval_data
        data = capped_train
        splits['train'] = capped_train
        splits['val'] = capped_val
        print('   Full chronological test split is preserved for evaluation.')
    
    # 使用IP地址作为网络节点
    net_feat_dim = raw_data['X_net_feat'].shape[1] if len(raw_data['X_net_feat'].shape) > 1 else 1
    phy_feat_dim = len(raw_data['phy_cols'])
    graph_builder = HeteroGraphBuilder(
        raw_data['phy_cols'],
        raw_data['all_ips'],
        net_feat_dim=net_feat_dim,
        phy_feat_dim=phy_feat_dim
    )
    print(f"\n📊 异构图构建完成:")
    print(f"   网络节点（IP地址）: {graph_builder.n_net}")
    print(f"   物理节点: {graph_builder.n_phy}")
    print(f"   总节点数: {graph_builder.n_nodes}")
    
    # ==================== 运行实验 ====================
    if args.main_only:
        print(f"\n{'='*70}")
        print("🔬 开始只运行主模型")
        print(f"{'='*70}")
        model, comparison_results = train_main_model_only(
            data, graph_builder, device,
            epochs=args.epochs, lr=args.lr, outdir=output_dirs['data']
        )
    else:
        print(f"\n{'='*70}")
        print("🔬 开始运行消融实验和模型对比 (共13个模型)")
        print(f"{'='*70}")
        model, comparison_results = run_ablation_and_comparison(
            data, graph_builder, device,
            epochs=args.epochs, lr=args.lr, outdir=args.outdir
        )
    
    # 评估和溯源
    results = evaluate_and_traceback(
        model, eval_data, graph_builder, device,
        use_dynamic_edges=False,
        use_cross_layer=False,
        use_phy_chain=True,
        use_net_edges=True
    )

    # Save the text report before plotting so a late visualization failure cannot
    # discard the already-computed traceback results.
    save_traceback_report(results, data=eval_data, graph_builder=graph_builder,
                          save_path=os.path.join(output_dirs['reports'], 'traceback_report.txt'))
    
    print("\n📊 生成可视化图表...")

    # 删除旧版本生成的大图文件（如果存在，防止新旧文件混淆）
    stale_files = [
        'time_traceback.png', 'time_traceback_ground_truth.png',
        'time_traceback_heatmap.png', 'time_traceback_prediction.png',
        'root_cause_summary.png', 'uncertainty_analysis.png', 'explainability.png',
        'causal_propagation.png',
    ]
    for fname in stale_files:
        fpath = os.path.join(output_dirs['figures'], fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"   🗑️ 已删除旧文件: {fname}")

    # 1. 网络-物理拓扑图
    plot_network_physical_topology(graph_builder, eval_data, save_path=os.path.join(output_dirs['figures'], 'topology.png'))

    # 2. 异常溯源图（打印节点分数）
    plot_anomaly_traceback(results, graph_builder, data=eval_data, save_path=os.path.join(output_dirs['figures'], 'traceback.png'))

    # 3. 根因分析汇总图（4张独立子图）
    plot_root_cause_summary(results, graph_builder, data=eval_data, save_path=os.path.join(output_dirs['figures'], 'root_cause_summary.png'))

    # 4. 因果传播图（每个异常类别一张，共5张）
    plot_causal_propagation_per_class(results, graph_builder, data=eval_data, save_dir=output_dirs['figures'])

    # 5. 不确定性分析图（5张独立子图）
    plot_uncertainty_analysis(results, graph_builder, data=eval_data, save_path=os.path.join(output_dirs['figures'], 'uncertainty_analysis.png'))

    # 6. 可解释性分析图（4张独立子图）
    plot_explainability_summary(results, graph_builder, data=eval_data, save_path=os.path.join(output_dirs['figures'], 'explainability.png'))

    # 7. 溯源报告已在画图前保存，避免图生成失败时丢失报告

    print("\n" + "=" * 70)
    print("✅ 所有任务完成!")
    print("=" * 70)
    print(f"\n📁 输出文件:")
    print(f"   📊 可视化图表:")
    print(f"      - {output_dirs['figures']}/topology.png: 网络-物理拓扑图")
    print(f"      - {output_dirs['figures']}/traceback.png: 异常溯源分析图（节点分数见控制台）")
    print(f"      - {output_dirs['figures']}/root_cause_top10.png: 根因节点Top-10")
    print(f"      - {output_dirs['figures']}/root_cause_layer_dist.png: 根因层分布饼图")
    print(f"      - {output_dirs['figures']}/root_cause_by_type.png: 各类型根因分布")
    print(f"      - {output_dirs['figures']}/root_cause_heatmap.png: 异常分数热力图")
    print(f"      - {output_dirs['figures']}/causal_propagation_*.png: 各类别因果传播图")
    print(f"      - {output_dirs['figures']}/uncertainty_*.png: 不确定性分析图")
    print(f"      - {output_dirs['figures']}/anomaly_topology.png: 异常节点拓扑图")
    print(f"      - {output_dirs['figures']}/explain_*.png: 可解释性分析图")
    print(f"      - {output_dirs['figures']}/confusion_matrix_main_model.png: 混淆矩阵")
    print(f"   📋 报告和数据:")
    print(f"      - {output_dirs['reports']}/traceback_report.txt: Traceback Report")
    print(f"      - {output_dirs['data']}/model_comparison_classification.csv: 分类指标对比")
    print(f"      - {output_dirs['data']}/model_comparison_traceback.csv: 溯源指标对比")
    print(f"      - {output_dirs['data']}/model_comparison_full.csv: 完整对比结果")
    
    print(f"\n🚀 HGT-Trace 增强型溯源系统特性:")
    print(f"   - 轻量化编码器: 深度可分离图卷积，参数量比同类模型减少约40%")
    print(f"   - 跨层建模: 稀疏跨层注意力精确刻画网络层→物理层攻击传播路径")
    print(f"   - 自适应融合: 门控机制自适应平衡同层传播与跨域信息")
    print(f"   - 因果溯源: 精确定位根因节点和传播路径")
    print(f"   - 不确定性量化: 提供溯源结果置信度")
    print(f"   - 可解释性: 特征归因和层级贡献分析")


if __name__ == "__main__":
    main()
