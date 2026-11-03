"""Data utilities and the nine comparison-model adapters used in the study."""
import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
ROOT_REFERENCE_POLICY = 'igcps-ns-two-pressure-switches-v2-20260907'
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

    # Controller-state fields are control-plane telemetry rather than process
    # variables. Keep a separate view for typed-token studies and retain them
    # in the information-layer feature vector used by the established model.
    control_cols = [
        c for c in ['phy_Controller status', 'phy_Controller status value']
        if c in df.columns
    ]
    phy_cols = [c for c in phy_cols if c not in control_cols]
    print(f"📊 独立控制上下文 ({len(control_cols)}): {control_cols}")

    # Identify endpoint columns by exact normalized names first. Aggregate
    # fields such as cyber_unique_src_ip_count must remain numeric features.
    normalized_columns = {
        c: c.lower().strip().replace(' ', '_').replace('-', '_')
        for c in df.columns
    }
    src_exact = {'source_ip', 'src_ip', 'sourceip'}
    dst_exact = {'destination_ip', 'dest_ip', 'dst_ip', 'destinationip', 'destip'}
    src_ip_col = next(
        (c for c, normalized in normalized_columns.items() if normalized in src_exact),
        None,
    )
    dst_ip_col = next(
        (c for c, normalized in normalized_columns.items() if normalized in dst_exact),
        None,
    )

    # If standard names are absent, fall back to conservative substring matching.
    if src_ip_col is None or dst_ip_col is None:
        for c in df.columns:
            c_lower = c.lower()
            if src_ip_col is None and 'ip' in c_lower and (
                'src' in c_lower or 'source' in c_lower
            ) and 'count' not in c_lower:
                src_ip_col = c
            elif dst_ip_col is None and 'ip' in c_lower and (
                'dst' in c_lower or 'dest' in c_lower
            ) and 'count' not in c_lower:
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

    control_data = df[control_cols].copy() if control_cols else pd.DataFrame(index=df.index)
    for c in control_cols:
        control_data[c] = pd.to_numeric(control_data[c], errors='coerce').fillna(0)
    X_control = (
        control_data.values.astype(np.float32)
        if control_cols else np.zeros((len(df), 0), dtype=np.float32)
    )

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
        'X_control': X_control,
        'X_net_feat': X_net_feat,  # 网络特征（用于边/节点特征）
        'src_ips': src_ips,         # 每个样本的源IP
        'dst_ips': dst_ips,         # 每个样本的目标IP
        'all_ips': all_ips,         # 所有唯一IP（网络节点）
        'ip_to_idx': ip_to_idx,     # IP到索引的映射
        'y': y,
        'labels_raw': labels,
        'anomaly_types': anomaly_types,
        'phy_cols': phy_cols,
        'control_cols': control_cols,
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
        'times', 'X_phy', 'X_control', 'X_net_feat', 'src_ips', 'dst_ips',
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
    scaler_control = None
    if train_data.get('X_control') is not None and train_data['X_control'].shape[1] > 0:
        scaler_control = StandardScaler().fit(train_data['X_control'])
    for split in (train_data, val_data, test_data):
        split['X_phy'] = scaler_phy.transform(split['X_phy']).astype(np.float32)
        split['X_net_feat'] = scaler_net.transform(split['X_net_feat']).astype(np.float32)
        if scaler_control is not None:
            split['X_control'] = scaler_control.transform(split['X_control']).astype(np.float32)

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


def _phy_indices_by_columns(graph_builder, column_names):
    """Return physical-node indices for exact released process variables."""
    wanted = {
        str(name).strip().lower().replace('_', '') for name in column_names
    }
    matches = set()
    for local_idx, column in enumerate(graph_builder.phy_cols):
        normalized = str(column).strip().lower().replace('_', '')
        if normalized in wanted:
            matches.add(graph_builder.n_net + local_idx)
    return matches


IGCPS_NS_ROOT_COLUMNS = (
    'phy_Switch status of high pressure switch',
    'phy_Switch status of sub-high pressure switch',
)


def infer_true_root_nodes(sample_idx, data, graph_builder):
    """Infer concrete root-cause node labels for strict traceback evaluation.

    Priority:
    1) Explicit root-cause columns if a future dataset provides them.
    2) Dataset-specific roots use independently documented devices/variables.
    3) IGCPS physical faults use small process-specific physical-node sets.
    4) Generic network-origin data use Source IP only when no stricter mapping exists.
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

    # TE-CUP-SEC publishes exact affected variables only for Attacks 4--8.
    # Attacks 1--3 alter control setpoints that are absent from the released
    # 53 process variables, so strict exact-node evaluation must skip them.
    if label_upper in {'ATTACK1', 'ATTACK2', 'ATTACK3'}:
        return set()
    if label_upper == 'ATTACK4':
        return _phy_indices_by_columns(graph_builder, ['phy_xmv3', 'phy_xmv4'])
    if label_upper == 'ATTACK5':
        return _phy_indices_by_columns(graph_builder, ['phy_xmv10'])
    if label_upper == 'ATTACK6':
        return _phy_indices_by_columns(graph_builder, ['phy_xmv11'])
    if label_upper == 'ATTACK7':
        return _phy_indices_by_columns(graph_builder, ['phy_xmeas15'])
    if label_upper == 'ATTACK8':
        separator_plc = graph_builder.ip_to_idx.get('192.168.6.2')
        return {separator_plc} if separator_plc is not None else set()

    if label_upper == 'NS':
        # Author-confirmed scenario: only these two pressure switches are affected.
        # A broad switch/status match also includes the unrelated heater switch.
        nodes = _phy_indices_by_columns(graph_builder, IGCPS_NS_ROOT_COLUMNS)
        if len(nodes) != len(IGCPS_NS_ROOT_COLUMNS):
            raise ValueError(
                'NS root references require both high- and sub-high-pressure '
                'switch columns; no substitute node is a valid reference.'
            )
        return nodes

    if label_upper in {'NM'}:
        nodes = _phy_indices_by_keywords(graph_builder, [
            ['medium', 'heater'], ['medium', 'temperature'], ['medium', 'outlet']
        ])
        return nodes or ({graph_builder.n_net} if graph_builder.n_phy else set())

    # The IGCPS attack scripts target PLC 10.0.6.30:102, and the captured
    # initiating S7/TCP flow is consistently 10.0.6.118 -> 10.0.6.30.  The
    # attacker endpoint, not whichever direction an arbitrary packet happens
    # to use, is the concrete root for PM/PS/SS.
    if label_upper in {'PM', 'PS', 'SS'}:
        attacker_node = graph_builder.ip_to_idx.get('10.0.6.118')
        if attacker_node is not None:
            return {attacker_node}

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


class HeteroGraphBuilder:
    """构建异构图：网络节点（IP地址） + 物理节点 + 边

    网络节点：Source IP 和 Destination IP 构成的唯一IP地址
    物理节点：按照燃气系统数据流顺序进行链式连接
    """

    def __init__(self, phy_cols, all_ips, net_feat_dim=1, phy_feat_dim=1,
                 preserve_net_features=False):
        # 保持phy_cols的原始顺序（表格中的顺序就是燃气流经顺序）
        self.phy_cols = phy_cols
        self.all_ips = all_ips  # 所有唯一IP地址
        self.ip_to_idx = {ip: i for i, ip in enumerate(all_ips)}
        self.net_feat_dim = net_feat_dim
        self.phy_feat_dim = phy_feat_dim
        self.preserve_net_features = bool(preserve_net_features)
        self.net_raw_offset = 6

        # 节点名称
        self.phy_nodes = [f"Phy_{c.replace('phy_', '')}" for c in phy_cols]
        self.net_nodes = [f"IP_{ip}" for ip in all_ips]  # 网络节点是IP地址
        self.all_nodes = self.net_nodes + self.phy_nodes

        self.n_net = len(self.net_nodes)
        self.n_phy = len(self.phy_nodes)
        self.n_nodes = self.n_net + self.n_phy

        # The legacy layout keeps four summary statistics.  The revised
        # layout also preserves every flow feature after the endpoint-role
        # fields, avoiding an information bottleneck before graph encoding.
        required_net_dim = (
            self.net_raw_offset + net_feat_dim
            if self.preserve_net_features else net_feat_dim
        )
        self.node_feat_dim = max(required_net_dim, phy_feat_dim, 8)

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
            if self.preserve_net_features:
                raw_end = self.net_raw_offset + min(len(net_feat_arr), self.net_feat_dim)
                net_feats[src_idx, self.net_raw_offset:raw_end] = net_feat_arr[:raw_end - self.net_raw_offset]

        if dst_ip in self.ip_to_idx:
            dst_idx = self.ip_to_idx[dst_ip]
            net_feats[dst_idx, :4] = base_net_feat[:4]
            net_feats[dst_idx, 4] = -1.0  # 目的IP标记
            net_feats[dst_idx, 5] = dst_idx / max(self.n_net, 1)  # 位置编码
            if self.preserve_net_features:
                raw_end = self.net_raw_offset + min(len(net_feat_arr), self.net_feat_dim)
                net_feats[dst_idx, self.net_raw_offset:raw_end] = net_feat_arr[:raw_end - self.net_raw_offset]

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
            if self.preserve_net_features:
                raw_end = self.net_raw_offset + min(len(net_feat_arr), self.net_feat_dim)
                net_feats[src_idx, self.net_raw_offset:raw_end] = net_feat_arr[:raw_end - self.net_raw_offset]

        if dst_ip in self.ip_to_idx:
            dst_idx = self.ip_to_idx[dst_ip]
            net_feats[dst_idx, :4] = base_net_feat
            net_feats[dst_idx, 4] = -1.0
            net_feats[dst_idx, 5] = dst_idx / max(self.n_net, 1)
            if self.preserve_net_features:
                raw_end = self.net_raw_offset + min(len(net_feat_arr), self.net_feat_dim)
                net_feats[dst_idx, self.net_raw_offset:raw_end] = net_feat_arr[:raw_end - self.net_raw_offset]

        # 时间特征：过去K步中该IP的活跃次数
        if src_ips_window is not None and dst_ips_window is not None:
            for k in range(K - 1):  # 不含当前时刻
                s_ip = src_ips_window[k]
                d_ip = dst_ips_window[k]
                if s_ip in self.ip_to_idx:
                    net_feats[self.ip_to_idx[s_ip], self.node_feat_dim] += 1.0 / K
                if d_ip in self.ip_to_idx:
                    net_feats[self.ip_to_idx[d_ip], self.node_feat_dim + 1] += 1.0 / K

        if self.preserve_net_features and K >= 2:
            previous = np.asarray(X_net_feat_window[-2]).flatten()
            width = min(len(previous), len(net_feat_arr), self.net_feat_dim)
            if width > 0:
                delta = net_feat_arr[:width] - previous[:width]
                active_nodes = []
                if src_ip in self.ip_to_idx:
                    active_nodes.append(self.ip_to_idx[src_ip])
                if dst_ip in self.ip_to_idx:
                    active_nodes.append(self.ip_to_idx[dst_ip])
                for node_idx in set(active_nodes):
                    net_feats[node_idx, self.node_feat_dim + 2] = np.mean(delta)
                    net_feats[node_idx, self.node_feat_dim + 3] = np.std(delta)

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
                if getattr(encoder, 'detection_encoder', None) is not None:
                    detection_z = encoder.encode_detection(x, adj_tensor)
                    detection_recon = encoder.decode_detection(detection_z)
                    detection_edge_loss = F.binary_cross_entropy(
                        detection_recon, adj_tensor
                    )
                    edge_loss = 0.5 * (edge_loss + detection_edge_loss)
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
                        verbose=True, model_name="Model",
                        encoder_lr_scale=0.1, class_weight_power=1.0,
                        apply_class_boosts=True, oversample=True,
                        label_smoothing=0.0, focal_gamma=0.0,
                        early_stopping_patience=15,
                        temporal_recency_strength=0.0, selection_data=None,
                        restore_best=True, distillation_teacher=None,
                        distillation_logits=None,
                        distillation_alpha=0.0, distillation_temperature=2.0,
                        joint_root_head=None, joint_root_weight=0.0,
                        joint_root_stride=1):
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
            or 'causal_traceback' in name or 'feature_enhance' in name
            or 'readout_norm' in name
        ):
            classifier_params.append(param)
        else:
            encoder_params.append(param)

    if verbose:
        print(f"     📊 编码器参数: {sum(p.numel() for p in encoder_params):,}")
        print(f"     📊 分类器参数: {sum(p.numel() for p in classifier_params):,}")

    # 计算类别权重（激进式类别平衡）
    val_data = selection_data if selection_data is not None else data.get('_val_data', data)
    y = data['y']
    class_counts = np.bincount(y)

    # The exponent makes class balancing tunable without changing the split.
    class_weights = (1.0 / (class_counts + 1e-6)) ** float(class_weight_power)
    class_weights = class_weights / class_weights.min()  # 归一化到最小权重=1

    # 特别增加NM类别权重（类别2），因为它是边界类别，容易混淆
    # NM = 物理故障（网络正常+物理中等故障）
    label_names = data.get('label_names', ['Normal', 'NS', 'NM', 'PM', 'PS', 'SS'])
    nm_idx = None
    if apply_class_boosts and 'NM' in label_names:
        nm_idx = label_names.index('NM')
        if nm_idx < len(class_weights):
            class_weights[nm_idx] *= 1.5  # 适度增加NM权重
            if verbose:
                print(f"     ⚠️ NM类别权重增强 (idx={nm_idx}): 权重={class_weights[nm_idx]:.3f}")

    # PS类别也是少数类，需要增强
    if apply_class_boosts and 'PS' in label_names:
        ps_idx = label_names.index('PS')
        if ps_idx < len(class_weights):
            class_weights[ps_idx] *= 2.0

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    if verbose:
        print(f"     📊 类别权重: {class_weights.cpu().numpy()}")

    # 差异化学习率优化器
    optimizer_groups = [
        {'params': encoder_params, 'lr': lr * float(encoder_lr_scale)},
        {'params': classifier_params, 'lr': lr}      # 分类器用正常学习率
    ]
    root_weight = max(0.0, float(joint_root_weight))
    root_stride = max(1, int(joint_root_stride))
    if joint_root_head is not None and root_weight > 0:
        joint_root_head.to(device)
        optimizer_groups.append({'params': list(joint_root_head.parameters()), 'lr': lr})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    kd_alpha = min(max(float(distillation_alpha), 0.0), 1.0)
    kd_temperature = max(float(distillation_temperature), 1e-3)
    cached_teacher_logits = None
    if distillation_logits is not None and kd_alpha > 0:
        cached_teacher_logits = torch.as_tensor(
            distillation_logits, dtype=torch.float32, device=device
        )
        if cached_teacher_logits.ndim != 2 or cached_teacher_logits.shape[0] != len(y):
            raise ValueError(
                'distillation_logits must have shape [number of training samples, classes].'
            )
        distillation_teacher = None
    elif distillation_teacher is not None and kd_alpha > 0:
        distillation_teacher.to(device)
        distillation_teacher.eval()
        for param in distillation_teacher.parameters():
            param.requires_grad_(False)
    else:
        distillation_teacher = None
        kd_alpha = 0.0

    if verbose and kd_alpha > 0 and (
        distillation_teacher is not None or cached_teacher_logits is not None
    ):
        print(
            f"     Distillation: alpha={kd_alpha:.2f}, "
            f"temperature={kd_temperature:.2f}, "
            f"source={'cached logits' if cached_teacher_logits is not None else 'live teacher'}"
        )
    if verbose and joint_root_head is not None and root_weight > 0:
        print(
            f"     Joint root supervision: weight={root_weight:.3f}, "
            f"stride={root_stride}"
        )

    X_phy = data['X_phy']
    X_net_feat = data['X_net_feat']
    src_ips = data['src_ips']
    dst_ips = data['dst_ips']
    n_samples = len(y)

    # Weight recent observations more strongly within each class without using
    # validation/test labels. A value of zero preserves the original objective.
    recency_weights = np.ones(n_samples, dtype=np.float32)
    recency_strength = max(0.0, float(temporal_recency_strength))
    if recency_strength > 0:
        for cls_id in range(len(class_counts)):
            cls_indices = np.where(y == cls_id)[0]
            if len(cls_indices) <= 1:
                continue
            progress = np.linspace(0.0, 1.0, len(cls_indices), dtype=np.float32)
            cls_weights = np.exp(recency_strength * (progress - 1.0))
            cls_weights /= max(float(cls_weights.mean()), 1e-8)
            recency_weights[cls_indices] = cls_weights
        if verbose:
            print(
                f"     🕒 Temporal recency strength={recency_strength:.2f}, "
                f"sample-weight range={recency_weights.min():.3f}--{recency_weights.max():.3f}"
            )

    # 过采样少数类索引
    oversampled_indices = []
    minority_target = 200 if n_samples < 1000 else 1000
    nm_target = 100 if n_samples < 1000 else 500
    if oversample:
        for cls_id in range(len(class_counts)):
            cls_indices = np.where(y == cls_id)[0]

            # Preserve the legacy NM cap only when the legacy manual boosts are enabled.
            if apply_class_boosts and nm_idx is not None and cls_id == nm_idx:
                repeat_times = max(1, nm_target // len(cls_indices))
                oversampled_indices.extend(np.repeat(cls_indices, repeat_times).tolist())
            elif class_counts[cls_id] < minority_target:
                repeat_times = max(1, minority_target // len(cls_indices))
                oversampled_indices.extend(np.repeat(cls_indices, repeat_times).tolist())
            else:
                oversampled_indices.extend(cls_indices.tolist())
    else:
        oversampled_indices.extend(np.arange(n_samples).tolist())
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
    best_root_state = None
    best_epoch = 0
    patience_counter = 0
    patience = int(early_stopping_patience)
    finetune_history = []

    if verbose:
        print(f"\n🔧 [{model_name}] 阶段2: 有监督微调 (epochs={epochs})...")

    # Traceback supervision is trained after classification fine-tuning. Keeping it
    # out of this loop preserves the classifier's early-stopping trajectory.
    traceback_loss_weight = 0.0
    traceback_loss_stride = 4 if False else 8
    show_large_progress = len(oversampled_indices) >= 100000
    progress_step = max(batch_size, (max(len(oversampled_indices) // 10, batch_size) // batch_size) * batch_size)

    for epoch in range(epochs):
        model.train()
        if joint_root_head is not None and root_weight > 0:
            joint_root_head.train()
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

                ce_loss = F.cross_entropy(
                    logits, label, weight=class_weights,
                    label_smoothing=float(label_smoothing),
                    # Each graph is evaluated separately inside the batch loop.
                    # ``mean`` would divide by that graph's target-class weight
                    # and cancel class weighting for a one-sample tensor.
                    reduction='none',
                )
                gamma = max(0.0, float(focal_gamma))
                if gamma > 0:
                    target_prob = torch.softmax(logits, dim=-1).gather(
                        1, label.view(-1, 1)
                    ).squeeze(1).clamp(1e-6, 1.0 - 1e-6)
                    cls_loss = (((1.0 - target_prob) ** gamma) * ce_loss).sum()
                else:
                    cls_loss = ce_loss.sum()
                if distillation_teacher is not None or cached_teacher_logits is not None:
                    if cached_teacher_logits is not None:
                        teacher_logits = cached_teacher_logits[t].unsqueeze(0)
                    else:
                        with torch.no_grad():
                            teacher_out = distillation_teacher(
                                x, adj_tensor, return_traceback=False
                            )
                            teacher_logits = (
                                teacher_out['anomaly_logits']
                                if isinstance(teacher_out, dict) else teacher_out
                            )
                            if teacher_logits.dim() == 1:
                                teacher_logits = teacher_logits.unsqueeze(0)
                    kd_loss = F.kl_div(
                        F.log_softmax(logits / kd_temperature, dim=-1),
                        F.softmax(teacher_logits / kd_temperature, dim=-1),
                        reduction='batchmean',
                    ) * (kd_temperature ** 2)
                    sample_loss = (
                        (1.0 - kd_alpha) * cls_loss + kd_alpha * kd_loss
                    ) * float(recency_weights[t])
                else:
                    sample_loss = cls_loss * float(recency_weights[t])

                needs_joint_root = (
                    joint_root_head is not None
                    and root_weight > 0
                    and int(y[t]) > 0
                    and ((total_samples + epoch) % root_stride == 0)
                    and isinstance(out, dict)
                    and 'embeddings' in out
                )
                if needs_joint_root:
                    root_target = build_root_target(t, data, graph_builder, device)
                    if root_target is not None:
                        root_scores = joint_root_head(out['embeddings'])
                        root_scores = root_scores[:graph_builder.n_nodes].clamp(1e-6, 1 - 1e-6)
                        root_target = root_target[:root_scores.shape[0]]
                        root_weights = torch.ones_like(root_target)
                        root_weights[root_target > 0] = max(
                            5.0,
                            graph_builder.n_nodes / max(root_target.sum().item(), 1.0),
                        )
                        root_loss = F.binary_cross_entropy(
                            root_scores, root_target, weight=root_weights
                        )
                        sample_loss = sample_loss + root_weight * root_loss

                if needs_traceback_loss and isinstance(out, dict):
                    root_target = build_root_target(t, data, graph_builder, device)
                    root_scores = None
                    if False and 'embeddings' in out:
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
            clip_params = list(model.parameters())
            if joint_root_head is not None and root_weight > 0:
                clip_params.extend(joint_root_head.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
            optimizer.step()
            total_loss += batch_loss.item() * len(batch_indices)

            # 清理GPU缓存
            if device.type == 'cuda' and batch_start % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

        scheduler.step()
        avg_loss = total_loss / len(oversampled_indices)

        # 每个epoch结束后在原始数据上评估
        model.eval()
        if joint_root_head is not None and root_weight > 0:
            joint_root_head.eval()
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
        finetune_history.append({
            'epoch': epoch + 1,
            'loss': float(avg_loss),
            'selection_accuracy': float(eval_acc),
            'selection_f1': float(eval_f1),
        })
        model.train()
        if joint_root_head is not None and root_weight > 0:
            joint_root_head.train()

        # 使用F1作为主要指标（更关注少数类）
        if eval_f1 > best_f1 or (eval_f1 == best_f1 and eval_acc > best_acc):
            best_f1 = eval_f1
            best_acc = eval_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if joint_root_head is not None and root_weight > 0:
                best_root_state = {
                    k: v.cpu().clone() for k, v in joint_root_head.state_dict().items()
                }
            best_epoch = epoch + 1
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
    if restore_best and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        if best_root_state is not None:
            joint_root_head.load_state_dict(
                {k: v.to(device) for k, v in best_root_state.items()}
            )

    model.finetune_history = finetune_history
    model.finetune_best_epoch = best_epoch

    if verbose:
        if restore_best:
            print(
                f"     ✅ 微调完成! Best epoch: {best_epoch}, "
                f"Best Acc: {best_acc:.2f}%, Best F1: {best_f1:.2f}%"
            )
        else:
            print(f"     ✅ 固定轮数重训完成! Retained epoch: {len(finetune_history)}")

    return model
