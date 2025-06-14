import os
import pandas as pd
import numpy as np
import re

# ===========================
# 參數設定
# ===========================
TXT_FOLDER = 'data/raw/test_data/'  # 存放txt的位置
CSV_FILE = 'data/raw/test_info.csv'    # metadata的位置
OUTPUT_FILE = 'data/processed/test_features.csv'  # 輸出路徑

# ===========================
# 工具函數
# ===========================


def parse_cut_point(cut_point_str):
    if pd.isna(cut_point_str) or cut_point_str.strip() == "":
        return []
    numbers = re.findall(r'\d+', cut_point_str)
    return list(map(int, numbers))


def extract_segment_features(segment):
    """
    segment: numpy array, shape = (n_rows, 6)
    """
    features = {}
    cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    for i, col in enumerate(cols):
        data = segment[:, i]
        features[f'{col}_mean'] = np.mean(data)
        features[f'{col}_std'] = np.std(data)
        features[f'{col}_min'] = np.min(data)
        features[f'{col}_max'] = np.max(data)
        features[f'{col}_median'] = np.median(data)
    return features

# ===========================
# 主流程
# ===========================


if __name__ == "__main__":
    # 讀取 metadata
    info_df = pd.read_csv(CSV_FILE)

    all_features = []

    for idx, row in info_df.iterrows():
        unique_id = row['unique_id']
        cut_point_str = row['cut_point']
        txt_path = os.path.join(TXT_FOLDER, f'{unique_id}.txt')

        if not os.path.exists(txt_path):
            print(f"[Warning] {txt_path} 不存在，跳過")
            continue

        # 讀取 txt 檔案
        # shape = (n_samples, 6)
        data = pd.read_csv(txt_path, header=None, delim_whitespace=True).values

        # 解析 cut_point
        cut_points = parse_cut_point(cut_point_str)
        if len(cut_points) == 0:
            cut_points = [len(data) - 1]  # 如果沒有cut point，整段算一段

        # 切割段落
        segments = []
        start_idx = 0
        for end_idx in cut_points:
            segments.append(data[start_idx:end_idx+1])  # 包含 end_idx
            start_idx = end_idx + 1

        # 每個段落抽特徵
        segment_features = []
        for seg in segments:
            if len(seg) < 3:
                continue  # 太短的段直接跳過
            segment_features.append(extract_segment_features(seg))

        if len(segment_features) == 0:
            print(f"[Warning] {unique_id} 沒有有效段，跳過")
            continue

        # 聚合：取各段特徵的 mean
        segment_features_df = pd.DataFrame(segment_features)
        agg_features = segment_features_df.mean().to_dict()

        # 加入 unique_id
        agg_features['unique_id'] = unique_id

        all_features.append(agg_features)

    # 整理成 DataFrame
    features_df = pd.DataFrame(all_features)

    # 合併原本的 csv （除了 cut_point，不需要）
    final_df = features_df.merge(info_df.drop(
        columns=['cut_point']), on='unique_id', how='left')

    # 輸出
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"完成特徵抽取，輸出到 {OUTPUT_FILE}")
