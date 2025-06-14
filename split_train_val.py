import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ===========================
# 參數設定
# ===========================
INPUT_FILE = 'data/processed/GAN_gender_train_features.csv'  # 特徵檔案
TRAIN_OUTPUT = 'data/processed/GAN_gender_train.csv'
VAL_OUTPUT = 'data/processed/GAN_gender_val.csv'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ===========================
# 主流程
# ===========================


if __name__ == '__main__':
    df = pd.read_csv(INPUT_FILE)

    # 以 gender 作為 stratify
    if 'gender' not in df.columns:
        raise ValueError('找不到 gender 欄位，無法做 stratified split')

    train_df, val_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['gender']
    )

    os.makedirs(os.path.dirname(TRAIN_OUTPUT), exist_ok=True)

    train_df.to_csv(TRAIN_OUTPUT, index=False)
    val_df.to_csv(VAL_OUTPUT, index=False)

    print(f"資料切分完成，train: {len(train_df)}, val: {len(val_df)}")
