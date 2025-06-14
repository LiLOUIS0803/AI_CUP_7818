import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['Arial Unicode Ms']

TASKS = {
    "gender": 2,
    "hold racket handed": 2,
    "play years": 3,
    "level": 4
}

TASK_FEATURES = {
    "gender": ['acc_z_std', 'acc_z_min', 'gyro_z_mean', 'acc_x_mean', 'gyro_x_median', 'acc_z_median', 'acc_z_mean', 'acc_x_max', 'acc_y_max', 'gyro_z_max', 'gyro_y_min', 'acc_x_std', 'gyro_z_min', 'gyro_x_mean', 'acc_y_mean'],
    "hold racket handed": ['acc_x_mean', 'gyro_y_mean', 'gyro_y_max', 'gyro_y_min', 'gyro_y_median', 'gender', 'acc_y_median', 'gyro_z_min', 'acc_y_min', 'gyro_y_std', 'gyro_x_median', 'gyro_z_max', 'gyro_z_mean', 'acc_x_std', 'acc_x_max'],
    "play years": ['gyro_y_mean', 'gyro_y_max', 'gyro_z_mean', 'acc_z_median', 'gender', 'gyro_x_mean', 'acc_z_mean', 'gyro_y_std', 'acc_y_max', 'gyro_z_std', 'acc_x_median', 'acc_z_min', 'acc_y_median', 'gyro_y_median', 'gyro_x_std'],
    "level": ['gyro_x_median', 'acc_z_min', 'gyro_z_std', 'gyro_x_std', 'gyro_x_mean', 'acc_y_median', 'acc_y_max', 'acc_x_median', 'acc_z_median', 'gyro_y_median', 'gender', 'acc_z_mean', 'gyro_y_std', 'gyro_z_mean', 'gyro_z_median']

}

CATEGORICAL_COLS = 'mode'
LABEL_COLS = list(TASKS.keys())


def load_data(df, test=False, LABEL_COLS=LABEL_COLS):
    if not test:
        df["gender"] = 1 - (df["gender"] - 1)
        df["hold racket handed"] = 1 - (df["hold racket handed"] - 1)
        df["level"] = df["level"] - 2
    # 特徵與標籤
        features = df.drop(columns=["unique_id", "player_id"] + LABEL_COLS)
        return features, df[LABEL_COLS]
    else:
        features = df.drop(columns=["unique_id"])
        return features

# 計算 ROC AUC


def compute_multiclass_auc(y_true, y_score):
    y_true_onehot = pd.get_dummies(y_true)
    return roc_auc_score(y_true_onehot, y_score, average='micro')

# 讀取資料


df_train = pd.read_csv("data/processed/GAN_gender_train.csv")
df_val = pd.read_csv("data/processed/GAN_gender_val.csv")
X_train_1, y_train_1 = load_data(df_train.copy())
X_train_2, y_train_2 = load_data(
    df_train.copy(), LABEL_COLS=["hold racket handed",  "play years", "level"])
X_val_1, y_val_1 = load_data(df_val.copy())
X_val_2, y_val_2 = load_data(
    df_val.copy(), LABEL_COLS=["hold racket handed",  "play years", "level"])

N_SPLITS = 7
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# 訓練模型與預測
models = {}

scores = {}

for task in TASKS.keys():
    print(task)
    num_class = TASKS[task]
    params = {
        'objective': 'multi:softprob' if num_class > 2 else 'binary:logistic',
        'eval_metric': 'auc',
        'num_class': num_class if num_class > 2 else None,
        'verbosity': 0,
        'max_depth': 4,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0,
        'min_child_weight': 5,
        'lambda': 1.0,
        'alpha': 0.0,
        'scale_pos_weight': 1.5
    }
    X_train = None
    y_train = None
    X_val = None
    y_val = None
    if task == "gender":
        X_train = X_train_1
        y_train = y_train_1
        X_val = X_val_1
        y_val = y_val_1
    else:
        X_train = X_train_2
        y_train = y_train_2
        X_val = X_val_2
        y_val = y_val_2
    X_train = X_train[TASK_FEATURES[task]]
    X_val = X_val[TASK_FEATURES[task]]
    X = X_train.values
    y = y_train[task].values

    fold_preds = np.zeros(
        (X_val.shape[0], num_class)) if num_class > 2 else np.zeros(X_val.shape[0])
    aucs = []
    fold_aucs = []
    times = 1
    for train_idx, _ in skf.split(X, y):
        print("fold:", times)
        dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
        dval_cv = xgb.DMatrix(X_val)

        bst = xgb.train(params, dtrain, num_boost_round=100)
        preds = bst.predict(dval_cv)
        if num_class == 2:
            auc = roc_auc_score(y_val[task], preds)
        else:
            auc = compute_multiclass_auc(y_val[task], preds)
        print("auc:", auc)
        fold_aucs.append(auc)

        fold_preds += preds / N_SPLITS
        models[task] = bst  # 最後一輪的模型可作為 test 使用
        times += 1
    if fold_preds.ndim == 1 or fold_preds.shape[1] == 1:
        auc = roc_auc_score(y_val[task], fold_preds)
    else:
        auc = compute_multiclass_auc(y_val[task], fold_preds)
    scores[task] = {
        "final_auc": auc,
        "fold_aucs": fold_aucs
    }


# 最終平均分數
final_score = sum([v["final_auc"] for v in scores.values()]) / len(scores)

print("各任務 ROC AUC:", scores)
print("總平均分數:", final_score)


df_test = pd.read_csv("data/processed/test_features.csv")
X_test = load_data(df_test, test=True)

output = pd.DataFrame()
output['unique_id'] = df_test['unique_id']

for task in TASKS.keys():
    if task is not "gender":
        X_test["gender"] = np.where(output["gender"] > 0.5, 1, 0)
    X_test_ = X_test
    X_test_ = X_test[TASK_FEATURES[task]]
    dtest = xgb.DMatrix(X_test_)
    model = models[task]
    y_proba = model.predict(dtest)

    # 預測結果寫入 output dataframe
    if TASKS[task] == 2:
        output[task] = y_proba
    else:
        for i in range(TASKS[task]):
            output[f"{task}_{i + (2 if task == 'level' else 0)}"] = y_proba[:,
                                                                            i].astype(float)

output.to_csv("result/xgboost_submission.csv",
              index=False, float_format='%.10f')


df_test = pd.read_csv("data/processed/train_features.csv")
X_test = load_data(df_test, test=True)


# N 圖像化: 各任務總 AUC
tasks = list(scores.keys())
auc_values = [scores[t]["final_auc"] for t in tasks]

plt.figure(figsize=(8, 6))
bars = plt.bar(tasks, auc_values)
plt.ylim(0, 1)
plt.ylabel("AUC")
plt.title("各任務最終 AUC 分數")
for bar, auc in zip(bars, auc_values):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.01, f"{auc:.4f}", ha='center')
plt.savefig('output/task_final.png')

# N 繪製總平均 AUC
plt.figure(figsize=(4, 4))
plt.bar(['平均 AUC'], [final_score])
plt.ylim(0, 1)
plt.text(0, final_score + 0.01, f"{final_score:.4f}", ha='center')
plt.title("總平均 AUC 分數")
plt.savefig('output/avg.png')

# N 繪製每個任務的 cross-validation AUC 分佈
for task in tasks:
    fold_aucs = scores[task]["fold_aucs"]
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(fold_aucs)+1), fold_aucs, marker='o')
    plt.ylim(0, 1)
    plt.title(f"{task} 每個 Fold 的 AUC")
    plt.xlabel("Fold")
    plt.ylabel("AUC")
    for i, auc in enumerate(fold_aucs):
        plt.text(i+1, auc + 0.01, f"{auc:.4f}", ha='center')
    plt.savefig(f'output/{task}_auc.png')
