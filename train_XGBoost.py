import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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


def train_model(params, skf, models, X, y, X_val):
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

        fold_preds = preds
        models[task] = bst  # 最後一輪的模型可作為 test 使用
        times += 1
    return aucs, fold_aucs, fold_preds, models


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


models = {}
scores = {}

models_all = {}
scores_all = {}

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

    X_train_all = X_train
    X_val_all = X_val

    X_train = X_train[TASK_FEATURES[task]]
    X_val = X_val[TASK_FEATURES[task]]

    X = X_train.values
    X_all = X_train_all.values
    y = y_train[task].values

    aucs, fold_aucs, fold_preds, models = train_model(
        params, skf, models, X, y, X_val)
    aucs_all, fold_aucs_all, fold_preds_all, models_all = train_model(
        params, skf, models_all, X_all, y, X_val_all)

    if fold_preds.ndim == 1 or fold_preds.shape[1] == 1:
        auc = roc_auc_score(y_val[task], fold_preds)
        auc_all = roc_auc_score(y_val[task], fold_preds_all)
    else:
        auc = compute_multiclass_auc(y_val[task], fold_preds)
        auc_all = compute_multiclass_auc(y_val[task], fold_preds_all
                                         )
    scores[task] = {
        "final_auc": auc,
        "fold_aucs": fold_aucs
    }
    scores_all[task] = {
        "final_auc": auc_all,
        "fold_aucs": fold_aucs_all
    }

    # N 計算與繪製混淆矩陣
    if num_class == 2:
        y_pred_label = (fold_preds > 0.5).astype(int)
        y_pred_label_all = (fold_preds_all > 0.5).astype(int)

    else:
        y_pred_label = np.argmax(fold_preds, axis=1)
        y_pred_label_all = np.argmax(fold_preds_all, axis=1)

    cm = confusion_matrix(y_val[task], y_pred_label)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"{task} 混淆矩陣_特徵選取")
    plt.savefig(f'output/{task}_confusion_matrix.png')
    plt.close()

    cm = confusion_matrix(y_val[task], y_pred_label_all)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"{task} 混淆矩陣")
    plt.savefig(f'output/{task}_all_confusion_matrix.png')
    plt.close()


# 最終平均分數
final_score = sum([v["final_auc"] for v in scores.values()]) / len(scores)

print("各任務 ROC AUC:", scores)
print("總平均分數:", final_score)

# N 圖像化: 各任務總 AUC
tasks = list(scores.keys())

auc_values = [scores[t]["final_auc"] for t in tasks]
auc_all_values = [scores_all[t]["final_auc"] for t in tasks]

final_score_all = sum(auc_all_values) / len(auc_all_values)

auc_values.append(final_score)
auc_all_values.append(final_score_all)

plt.figure(figsize=(10, 6))
x = np.arange(len(tasks) + 1)
width = 0.35

bars1 = plt.bar(x - width/2, auc_values, width, label='特徵選取')
bars2 = plt.bar(x + width/2, auc_all_values, width, label='無特徵選取')

plt.xticks(x, tasks + ['平均 AUC'])
plt.ylim(0, 1)
plt.ylabel("AUC")
plt.title("各任務最終 AUC 分數比較")
plt.legend()

for bar in bars1:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{bar.get_height():.4f}", ha='center')
for bar in bars2:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{bar.get_height():.4f}", ha='center')

plt.savefig('output/task_final_compare.png')
plt.close()

# N 繪製每個任務的 cross-validation AUC 分佈比較
for task in tasks:
    fold_aucs = scores[task]["fold_aucs"]
    fold_aucs_all = scores_all[task]["fold_aucs"]

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(fold_aucs)+1), fold_aucs, marker='o', label='特徵選取')
    plt.plot(range(1, len(fold_aucs_all)+1),
             fold_aucs_all, marker='s', label='無特徵選取')
    plt.ylim(0, 1)
    plt.title(f"{task} 每個 Fold 的 AUC 比較")
    plt.xlabel("Fold")
    plt.ylabel("AUC")
    plt.legend()
    plt.savefig(f'output/{task}_auc_compare.png')
    plt.close()

df_test = pd.read_csv("data/processed/test_features.csv")
X_test = load_data(df_test, test=True)
output = pd.DataFrame()
output['unique_id'] = df_test['unique_id']

for task in TASKS.keys():
    if task != "gender":
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
