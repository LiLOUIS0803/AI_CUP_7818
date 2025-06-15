AI_CUP_7818
===
**環境安裝**

使用以下指令安裝所需套件

`pip install -r requirements.txt`

---
**資料前處理**

需要將資料放在data/raw，最後處理過後的資料會在data/processed

`python process_data.py`

---
**模型訓練和推論**

需要將資料放在data/processed，最後的資料和圖片分別會在result/xgboost_submission.csv和output

`python train_XGBoost.py`
