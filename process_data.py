from src.preprocess import preprocess
from src.tabular_GAN import gan_data
from src.split_train_val import split_train_val

if __name__ == '__main__':
    train_TXT_FOLDER = 'data/raw/train_data/'  # 存放txt的位置
    train_CSV_FILE = 'data/raw/train_info.csv'    # metadata的位置
    train_OUTPUT_FILE = 'data/processed/train_features.csv'  # 輸出路徑

    test_TXT_FOLDER = 'data/raw/test_data/'  # 存放txt的位置
    test_CSV_FILE = 'data/raw/test_info.csv'    # metadata的位置
    test_OUTPUT_FILE = 'data/processed/test_features.csv'  # 輸出路徑

    preprocess(train_TXT_FOLDER, train_CSV_FILE, train_OUTPUT_FILE)
    preprocess(test_TXT_FOLDER, test_CSV_FILE, test_OUTPUT_FILE)

    gan_OUTPUT_FILE = 'data/processed/GAN_gender_train_features.csv'
    gan_data(train_OUTPUT_FILE, gan_OUTPUT_FILE)

    train_OUTPUT = 'data/processed/GAN_gender_train.csv'
    val_OUTPUT = 'data/processed/GAN_gender_val.csv'
    split_train_val(gan_OUTPUT_FILE, train_OUTPUT, val_OUTPUT)
