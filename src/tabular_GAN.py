import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata


def gan_data(data_file, output_path):
    # 讀取資料
    df = pd.read_csv(data_file)
    target_columns = ["gender"]
    categorical_columns = target_columns.copy()

    # 建立 metadata 並標註類別欄位
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    for col in categorical_columns:
        metadata.update_column(column_name=col, sdtype='categorical')

    # 多任務補資料

    for target_col in target_columns:
        print(f"處理目標欄位: {target_col}")
        df[target_col] = df[target_col].astype(str)

        value_counts = df[target_col].value_counts()

        max_count = value_counts.max()
        for cls, count in value_counts.items():
            if count < max_count:
                samples_needed = max_count - count
                subset = df[df[target_col] == cls]

                print(f"  類別 {cls} 需要補 {samples_needed} 筆")

                synthesizer = CTGANSynthesizer(metadata)
                synthesizer.fit(subset)
                synthetic = synthesizer.sample(num_rows=samples_needed)
                df = pd.concat([df, synthetic], ignore_index=True)

    df.to_csv(output_path, index=False)
    print(f"多任務補齊資料已儲存至 {output_path}")
