import pandas as pd

# Đọc hai file TSV (không có header)
ser_df = pd.read_csv("SER_end.tsv", sep="\t", header=None, names=["file", "ser", "asr_dummy"])
asr_df = pd.read_csv("ASR_end_pro.tsv", sep="\t", header=None, names=["file", "ser_dummy", "asr"])

# Merge dựa trên tên file
merged_df = pd.merge(ser_df[["file", "ser"]],
                     asr_df[["file", "asr"]],
                     on="file", how="inner")

# Lưu ra results.tsv
merged_df.to_csv("results_pro.tsv", sep="\t", index=False, header=False)
