import pandas as pd


lang_info = pd.read_excel("data/final_language_info.xlsx")
data_info = pd.read_csv("data/corpus_merge.csv")

summary_df = pd.merge(lang_info, data_info, left_on="ISO", right_on="lang")
summary_df.to_excel("data/data_final.merge.xlsx", index=False)
