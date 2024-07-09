import os
import pandas as pd
import seaborn as sns

result_summary_dir = "D:/Project/XLLaMA/paper_results/data"

model_names = os.listdir(result_summary_dir)

model_names = [model_name for model_name in model_names if os.path.isdir(os.path.join(result_summary_dir, model_name)) and model_name not in ["gpt-4", "mala-500", "aya-101", "llama2"]]
print(model_names)

threshold = 10

res_df = pd.DataFrame(columns=["model_name", "spbleu", "direct"])

display_names = {

'aya-23-8B': "Aya-23-8B",
'bayling': "Bayling-7B",
'bloomz-7b1-mt': "Bloomz-7B",
'llama2-alpaca': "LLaMA2-Alpaca-7B",
# 'llama2-our-alpaca-model-average_v4-v5-v7': "TransLLaMA2-Alpaca-7B",
'llama3-alpace': "LLaMA3-Alpaca-8B",
'madlad400-7b-mt': "MADLAD-7B",
'mistral-7B-v0.1-llama': "Mistral-7B",
'polylm': "PolyLM-13B",
'TowerInstruct-7B-v0.2': "TowerInstruct-7B",
'yayi2-30b': "Yayi2-30B"



}

for model_name in model_names:
    # if model_name == "gpt-4": continue
    if model_name not in display_names.keys():
        continue
    detail_fpath = os.path.join(result_summary_dir, model_name, "sacrebleu_score", "summary_101.csv")
    df = pd.read_csv(detail_fpath)
    for _, row in df.iterrows():
        direct = row["direct"]
        if direct == "x-en" or direct == "x-ceb":
            spbleu = row["all"]
            res_df.loc[res_df.shape[0]] = [display_names[model_name], spbleu, direct]
res_df = res_df.sort_values("direct")


import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 20})

fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(data=res_df, x='model_name', y='spbleu', hue='direct')
ax.axhline(y=30.63, color='#420420', linestyle='--', linewidth=5)
ax.axhline(y=16.11, color='#EA4335', linestyle='-', linewidth=5)
ax.text(3, 5.0, 'LLaMA-7B', color='#420420', va='bottom')
ax.xaxis.set_tick_params(rotation=70)

plt.tight_layout()
plt.show()