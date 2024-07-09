import os
import pandas as pd

result_summary_dir = "D:/Project/XLLaMA/paper_results/data"

display_names = {

'aya-23-8B': "Aya-23-8B",
# 'bayling': "Bayling-7B",
# 'bloomz-7b1-mt': "Bloomz-7B",
'llama2-7b-hf': "LLaMA2-7B",
'llama2-our-alpaca-model-average_v4-v5-v7': "LLaMAX2-Alpaca-7B",
'llama-3-8b-hf': "LLaMA3-8B",
'madlad400-7b-mt': "MADLAD-7B",
# 'mistral-7B-v0.1-llama': "Mistral-7B",
'polylm': "PolyLM-13B",
'TowerInstruct-7B-v0.2': "TowerInstruct-7B",
'yayi2-30b': "Yayi2-30B"



}

model_names = os.listdir(result_summary_dir)

model_names = [model_name for model_name in model_names if os.path.isdir(os.path.join(result_summary_dir, model_name)) and  model_name in display_names.keys()]
print(model_names)

threshold = 10

res_df = pd.DataFrame(index=model_names, columns=['en-x', 'x-en', 'zh-x', 'x-zh', 'de-x', 'x-de', 'ar-x', 'x-ar', 'ne-x', 'x-ne', 'az-x', 'x-az', 'ceb-x', 'x-ceb', "model_type"])

for model_name in model_names:
    # if model_name == "gpt-4": continue

    detail_fpath = os.path.join(result_summary_dir, model_name, "sacrebleu_score", "scarebleu_score.csv")
    df = pd.read_csv(detail_fpath)
    for _, row in df.iterrows():
        direct = row[0]
        above = sum([1 if float(score) > threshold else 0 for score in row.values[1:] if score != "/"])
        res_df.loc[model_name][direct] = above
res_df


import matplotlib.pyplot as plt
import numpy as np

# Create some data
x1 = list(res_df["x-en"].values)
y1 = list(res_df["en-x"].values)

x2 = list(res_df["x-ar"].values)
y2 = list(res_df["ar-x"].values)

# x2 = list(res_df["x-az"].values)
# y2 = list(res_df["az-x"].values)

# markers =['o', '^', 'v', '8', 'd', '*',  'D', 'p', 'h', 'H', '8', 's']
markers =['o', '^', 'v', '*', 's', 'p', 'h', '8']

print(len(x2), len(markers))

for i in range(len(model_names)):
    if model_names[i] == "llama2-our-alpaca-model-average_v4-v5-v7":
        plt.scatter(x1[i], y1[i], marker=markers[i], color="#4285F4", s=300, edgecolors='black')
    else:
        plt.scatter(x1[i], y1[i], marker=markers[i], color="#4285F4", s=150, edgecolors='black')

for i in range(len(model_names)):
    if model_names[i] == "llama2-our-alpaca-model-average_v4-v5-v7":
        plt.scatter(x2[i], y2[i], marker=markers[i], label=display_names[model_names[i]], color="#EA4335", s=300, edgecolors='black')
    else:
        plt.scatter(x2[i], y2[i], marker=markers[i], label=display_names[model_names[i]], color="#EA4335", s=150, edgecolors='black')

plt.ylim((-3, 100))
plt.xlim((-3, 100))
legend = plt.legend(loc="best", ncol=2, fontsize=10)
for handle in legend.legendHandles:
    handle.set_color('black')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel("# spBLEU (X$\\rightarrow$LG) > 10", fontsize=15)
plt.ylabel("# spBELU (LG$\\rightarrow$X) > 10", fontsize=15)
plt.tight_layout()

# Show the plot
plt.savefig("./introduction.pdf", dpi=600)
# plt.show()