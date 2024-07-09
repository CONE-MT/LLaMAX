import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 25})


gpt4 = np.array([
    [np.nan, 30.69, 47.19, 35.96, 41.50, 24.60, 43.13],
    [29.10, np.nan, 24.47, 17.51, 21.85, 16.03, 19.57],
    [45.51, 25.29, np.nan, 24.55, 30.49, 20.14, 29.06],
    [22.32, 12.90, 16.32, np.nan, 14.54, 11.87, 14.51],
    [32.63, 18.72, 25.11, 17.67, np.nan, 16.25, 20.59],
    [15.95, 11.94, 14.22, 10.51, 13.24, np.nan, 10.85],
    [29.68, 16.49, 22.00, 17.87, 19.91, 14.12, np.nan]
])

xllama2_alpaca = np.array([
    [np.nan, 28.71, 44.16, 33.31, 35.90, 22.05, 38.64],
    [22.40, np.nan, 18.63, 14.42, 15.68, 13.24, 15.29],
    [35.83, 19.72, np.nan, 19.73, 22.53, 16.18, 22.01],
    [26.28, 16.89, 21.04, np.nan, 17.27, 14.31, 17.38],
    [26.34, 15.82, 20.72, 15.62, np.nan, 13.87, 17.40],
    [15.90, 12.43, 14.41, 12.29, 12.73, np.nan, 11.06],
    [29.77, 16.73, 22.44, 16.57, 19.65, 12.90, np.nan]
])

merged = np.around((xllama2_alpaca - gpt4), 2)

annot = np.vectorize(lambda x: f"{x:+.1f}")(merged)

labels = ["en", "zh", "de", "ne", "ar", "az", "ceb"]

plt.figure(figsize=(10, 8))
sns.heatmap(merged, annot=annot, fmt='', xticklabels=labels, yticklabels=labels, cmap="coolwarm", linewidths=.5, center=0)
# plt.title("LLaMAX2-Alpaca vs. GPT-4")
plt.xlabel("Source Language")
plt.ylabel("Target Language")
plt.tight_layout()
plt.savefig('figure_heatmap_gpt4.pdf', dpi=600)