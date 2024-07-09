import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({'font.size': 20})

systems = ['LLAMA2-SFT', 'xLLAMA2-SFT (ours)']
metrics = ['En$\\rightarrow$X (seen)', 'En$\\rightarrow$X (unseen)', 'X$\\rightarrow$En (seen)', 'X$\\rightarrow$En (unseen)']
llama2_sft = [9.44, 6.04, 16.44, 11.89]
xllama2_sft = [23.17, 14.44, 30.63, 22.14]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, llama2_sft, width, label='LLaMA2-Alpaca', color='#4285F4')
rects2 = ax.bar(x + width/2, xllama2_sft, width, label='LLaMAX2-Alpaca', color='#EA4335')

# ax.set_title('Flores-200', fontsize=15)
ax.set_ylabel('spBLEU', fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(fontsize=20)

# plt.rcParams.update({'font.size': 12})
ax.tick_params(axis='both', which='major', labelsize=17)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=15)

autolabel(rects1)
autolabel(rects2)

ax.set_ylim(0, 33)

fig.tight_layout()

plt.show()
fig.savefig('figure_bar_flores200.pdf', dpi=600)