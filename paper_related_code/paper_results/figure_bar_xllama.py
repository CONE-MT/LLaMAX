import matplotlib.pyplot as plt
import numpy as np


labels = ['X-CSQA', 'XNLI', 'MGSM']
llama2_sft = [50.93, 70.62, 38.32]
xllama2_sft = [55.11, 76.22, 43.70]

x = np.arange(len(labels))
width = 0.3

plt.rcParams.update({'font.size': 20})

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width/2, llama2_sft, width, label='LLaMA2-Task', color='#4285F4')
rects2 = ax.bar(x + width/2, xllama2_sft, width, label='LLaMAX2-Task', color='#EA4335')

ax.set_ylabel('Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(ncol=1, fontsize=19)
ax.set_ylim(30, 90)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()

# plt.show()
plt.savefig('figure_bar_xllama.pdf', dpi=600)