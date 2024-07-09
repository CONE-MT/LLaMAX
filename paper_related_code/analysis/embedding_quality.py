import os
import sys

from matplotlib.markers import MarkerStyle
from scipy.stats import spearmanr
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import seaborn as sns

# from analysis.common import LLAMA2_MODEL_PATH, LGS_DICT

sns.set_theme(style='white')

import matplotlib.pyplot as plt
import pandas as pd


LLAMA2_MODEL_PATH = "/nas/shared/NLP_A100/feyuan/model/llama_family/llama2-7b-hf"

LGS_DICT = {"afr": "af", "amh": "am", "ara": "ar", "hye": "hy", "asm": "as", "ast": "ast", "azj": 'az', "bel": 'be',
            "ben": 'bn', "bos": 'bs', "bul": 'bg', "mya": 'my',
            "cat": 'ca', "ceb": 'ceb', "zho_simpl": 'zh', "zho_trad": 'zhtrad', "hrv": 'hr', "ces": 'cs', "dan": 'da',
            "nld": 'nl', "eng": 'en', "est": 'et', "tgl": 'tl',
            "fin": 'fi', "fra": 'fr', "ful": 'ff', "glg": 'gl', "lug": 'lg', "kat": 'ka', "deu": 'de', "ell": 'el',
            "guj": 'gu', "hau": 'ha', "heb": 'he', "hin": 'hi',
            "hun": 'hu', "isl": 'is', "ibo": 'ig', "ind": 'id', "gle": 'ga', "ita": 'it', "jpn": 'ja', "jav": 'jv',
            "kea": 'kea', "kam": 'kam', "kan": 'kn', "kaz": 'kk',
            "khm": 'km', "kor": 'ko', "kir": 'ky', "lao": 'lo', "lav": 'lv', "lin": 'ln', "lit": 'lt', "luo": 'luo',
            "ltz": 'lb', "mkd": 'mk', "msa": 'ms', "mal": 'ml',
            "mlt": 'mt', "mri": 'mi', "mar": 'mr', "mon": 'mn', "npi": 'ne', "nso": 'ns', "nob": 'no', "nya": 'ny',
            "oci": 'oc', "ory": 'or', "orm": 'om', "pus": 'ps',
            "fas": 'fa', "pol": 'pl', "por": 'pt', "pan": 'pa', "ron": 'ro', "rus": 'ru', "srp": 'sr', "sna": 'sn',
            "snd": 'sd', "slk": 'sk', "slv": 'sl', "som": 'so',
            "ckb": 'ku', "spa": 'es', "swh": 'sw', "swe": 'sv', "tgk": 'tg', "tam": 'ta', "tel": 'te', "tha": 'th',
            "tur": 'tr', "ukr": 'uk', "umb": 'umb', "urd": 'ur',
            "uzb": 'uz', "vie": 'vi', "cym": 'cy', "wol": 'wo', "xho": 'xh', "yor": 'yo', "zul": 'zu'}

def get_tensor_dict(file_path):
    result = []
    with open(file_path, 'r', encoding="utf-8") as reader:
        for line in reader:
            tmp = list(map(float, line.strip().split()))
            tmp_tensor = torch.tensor(tmp, dtype=torch.float32)
            result.append(tmp_tensor)
    return torch.stack(result, dim=0)


def calculate_cos_smi(input_tensor, target_tensor):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(input_tensor, target_tensor)


def calculate_rank(result):
    R_1,  R_2, R_5, R_10 = [], [], [], []
    for i, tmp in enumerate(result):
        top10_indices = [item.item() for item in list(torch.topk(tmp, 10).indices)]

        top_1 = 1 if i in top10_indices[:1] else 0
        R_1.append(top_1)

        top_2 = 1 if i in top10_indices[:2] else 0
        R_2.append(top_2)

        top_5 = 1 if i in top10_indices[:5] else 0
        R_5.append(top_5)

        top_10 = 1 if i in top10_indices else 0
        R_10.append(top_10)
    return np.mean(R_1), np.mean(R_2), np.mean(R_5), np.mean(R_10)


def plot_png(res_df, save_fpath):
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    sns.barplot(x="lang", y="tokenization_ratio", data=res_df, ax=ax1, color="#EA4335", label="Fertility")
    for i, p in enumerate(ax1.patches):
        lg = list(res_df["lang"])[i]
        if lg in ["ro", "el", "bn"]:
            ax1.text(p.get_x() + p.get_width() / 2., p.get_height(), '%s\n%.2f' % (lg, round(p.get_height(), 2)),
                     fontsize=12, color='#FBBB01', ha='center', va='bottom')
            p.set_color('#FBBB01')
    sns.lineplot(x="lang", y='cos_similarity', data=res_df, ax=ax2, label=" Cosine Similarity", linewidth=3, color="#4285F4")
    sns.lineplot(x="lang", y='R@1', data=res_df, ax=ax2, label="Recall@1", linewidth=3, color="#34A853")
    # plt.legend(ncol=2, fontsize=20)

    marker = MarkerStyle(marker='*')

    for i, lg in enumerate(list(res_df["lang"])):
        if lg in ["ro", "bn"]:
            ax2.plot(i, list(res_df["cos_similarity"])[i], marker=marker, color="#4285F4", fillstyle='none', markersize=15)
            ax2.plot(i, list(res_df["R@1"])[i], marker=marker, color="#34A853", fillstyle='none', markersize=15)

    ax1.xaxis.set_tick_params(rotation=70)
    ax1.grid(False)
    ax2.grid(False)
    plt.xlabel("Languages")
    ax1.set_ylabel("Fertility", fontsize=20)
    ax1.set_xlabel("Languages", fontsize=20)
    ax1.set_ylim(0, 89)

    ax2.set_ylabel("Embedding Quality", fontsize=20)
    ax2.set_xlabel("Languages", fontsize=20)
    ax2.set_ylim(0, 0.65)
    plt.xticks([x for i, x in enumerate(res_df["lang"]) if i % 3 == 0], ha='center', fontsize=18)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.tick_params(axis='x', labelsize=20)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, ncol=3, loc="upper left", fontsize=20)
    ax1.legend().set_visible(False)


    plt.tight_layout()
    plt.show()
    # plt.savefig("/cpfs01/user/yuanfei/figures/correlation_fertility_and_quality.pdf", dpi=1200)
    # plt.savefig(save_fpath, dpi=1200)


if __name__ == '__main__':
    model_id = sys.argv[1] if len(sys.argv) > 1 else LLAMA2_MODEL_PATH
    output_save_dir = os.path.join(model_id, "output_embedding_files")
    input_save_dir = os.path.join(model_id, "input_embedding_files")

    for dir_path, type_name in zip([input_save_dir, output_save_dir], ["input", "output"]):

        target_file = "eng.devtest"
        file_names = os.listdir(dir_path)

        similarity_file = os.path.join(dir_path, "%s_similarity_result.csv" % type_name)

        if os.path.exists(similarity_file):
            res_df = pd.read_csv(similarity_file)
        else:
            res_df = pd.DataFrame(columns=["lang", "cos_similarity", "R@1", "R@2", "R@5", "R@10"])
            eng_tensor = get_tensor_dict(os.path.join(input_save_dir, target_file))

            for file_name in tqdm(file_names):
                if file_name == target_file or not file_name.endswith("devtest"):
                    continue
                file_path = os.path.join(dir_path, file_name)
                current_tensor = get_tensor_dict(os.path.join(dir_path, file_name))
                cos_smi_result = calculate_cos_smi(eng_tensor, current_tensor).mean().item()
                lang = LGS_DICT[file_name.split(".")[0]]

                result = []
                for i in range(eng_tensor.size(0)):
                    tmp = calculate_cos_smi(eng_tensor[i, :], current_tensor)
                    result.append(tmp)

                R_1,  R_2, R_5, R_10 = calculate_rank(result)
                res_df.loc[res_df.shape[0]] = [lang, cos_smi_result, R_1, R_2, R_5, R_10]
            res_df.to_csv(similarity_file, index=False)

        tokenized_file_path = os.path.join(dir_path, "%s_fertility_result.csv" % type_name)
        tokenized_summary = pd.read_csv(tokenized_file_path)

        sim_df = pd.merge(tokenized_summary, res_df, left_on="lang", right_on="lang")
        sim_df = sim_df.sort_values(by='tokenization_ratio', ascending=True)

        plot_png(sim_df, save_fpath=os.path.join(dir_path, "%s_correlation_fertility_and_quality.pdf" % type_name))

        corr, _ = spearmanr(sim_df["tokenization_ratio"], sim_df["cos_similarity"])
        print('%s Spearmans correlation: %.3f' % (type_name, corr))

        corr, _ = spearmanr(sim_df["tokenization_ratio"], sim_df["R@1"])
        print('%s Spearmans correlation: %.3f' % (type_name, corr))

        sim_df.to_csv(os.path.join(dir_path, "%s_sim_df_combine.csv" % type_name))




