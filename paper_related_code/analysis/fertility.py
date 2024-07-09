import os
import sys

import numpy as np

import seaborn as sns

# from analysis.common import LLAMA2_MODEL_PATH, FLORES_DEVTEST_DIR, LGS_DICT, get_tokenizer

sns.set_theme(style='white')

import matplotlib.pyplot as plt
import pandas as pd


from transformers import AutoTokenizer

LLAMA2_MODEL_PATH = "/nas/shared/NLP_A100/feyuan/model/llama_family/llama2-7b-hf"
FLORES_DEVTEST_DIR = "/nas/shared/NLP_A100/feyuan/data/flores_101_devtest"

def get_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return tokenizer



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

FLORES_LGS = LGS_DICT.values()

def get_tokenized_df(model_path, save_path):
    tokenizer = get_tokenizer(model_path)

    file_names = os.listdir(FLORES_DEVTEST_DIR)

    result_df = pd.DataFrame(
        columns=["lang", "input_sentence_len", "tokenization_len", "tokenization_ratio"])

    for file_name in file_names:
        if "devtest" not in file_name:
            continue
        lang = LGS_DICT[file_name.split(".")[0]]
        file_path = os.path.join(FLORES_DEVTEST_DIR, file_name)
        seq_list, tokenized_list, ratio_list = [], [], []
        with open(file_path, 'r', encoding="utf-8") as reader:
            for line in reader:
                line = line.strip()
                seq = line.split()
                if "zh" in lang and "ja" in lang:
                    seq = [c for c in line]
                tokenized_seq = tokenizer(line, add_special_tokens=False).input_ids
                seq_list.append(len(seq))
                tokenized_list.append(len(tokenized_seq))
                ratio_list.append(len(tokenized_seq) / len(seq))
        result_df.loc[result_df.shape[0]] = [lang, np.mean(seq_list), np.mean(tokenized_list), np.mean(ratio_list)]
    result_df.to_csv(save_path, index=False)
    return result_df


def plot_png(res_df):
    fig = plt.figure(figsize=(13, 7))
    ax1 = fig.add_subplot(111)
    # ax2 = ax1.twinx()

    # sns.barplot(x="lang", y="tokenization_ratio", data=res_df, ax=ax1, color="#4285F4")
    # for i, p in enumerate(ax1.patches):
    #     ax1.text(p.get_x() + p.get_width() / 2., p.get_height(), '%.2f' % round(p.get_height(), 2),
    #              fontsize=12, color='black', ha='center', va='bottom')
    sns.lineplot(x="lang", y='input_sentence_len', data=res_df, ax=ax1, label="Input Sentence Length", linewidth=2, color="#34A853")
    sns.lineplot(x="lang", y='tokenization_len', data=res_df, ax=ax1, label="Tokenized Sequence Length", linewidth=2, color="#EA4335")
    plt.legend(ncol=2, fontsize=20)

    ax1.xaxis.set_tick_params(rotation=70)
    ax1.grid(False)
    # ax2.grid(False)
    plt.xlabel("Languages", fontsize=20)
    ax1.set_ylabel("Length", fontsize=20)
    # ax2.set_ylabel("Tokenization Ratio")
    plt.xticks([x for i, x in enumerate(res_df["lang"]) if i % 2 == 0], ha='center', fontsize=15)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    model_id = sys.argv[1] if len(sys.argv) > 1 else LLAMA2_MODEL_PATH
    output_save_dir = os.path.join(model_id, "output_embedding_files")
    input_save_dir = os.path.join(model_id, "input_embedding_files")

    for dir_path, type_name in zip([input_save_dir, output_save_dir], ["input", "output"]):
        print(dir_path)
        tokenized_file_path = os.path.join(dir_path, "%s_fertility_result.csv" % type_name)
        print(tokenized_file_path)
        if os.path.exists(tokenized_file_path):
            result_df = pd.read_csv(tokenized_file_path)
        else:
            result_df = get_tokenized_df(model_id, tokenized_file_path)

        result_df = result_df.sort_values(by='tokenization_len', ascending=True)

        plot_png(result_df)
