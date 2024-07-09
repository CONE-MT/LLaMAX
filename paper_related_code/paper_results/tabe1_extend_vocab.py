import sys

import pandas as pd
import os
import torch
from torch import nn
import numpy as np
from scipy.stats import ks_2samp
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

LLAMA2_MODEL_PATH = "/nas/shared/NLP_A100/feyuan/model/llama_family/llama2-7b-hf"

root_dir = "/nas/shared/NLP_A100/feyuan/model/tokenizer_exps/tokenizer_scaling_only_new/baselines_llama2/"

model_paths = [
# f"{root_dir}/fully_tuning/ro",
# f"{root_dir}/extend_vocab_sft/vocab_100/ro", f"{root_dir}/extend_vocab_sft/vocab_800/ro", f"{root_dir}/extend_vocab_sft/vocab_1600/ro",
# f"{root_dir}/extend_vocab_sft/vocab_6400/ro", f"{root_dir}/extend_vocab_sft/vocab_12800/ro", f"{root_dir}/extend_vocab_sft/vocab_25600/ro",
# f"{root_dir}/extend_vocab_sft/vocab_51200/ro",
#
# f"{root_dir}/fully_tuning/el", f"{root_dir}/extend_vocab_sft/vocab_100/el", f"{root_dir}/extend_vocab_sft/vocab_800/el",
# f"{root_dir}/extend_vocab_sft/vocab_1600/el", f"{root_dir}/extend_vocab_sft/vocab_6400/el", f"{root_dir}/extend_vocab_sft/vocab_12800/el",
# f"{root_dir}/extend_vocab_sft/vocab_25600/el", f"{root_dir}/extend_vocab_sft/vocab_51200/el",
#
# f"{root_dir}/fully_tuning/bn", f"{root_dir}/extend_vocab_sft/vocab_100/bn",
# f"{root_dir}/extend_vocab_sft/vocab_800/bn", f"{root_dir}/extend_vocab_sft/vocab_1600/bn", f"{root_dir}/extend_vocab_sft/vocab_6400/bn",
# f"{root_dir}/extend_vocab_sft/vocab_12800/bn", f"{root_dir}/extend_vocab_sft/vocab_25600/bn", f"{root_dir}/extend_vocab_sft/vocab_51200/bn",
"/cpfs01/shared/public/yuanfei/average/"
]

# type_name = "input"
# for model_path in model_paths:
#     file_name = os.path.join(model_path, "%s_embedding_files" % type_name, "%s_sim_df_combine.csv" % type_name)
#     df = pd.read_csv(file_name)
#     lang = os.path.basename(model_path)
#     row = df[df["lang"]==lang]
#
#     print(file_name, "\n", round(row["tokenization_ratio"].values[0], 2),
#           round(row["cos_similarity"].values[0], 2),
#           round(row["R@1"].values[0], 2))
#     print()


def calculate_cos_smi(input_tensor, target_tensor):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(input_tensor, target_tensor)

def calculate_ks(base_tensor, other_tensor, save_path, alpha=0.25):
    result = []
    if os.path.exists(save_path):
        tmp_df = pd.read_csv(save_path)
        for _, row in tmp_df.iterrows():
            if row["pvalue"] < alpha:
                result.append(row["statistic"])
        return round(np.mean(result), 4), len(result)

    tmp_df = pd.DataFrame(columns=["index", "statistic", "pvalue"])
    for i in range(base_tensor.size(0)):
        ks_res = ks_2samp(base_tensor[i, :].detach().numpy(), other_tensor[i, :].detach().numpy(), method="asymp")
        tmp_df.loc[tmp_df.shape[0]] = [i, ks_res.statistic, ks_res.pvalue]
        if ks_res.pvalue < alpha:
            result.append(ks_res.statistic)
    tmp_df.to_csv(save_path, index=False)
    return round(np.mean(result), 4), len(result)


def load_input_embed(model_dir):
    if os.path.exists(os.path.join(model_dir, "input_embed.pt")):
        input_embed = torch.load(os.path.join(model_dir, "input_embed.pt"))
    else:
        input_embed = AutoModelForCausalLM.from_pretrained(model_dir).get_input_embeddings().weight.data
        torch.save(input_embed, os.path.join(model_dir, "input_embed.pt"))
    return input_embed


def load_output_embed(model_dir):
    if os.path.exists(os.path.join(model_dir, "output_embed.pt")):
        output_embed = torch.load(os.path.join(model_dir, "output_embed.pt"))
    else:
        output_embed = AutoModelForCausalLM.from_pretrained(model_dir).get_output_embeddings().weight.data
        torch.save(output_embed, os.path.join(model_dir, "output_embed.pt"))
    return output_embed


alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5

llama2_embed = load_input_embed(LLAMA2_MODEL_PATH)

for model_path in model_paths:
    tmp_embed = load_input_embed(model_path)
    x = calculate_cos_smi(llama2_embed, tmp_embed[:llama2_embed.shape[0]])
    y = calculate_ks(llama2_embed, tmp_embed, save_path=os.path.join(model_path, "input_ks.csv"), alpha=alpha)
    print(alpha, model_path, y)


