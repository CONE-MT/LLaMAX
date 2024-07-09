import os
import shutil
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
import torch.nn as nn

# # from analysis import LLAMA2_MODEL_PATH, FLORES_DEVTEST_DIR
# #
LLAMA2_MODEL_PATH = "/nas/shared/NLP_A100/feyuan/model/llama_family/llama2-7b-hf"
FLORES_DEVTEST_DIR = "/nas/shared/NLP_A100/feyuan/data/flores_101_devtest"


target_files = os.listdir(FLORES_DEVTEST_DIR)
model_id = sys.argv[1] if len(sys.argv) > 1 else LLAMA2_MODEL_PATH
output_save_dir = os.path.join(model_id, "output_embedding_files")
input_save_dir = os.path.join(model_id, "input_embedding_files")

if os.path.exists(input_save_dir):
    if os.path.isfile(input_save_dir):
        os.remove(input_save_dir)
    else:
        shutil.rmtree(input_save_dir)

if os.path.exists(output_save_dir):
    if os.path.isfile(output_save_dir):
        os.remove(output_save_dir)
    else:
        shutil.rmtree(output_save_dir)

os.makedirs(input_save_dir, exist_ok=True)
os.makedirs(output_save_dir, exist_ok=True)


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
input_embeddings = model.get_input_embeddings()
output_embeddings = nn.Embedding.from_pretrained(model.get_output_embeddings().weight.data)

for file_name in target_files:
    if "devtest" not in file_name:
        continue
    file_path, save_input_path, save_output_path = os.path.join(FLORES_DEVTEST_DIR, file_name), \
        os.path.join(input_save_dir, file_name), os.path.join(output_save_dir, file_name)

    with open(file_path, 'r', encoding="utf-8") as reader, open(save_output_path, 'a', encoding="utf-8") as output_writer, \
            open(save_input_path, 'a', encoding="utf-8") as input_writer:
        for line in tqdm(reader, desc=f"{file_name} encoding"):
            prompt = line.strip()
            input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            input_embeds = input_embeddings(torch.LongTensor([input_ids]))
            output_embeds = output_embeddings(torch.LongTensor([input_ids]))

            input_mean = torch.mean(input_embeds[0], 0).detach().numpy()
            output_mean = torch.mean(output_embeds[0], 0).detach().numpy()

            input_writer.writelines("%s\n" % (" ".join(map(str, input_mean))))
            output_writer.writelines("%s\n" % (" ".join(map(str, output_mean))))