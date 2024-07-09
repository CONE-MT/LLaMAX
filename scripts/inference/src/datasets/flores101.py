import argparse
import sys

from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# Instruction language, default: 'en'
def get_lang_instruction(lang_instruction_path):
    df = pd.read_excel(lang_instruction_path)
    return dict(zip(df['language_code'], df['language_en']))

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n"
        "### Instruction:\n{instruction}\n### Input:\n{input}\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n"
        "### Instruction:\n{instruction}\n### Response:"
    ),
}


# Read task instruction, fill in languages
def read_instruct(path, src, tgt, lang_instruction):
    source, target = lang_instruction[src], lang_instruction[tgt]
    ins_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for l in f:
            line = l.strip().replace("[SRC]", source).replace("[TGT]", target)
            ins_list.append(line)
    return ins_list


# Read input data for inference
def read_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        input_data = f.readlines()
    return input_data


# Assembly instruction and input data, handle hints
def create_prompt(instruct, input_data, template="prompt_no_input"):
    if "###" in instruct:
        instruct, input_suffix = instruct.split("###")
        hint = "\n\n### Hint: {}".format(input_suffix)
    else:
        instruct =  instruct
        hint = ""
    if template == "prompt_input":
        list_data_dict = [{"instruction": instruct, "input": p.strip() + hint} for p in input_data]
        prompt_input = PROMPT_DICT[template]
        sources = [ prompt_input.format_map(example) for example in list_data_dict ]
    else:
        list_data_dict = [{"instruction": "\n\n".join([instruct, p.strip() + hint]).strip(), "input": ""} for p in input_data]
        prompt_input = PROMPT_DICT[template]
        sources = [ prompt_input.format_map(example) for example in list_data_dict ]
    return sources




def dataloader(config):

    inst_fix = config.dataset.inst_fix
    inst_index = config.dataset.inst_index if inst_fix else None
    inst_with_input = config.dataset.inst_with_input
    inst_placeholder_list = config.dataset.inst_placeholder
    input_file = os.path.join(config.dataset.path, config.dataset.input_file)
    input_size = config.dataset.input_size
    input_extra_file = os.path.join(config.dataset.path, config.dataset.input_extra_file) if config.dataset.input_extra_file is not None else None 
    inst_file  = config.dataset.inst_file

    template = config.generat.template

    # Prepare input data
    srcl, tgtl = config.dataset.lang_pair.split('-')
    lang_instruction = get_lang_instruction(config.dataset.lang_instruction)

    if inst_file is not None:
        instructs = read_instruct(inst_file, srcl, tgtl, lang_instruction)
        instruct = instructs[0] if len(instructs) > 0 else ""
    else: # In case instruction file is missing, then use input as instruction
        instruct = ""
        template = "prompt_no_input"
    input_data = read_input(input_file)
    prompt = create_prompt(instruct, input_data, template)

    data_dict = {
        "prompt_list": prompt,
        "input_data": input_data,
        "input_file":input_file
    }
    return data_dict
