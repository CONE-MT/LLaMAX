import argparse
import logging
import sacrebleu

import sys
from src.datasets.common import get_translation_from_hyp, get_spBLEU
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from peft import PeftModel
import sys
from evaluate import load
from comet import load_from_checkpoint, download_model

def load_model(config):
    # Load checkpoints
    torch_dtype_dict = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32
    }

    print("torchdtype: {}".format(config.model_path.torch_dtype))

    model_name_or_path = config.model_path.base_model

    model = AutoModelForCausalLM.from_pretrained(config.model_path.base_model, 
                                                torch_dtype=torch_dtype_dict[config.model_path.torch_dtype], #trust_remote_code=True,
                                                device_map="auto")
    if config.model_path.lora is not None:
        model = PeftModel.from_pretrained(model, config.model_path.lora, 
                                            torch_dtype=torch_dtype_dict[config.model_path.torch_dtype], #trust_remote_code=True,
                                            )
    print(model.hf_device_map)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "left"


    if tokenizer.pad_token is None and config.model_path.llama_type == "llama3":
        tokenizer.add_special_tokens(
            {
                "eos_token": '<|end_of_text|>',
                "bos_token": '<|begin_of_text|>',
            }
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.pad_token is None and config.model_path.llama_type == "llama2":
        tokenizer.add_special_tokens(
            {
                "eos_token": "</s>",
                "bos_token": "<s>",
                "pad_token": tokenizer.eos_token,
            }
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id


    print("special tokens: ",tokenizer.eos_token_id,tokenizer.pad_token_id, tokenizer.eos_token, tokenizer.pad_token)
    beam_size = config.generat.beam_size
    search = config.generat.search
    temperature = config.generat.temperature
    do_sample =  config.generat.do_sample

    if search == "sample":
        gen_config = GenerationConfig(temperature=temperature,
                                      top_p=0.9,
                                      do_sample=True,
                                      num_beams=beam_size,
                                      max_new_tokens=256,
                                      eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.pad_token_id,
                                      )
        
        print(f"sample")
    elif search == "beam":
        max_new_tokens = 256
        gen_config = GenerationConfig(max_new_tokens=max_new_tokens,
                                      num_beams=beam_size,
                                      eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.pad_token_id,
                                      )
        print("beam search, max_new_tokens: {}".format(max_new_tokens), flush=True)
    else:
        raise ValueError("generat sample setting not right!")
        
        
    return model, tokenizer, gen_config


# Post-process the output, extract translations
def post_process(text):
    text = text.split("### Response:")[1].strip()
    text = text.replace("\n", " ")
    # Cut for contrastive instruction
    if "</p>" in text:
        text = text.split("</p>")[0].split("<p>")[-1]
    return text


def test_process(config, data_dict):
    batch_size = config.generat.batch_size
    print(f"loading mode...")
    model, tokenizer, gen_config = load_model(config)
    print(f"loading is done")
    
    print(f"begin to generate...")
    input_data = data_dict["input_data"]
    prompt = data_dict["prompt_list"]
    input_file=data_dict["input_file"]
    
    record_base_path = config.model_path.lora if config.model_path.lora else config.model_path.base_model
    dataset_name = config.dataset.loader
    lang_pair_name = config.dataset.lang_pair if config.dataset.lang_pair else ""
    record_dir=os.path.join(record_base_path,dataset_name)
    os.makedirs(record_dir, exist_ok=True)
    if config.output.subpath is not None:
        record_dir=os.path.join(record_base_path,dataset_name, config.output.subpath)
        os.makedirs(record_dir, exist_ok=True)
    
    record_lang_dir=os.path.join(record_base_path,dataset_name,config.output.subpath,lang_pair_name)
    os.makedirs(record_lang_dir, exist_ok=True)
    lang_pair = config.dataset.lang_pair
    
    
    output_file = os.path.join(record_lang_dir, config.output.output_file_prefix+".txt")
    output_hyp_file = os.path.join(record_lang_dir, config.output.output_file_prefix+"_.hyp")
    # Generate
    torch.manual_seed(0)
    with open(output_file, 'w', encoding='utf-8') as fo, open(output_hyp_file, 'w', encoding='utf-8') as fo2:
        for i in tqdm(range(0, len(prompt), batch_size)):
            with torch.autocast("cuda"):
                p = prompt[i:i+batch_size]
                tokenized = tokenizer(p, padding=True, return_tensors="pt")
                input_ids = tokenized.input_ids.to(model.device)
                attn_mask = tokenized.attention_mask.to(model.device)
                input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
                attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask
                with torch.no_grad():
                    generated_ids = model.generate(inputs=input_ids, attention_mask=attn_mask, generation_config=gen_config,
                                                   pad_token_id=tokenizer.eos_token_id)


                decoded_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                for dec, gen_ids in zip(decoded_tokens, generated_ids):
                    print(dec, file=fo, flush=True)
                    print(post_process(dec), file=fo2, flush=True)
    
    hyps, refs, repeat_num = get_translation_from_hyp(output_file, os.path.dirname(input_file), lang_pair)
    score = get_spBLEU(hyps, refs)
    print("lang pair: {} scarebleu score: {}".format(lang_pair, score), flush=True)
    metric_file = os.path.join(os.path.dirname(output_file), "spBLEU_summary.csv")
    if os.path.exists(metric_file):
        with open(metric_file, 'a', encoding="utf-8") as writer:
            writer.writelines(f"{lang_pair} {score} \n")
    else:
        with open(metric_file, 'w', encoding="utf-8") as writer:
            writer.writelines(f"{lang_pair} {score} \n")

    comet_score_path = os.path.join(os.path.dirname(output_file), "comet.csv")
    with open(input_file, "r") as f_s:
        source = []
        source_lines = f_s.readlines()
        for l in source_lines:
            source.append(l.strip())
    calculate_comet(source, hyps, refs, comet_score_path, lang_pair)

    print("Finished!")


def compute(model, sources, predictions, references, gpus=None, progress_bar=False):
    if gpus is None:
        gpus = 1 if torch.cuda.is_available() else 0
    data = {"src": sources, "mt": predictions, "ref": references}
    data = [dict(zip(data, t)) for t in zip(*data.values())]
    output = model.predict(data, gpus=gpus, batch_size=16, progress_bar=progress_bar)
    scores, mean_score = output.scores, output.system_score
    return {"mean_score": mean_score, "scores": scores}

def calculate_comet(source, hyps, refs, score_path, lang_pair):
    model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

    results = compute(model=model, predictions=hyps, references=refs, sources=source, gpus=None, progress_bar=True)
    # comet_score = round(results["mean_score"], 2)
    comet_score = results["mean_score"]
    print("comet_score: {} {}".format(lang_pair, str(comet_score)), flush=True)

    if os.path.exists(score_path):
        with open(score_path, 'a', encoding="utf-8") as writer:
            writer.writelines(f"{lang_pair} {comet_score} \n")
    else:
        with open(score_path, 'w', encoding="utf-8") as writer:
            writer.writelines(f"{lang_pair} {comet_score} \n")


