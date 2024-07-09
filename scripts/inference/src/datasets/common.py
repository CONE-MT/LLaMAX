from collections import Counter
import sacrebleu
import pandas as pd
import torch
import os
from transformers import AutoTokenizer, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


LGS_DICT = {"afr": "af", "amh": "am", "ara": "ar", "hye": "hy", "asm": "as", "ast": "ast", "azj": 'az', "bel": 'be',
            "ben": 'bn', "bos": 'bs', "bul": 'bg', "mya": 'my',
            "cat": 'ca', "ceb": 'ceb', "zho_simpl": 'zh', "zho_trad": 'zhtrad', "hrv": 'hr', "ces": 'cs', "dan": 'da',
            "nld": 'nl', "eng": 'en', "est": 'et', "tgl": 'tl',
            "fin": 'fi', "fra": 'fr', "ful": 'ff', "glg": 'gl', "lug": 'lg', "kat": 'ka', "deu": 'de', "ell": 'el',
            "guj": 'gu', "hau": 'ha', "heb": 'he', "hin": 'hi', "Latvian": "lij",
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

REVERSE_LGS_DICT = {v: k for k, v in LGS_DICT.items()}


def get_translation_from_hyp(hyp_file, ref_dir, lang_pair):
    format_string = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    src, trg = lang_pair.split("-")
    src_file_name = os.path.join(ref_dir, f"{REVERSE_LGS_DICT[src]}.devtest")
    trg_file_name = os.path.join(ref_dir, f"{REVERSE_LGS_DICT[trg]}.devtest")
    print(f"src_file_name:{src_file_name}, trg_file_name:{trg_file_name}")
    hyps_string, hyps, copy_ratio = "", [], []
    with open(hyp_file, 'r', encoding="utf-8") as reader, \
            open(src_file_name, 'r', encoding="utf-8") as r_reader, \
            open(trg_file_name, 'r', encoding="utf-8") as t_reader:
        for line in reader:
            hyps_string += line.strip()

        refs = [l.strip() for l in t_reader.readlines()]
        src_inputs = [l.strip() for l in r_reader.readlines()]

    parts = hyps_string.split(format_string)
    for part in parts:
        if len(part) == 0:
            continue
        tmp_ps = part.split("###")
        tmp_ref = src_inputs[len(hyps)].strip()
        hyp = ""
        for i, tmp in enumerate(tmp_ps):
            if tmp.startswith(" Input") and tmp != " Input":
                t_input = tmp.split(":")[1].strip()
                if tmp_ref.startswith(t_input) or sacrebleu.sentence_bleu(t_input, [tmp_ref],
                                                                          tokenize="spm").score > 0.9:
                    if i + 1 >= len(tmp_ps):
                        break
                    else:
                        x = tmp_ps[i + 1].split(":")
                        hyp = x[1].strip() if len(x) > 1 else ""
                        if len(hyp) > 0:
                            break
        tmp_counter = Counter(hyp.split())
        tmp = [k for k, v in tmp_counter.items() if v > 2]
        ratio = len(tmp) / len(tmp_counter) if len(hyp) != 0 else 0
        copy_ratio.append(ratio)

        hyps.append(hyp)
        if len(hyps) == len(refs):
            break

    repeat_num = len([i for i in copy_ratio if i > 0.5])
    return hyps, refs, repeat_num


def get_spBLEU(hyps, refs):
    if len(hyps) != len(refs):
        return None
    result = sacrebleu.corpus_bleu(hyps, [refs], tokenize="spm", force=True).score
    return result


def get_translation_from_hyp_2(hyp_file, ref_dir, lang_pair):
    format_string = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    src, trg = lang_pair.split("-")
    src_file_name = os.path.join(ref_dir, "{}.devtest".format(src))
    trg_file_name = os.path.join(ref_dir, "{}.devtest".format(trg))
    print(f"src_file_name:{src_file_name}, trg_file_name:{trg_file_name}")
    hyps_string, hyps, copy_ratio = "", [], []
    with open(hyp_file, 'r', encoding="utf-8") as reader, \
            open(src_file_name, 'r', encoding="utf-8") as r_reader, \
            open(trg_file_name, 'r', encoding="utf-8") as t_reader:
        for line in reader:
            hyps_string += line.strip()

        refs = [l.strip() for l in t_reader.readlines()]
        src_inputs = [l.strip() for l in r_reader.readlines()]

    parts = hyps_string.split(format_string)
    for part in parts:
        if len(part) == 0:
            continue
        tmp_ps = part.split("###")
        tmp_ref = src_inputs[len(hyps)].strip()
        hyp = ""
        for i, tmp in enumerate(tmp_ps):
            if tmp.startswith(" Input") and tmp != " Input":
                t_input = tmp.split(":")[1].strip()
                if tmp_ref.startswith(t_input) or sacrebleu.sentence_bleu(t_input, [tmp_ref],
                                                                          tokenize="spm").score > 0.9:
                    if i + 1 >= len(tmp_ps):
                        break
                    else:
                        x = tmp_ps[i + 1].split(":")
                        hyp = x[1].strip() if len(x) > 1 else ""
                        if len(hyp) > 0:
                            break
        tmp_counter = Counter(hyp.split())
        tmp = [k for k, v in tmp_counter.items() if v > 2]
        ratio = len(tmp) / len(tmp_counter) if len(hyp) != 0 else 0
        copy_ratio.append(ratio)

        hyps.append(hyp)
        if len(hyps) == len(refs):
            break

    repeat_num = len([i for i in copy_ratio if i > 0.5])
    return hyps, refs, repeat_num
