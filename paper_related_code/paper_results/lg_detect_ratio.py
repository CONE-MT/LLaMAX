import json
import os

from ftlangdetect import detect

chinese_llama_root_dir = "/nas/shared/NLP_A100/yqlu/model/base_model/chinese-alpaca-2-7b/flores/"
llama2_root_dir = "/cpfs01/user/luyinquan/model/base_model/llama2-7b-hf_alpaca_new_env_128_1_bfloat16_2e-5/checkpoint-813/flores"
jpn_llama_root_dir = "/cpfs01/user/luyinquan/model/base_model/Swallow-7b-instruct-v0.1/flores"
xllama_root_dir = "/cpfs01/user/luyinquan/model/base_model/llama2-our-alpaca-model-average_v4-v5-v7/flores"


# model_path = "/nas/shared/NLP_A100/feyuan/model/non_llama_family/lg_identification/model.bin"
# model = fasttext.load_model(model_path)


def calculate_statistics_info(result, training_lg, inference_lg):
    training_count, inference_count = 0, 0
    for lg in result:
        if lg == training_lg:
            training_count += 1
        elif lg == inference_lg:
            inference_count += 1
    return round(training_count/len(result) * 100, 2), round(inference_count/len(result) * 100, 2)


def lg_detect(model_dir, training_lg):
    dir_names = sorted(os.listdir(model_dir))
    for dir_name in dir_names:
        _, inference_lg = dir_name.split("-")
        dir_path = os.path.join(model_dir, dir_name)
        if not os.path.isdir(dir_path) or not dir_name.startswith(training_lg+"-"):
            continue
        if model_dir == xllama_root_dir or model_dir == llama2_root_dir:
            fpath = os.path.join(dir_path, "generation_results_.hyp")
            with open(fpath, 'r', encoding="utf-8") as reader:
                result = []
                for i, line in enumerate(reader):
                    line = line.strip().replace("\n", "")
                    lang = detect(text=line, low_memory=False)["lang"]
                    result.append(lang)
        else:
            fpath = os.path.join(dir_path, "generate.txt")
            with open(fpath, 'r', encoding="utf-8") as reader:
                result = []
                for i, line in enumerate(reader):
                    line = json.loads(line)[str(i)].strip().replace("\n", "")
                    lang = detect(text=line, low_memory=False)["lang"]
                    result.append(lang)
        training_ratio, inference_ratio = calculate_statistics_info(result, training_lg, inference_lg)
        print(training_lg, inference_lg, training_ratio, inference_ratio)


# zh_detect()
# lg_detect(llama2_root_dir, "zh")
# print("=====" * 20)
# lg_detect(chinese_llama_root_dir, "zh")
# print("=====" * 20)
# lg_detect(xllama_root_dir, "zh")

lg_detect(llama2_root_dir, "ja")
print("=====" * 20)
lg_detect(jpn_llama_root_dir, "ja")
print("=====" * 20)
lg_detect(xllama_root_dir, "ja")
