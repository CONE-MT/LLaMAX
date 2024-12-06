# LLaMAX: Scaling Linguistic Horizons of LLM by Enhancing Translation Capabilities Beyond 100 Languages

[**HuggingFace**](https://huggingface.co/LLaMAX) | [**Arxiv**](http://arxiv.org/abs/2407.05975) | [**Citation**](#citation) |

LLaMAX is a large language model designed for multilingual scenarios. It is based on Meta's LLaMA series models and 
continues training on over 100 languages. Without losing its generalization ability, the multilingual capabilities of LLaMAX 
significantly exceeds that of existing LLMs. Only simple supervised fine-tuning(SFT) is needed to meet multilingual requirements in downstream tasks.

## News
**📢[Jul 26, 2024] LLaMAX3.1-8B is launched!**

**🔥[Jul 26, 2024] Welcome to try the [online translation demo](https://huggingface.co/spaces/vilarin/LLaMAX3-Translator) based on LLaMAX on Hugging Face. Thanks to [Vila](https://twitter.com/anniodance/status/1816393184336806311?s=61&t=Ildb-_VXuT69jEQUnzS7YA) for creating this awesome demo!**

**🔥[Jul 6, 2024] Released the multilingual math reasoning model [LLaMAX2-7B-MetaMath](https://huggingface.co/LLaMAX/LLaMAX2-7B-MetaMath), only trained on English MGSM dataset**

**🔥[Jul 6, 2024] Released the multilingual natural language inference model [LLaMAX2-7B-XNLI](https://huggingface.co/LLaMAX/LLaMAX2-7B-XNLI), only trained on English MultiNLI dataset**

**🔥[Jul 6, 2024] Released the multilingual commonsense reasoning model [LLaMAX2-7B-X-CSQA](https://huggingface.co/LLaMAX/LLaMAX2-7B-X-CSQA), only trained on five English commonsense reasoning datasets, including X-CSQA, ARC-Easy, ARC-Challenge, OpenBookQA, and QASC.**

**🔥[Jul 6, 2024] Released the multilingual instruction-tuned models [LLaMAX2-7B-Alpaca](https://huggingface.co/LLaMAX/LLaMAX2-7B-Alpaca), [LLaMAX3-8B-Alpaca](https://huggingface.co/LLaMAX/LLaMAX3-8B-Alpaca), only trained on English instruction data [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)**

**🔥[Jul 6, 2024] Released the multilingual base models [LLaMAX2-7B](https://huggingface.co/LLaMAX/LLaMAX2-7B), [LLaMAX3-8B](https://huggingface.co/LLaMAX/LLaMAX3-8B)**

## Model Download
We implement multiple versions of the LLaMAX model, the model links are as follows:

| Model          | Description                                                                                                                                          | HuggingFace Model Path                                                   |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| LLaMAX2-7B | base model                                                                                                                                           | [LLaMAX2-7B](https://huggingface.co/LLaMAX/LLaMAX2-7B)                   |
| LLaMAX3-8B | base model                                                                                                                                           | [LLaMAX3-8B](https://huggingface.co/LLaMAX/LLaMAX3-8B)                   |
| LLaMAX2-7B-Alpaca | instruction-tuned model, trained on Alpaca data                                                                                                      | [LLaMAX2-7B-Alpaca](https://huggingface.co/LLaMAX/LLaMAX2-7B-Alpaca)     |
| LLaMAX3-8B-Alpaca| instruction-tunedmodel, trained on Alpaca data                                                                                                       | [LLaMAX3-8B-Alpaca](https://huggingface.co/LLaMAX/LLaMAX3-8B-Alpaca)     |
| LLaMAX2-7B-X-CSQA | commonsense reasoning model | [LLaMAX/LLaMAX2-7B-X-CSQA](https://huggingface.co/LLaMAX/LLaMAX2-7B-X-CSQA) |
| LLaMAX2-7B-XNLI | natural language inference model                                                                                       | [LLaMAX2-7B-XNLI](https://huggingface.co/LLaMAX/LLaMAX2-7B-XNLI)         |
| LLaMAX2-7B-MetaMath | math reasoning model                                                                                                   | [LLaMAX2-7B-MetaMath](https://huggingface.co/LLaMAX/LLaMAX2-7B-MetaMath) |

## Results
**Note that all the following results are obtained using the zero-shot setting. If you want to reproduce our model's results on translation tasks, you can refer to [**this tutorial**](./scripts/README.md). For the commonsense reasoning, natural language inference and math reasoning tasks, you can use evaluation scripts from [this repo](https://github.com/CONE-MT/MindMerger).** 

LLaMAX2-Alpaca achieves an average spBLEU score improvement of over 10 points compared to the corresponding LLaMA2-Alpaca model on the Flores-101 dataset.
We also evaluate other LLMs emphasizing multilingual capabilities and translation models. 
The translation ability of our model is significantly higher than other LLMs and on par with the strongest translation models. For a detailed analysis, please refer to Table 4 in [our paper](http://arxiv.org/abs/2407.05975).

![other models' results](./images/other_models_results.png "other models' results")

We evaluate the languages in Flores-200 that are not covered by the training data (unseen). 
As shown in the left figure, our model still shows significant improvements.

For more downstream tasks, we fine-tuned LLaMAX using only the English training set, which also shows significant improvements in non-English. 
We provide evaluation results in the right figure for multilingual testset of the following three tasks: 
Commonsense Reasoning, Natural Language Inference and Math Reasoning.



<p float="left">
  <img src="./images/flores200_results.png" alt="flores200 results" width="45%" />
  <img src="./images/other_tasks.png" alt="other tasks results" width="45.5%" />
</p>

## Supported Languages
[language list](./supported_languages.csv)

<a name="citation"></a>
## Citation
If our model helps your work, please cite this paper:
```
@inproceedings{lu-etal-2024-llamax,
    title = "{LL}a{MAX}: Scaling Linguistic Horizons of {LLM} by Enhancing Translation Capabilities Beyond 100 Languages",
    author = "Lu, Yinquan  and
      Zhu, Wenhao  and
      Li, Lei  and
      Qiao, Yu  and
      Yuan, Fei",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.631",
    doi = "10.18653/v1/2024.findings-emnlp.631",
    pages = "10748--10772",
    abstract = "Large Language Models (LLMs) demonstrate remarkable translation capabilities in high-resource language tasks, yet their performance in low-resource languages is hindered by insufficient multilingual data during pre-training. To address this, we conduct extensive multilingual continual pre-training on the LLaMA series models, enabling translation support across more than 100 languages. Through a comprehensive analysis of training strategies, such as vocabulary expansion and data augmentation, we develop LLaMAX. Remarkably, without sacrificing its generalization ability, LLaMAX achieves significantly higher translation performance compared to existing open-source LLMs (by more than 10 spBLEU points) and performs on-par with specialized translation model (M2M-100-12B) on the Flores-101 benchmark. Extensive experiments indicate that LLaMAX can serve as a robust multilingual foundation model. The code and the models are publicly available.",
}
```
