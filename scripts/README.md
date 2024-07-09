# Inference
If you want to reproduce our model's results on Flores-101 from our main experiment table, you can refer to the following steps.
## Environment
We used Python 3.9 to run our inference code on A100. Please follow the steps below to install the Python environment.
```sh
pip install -r requirements.txt
```
## Data
```sh
cd ../../data/
wget https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz
tar -xvzf flores101_dataset.tar.gz
```

## Configure
You can adjust the following parameters in the config file(./configs/flores101.yaml) according to your needs.
```sh
model_path: 
   base_model: ../../model/LLaMAX2-7B-Alpaca/  ## your model path
   lora: null
   torch_dtype: bf16  ## torch dtype
   llama_type: llama2  ## llama2 or llama3 model
dataset:
   loader: flores101 ## dataset name
   path: ../../data//flores101_dataset/devtest/ ## dataset root
   input_file: eng.devtest ## data path
   inst_file: ./prepare/instruct_inf.txt  ## instruction file
   lang_instruction: ./prepare/language_name_code_pair.xlsx  ## a table of language IDs and their full names
   input_extra_file: null
   lang_pair: en-zh  ## translation pair
   inst_placeholder: null
   inst_fix: true
   inst_index: 0
   labels: null
   inst_with_input: true 
   input_size: -1
   use_match: false

generat:
    beam_size: 4   ## beam size
    temperature: 0.1
    batch_size: 8  ## batch size
    template: prompt_input
    search: beam 
    do_sample: false
    
output:
    subpath: ""
    output_file_prefix: generation_results
 

```

## Usage
```sh
cd inference
python eval.py --cfg ./configs/flores101.yaml
```