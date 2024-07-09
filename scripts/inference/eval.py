import yaml
import os
import json
import argparse
import importlib.util
import getpass

def read_yaml_file(file_path):
    cwd = os.getcwd()
    print(cwd)
    with open(file_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            
class YamlReader():

    def __init__(self, yamlf):

        if os.path.exists(yamlf):
            self.yamlf = yamlf
        else:
            raise FileExistsError("文件不存在!")
        self._data = None  # 私有属性

    @property
    def data(self):
        if not self._data:
            with open(self.yamlf, 'rb') as f:
                self._data = list(yaml.safe_load_all(f))
        return self._data[0]

class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def map_nested_dicts(ob):
        if isinstance(ob, dict):
            return AttributeDict({k: AttributeDict.map_nested_dicts(v) for k, v in ob.items()})
        elif isinstance(ob, list):
            return [AttributeDict.map_nested_dicts(i) for i in ob]
        else:
            return ob

def load_module_from_path(path):
    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_args(parser):
    parser.add_argument('--cfg', type=str, help='config file')
    parser.add_argument('--base_model', type=str, default=None, help='model name in the hub or local path')
    parser.add_argument('--llama_type', type=str, default=None, help='choose from llama2 or llama3')
    parser.add_argument('--project_path', type=str, default="", help='project path')

    return parser.parse_args()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = get_args(parser)

    config = read_yaml_file(args.cfg)
    config = AttributeDict.map_nested_dicts(config)
    
    if args.base_model is not None:
        config.model_path.base_model = args.base_model
    if args.llama_type is not None:
        config.model_path.llama_type = args.llama_type

    print(config, flush=True)
    data_process_func =  load_module_from_path(f"{args.project_path}src/datasets/{config.dataset.loader}.py").dataloader
    
    test_process_func = load_module_from_path(f"{args.project_path}src/tasks/{config.dataset.loader}.py").test_process
    
    data_dict = data_process_func(config)
    test_process_func(config, data_dict)
