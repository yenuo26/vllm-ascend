import pytest
import os
import yaml

def load_config():
    """读取配置文件"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'configs', 'test_config.yaml')
    print(config_path)
    config_path = os.getenv('TEST_CONFIG', config_path)
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'配置文件不存在：{config_path}')
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("no config find")