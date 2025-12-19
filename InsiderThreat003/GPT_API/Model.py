import random
import os
#import torch
import numpy as np


class Model_API:
    def __init__(self, config):
        self.provider = config["model_info"]["provider"]
        self.name = config["model_info"]["name"]
        self.seed = int(config["params"]["seed"])
        self.temperature = float(config["params"]["temperature"])
        self.gpus = [str(gpu) for gpu in config["params"]["gpus"]]
        #self.initialize_seed()
        #if len(self.gpus) > 0:
        #    self.initialize_gpus()

    def print_model_info(self):
        print(f"{'-'*len(f'| Model name: {self.name}')}\n| Provider: {self.provider}\n| Model name: {self.name}\n{'-'*len(f'| Model name: {self.name}')}")

    def set_API_key(self):
        raise NotImplementedError("ERROR: Interface doesn't have the implementation for set_API_key")
    
    def query(self):
        raise NotImplementedError("ERROR: Interface doesn't have the implementation for query")
    