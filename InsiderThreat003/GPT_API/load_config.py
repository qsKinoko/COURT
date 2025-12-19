import json,os
GPT_VERSION = "GPT_API/gpt4_config.json"
#GPT_VERSION = "GPT_API/dpr1_config.json"

CONFIG_PATH = os.path.join(os.getcwd(),GPT_VERSION)
#print(os.getcwd())
#print(CONFIG_PATH)
gpt4_config = json.load(open(CONFIG_PATH,'r'))
