from openai import OpenAI
# import openai
from GPT_API.Model import Model_API
from urllib.parse import urljoin

import requests
import json, random, copy, re


class GPT(Model_API):
    def __init__(self, config,instruction=[{"role": "system", "content": "You are a helpful assistant."}]):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.client = OpenAI(api_key=api_keys[api_pos])
        # openai.api_key = api_keys[api_pos]
        
        self.api_url = config["api_key_info"]["api_url"]
        self.api_domain = config["api_key_info"]["api_domain"]
        self.api_key = api_keys[api_pos]

        self.instruction = instruction[0]# [{"role": "system", "content": "You are a helpful assistant."}]
        self.history =  instruction# [{"role": "system", "content": "You are a helpful assistant."}]
        self.history_backup = None
        #self.backup = []

    def query(self, msg):
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            'model': self.name,
            'messages': self.history + [
                {"role": "user", "content": msg}
            ],
            #"stream": True,
            'response_format':{
                'type': 'json_object'
            },
            'max_tokens': self.max_output_tokens,
            'temperature': self.temperature
        }
        #print(data)

        try:
            # 发送一次性请求，获取完整响应
            self.history_backup = None
            response = requests.post(self.api_url, headers=headers, data=json.dumps(data),timeout=300)
            response.raise_for_status()  # 检查请求是否成功
            #response_data = b""  # 用于存储完整的 JSON 数据

            #print(f"响应状态码: {response.status_code}")
            #print(f"响应头: {response.headers}")
            #print(f"响应内容: {response.text}")  # 打印原始响应
            # 解析响应 JSON
            completion = response.json()
            #completion = json.loads(response_data.decode('utf-8'))
            #print(completion)
            response_text = completion['choices'][0]['message']['content']
            response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)


            self.history_backup = copy.deepcopy(self.history)

            self.history.append({"role": "user", "content": msg})
            self.history.append({"role": "assistant", "content": response_text})
            # self.backup.append([{"role": "user", "content": msg},{"role": "assistant", "content": response_text}])

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            response_text = ""
        except KeyError:
            print("Unexpected response format.")
            response_text = ""

        #print(response_text)
        return response_text
    
    def roll_update_history(self):
        #print(self.instruction)
        if len(self.history)>7:
            self.history = [self.instruction]+self.history[-6:]
            #random_lists = random.sample(self.backup, 5)
            #self.history = [self.instruction] + [item for sublist in random_lists for item in sublist]
            #print(self.history)

    def history_rollback(self):
        #print(self.instruction)
        if self.history_backup is not None:
            self.history = copy.deepcopy(self.history_backup)
    
    def clean_history(self):
        self.history = [self.instruction] #if self.instruction is not None else []
    
    def get_instruction(self):
        return self.instruction

    def set_instruction(self,instruction):
        self.instruction = instruction
