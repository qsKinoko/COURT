# Masking and Spliting

import json,argparse,os
import re
import random
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser("Data processing for experiments.")

    parser.add_argument('--text_data_path', type=str, default="", help='Path to processed paragraph.')

    return parser.parse_args()


patterns_act = [r'\b(Log on|Log off|Connect the device|Disconnect the device|Send external email|View external email|Send internal email|View internal email|Copy file|Delete file|Write file|Open file|Download information|Upload information)\b']  # 1. Mask 行为类型（前两个或前三个词）

patterns_attri = [
        r'\b(own|colleague\'s|supervisor\'s)\b',#, # 4. Mask 'on/with' 和 'pc' 中间的那个词
        r'\b(\d+)\b' #次数
        #下面两项不是mask，是在相应位置添加MASK标志
        #r'\b(about job application|job application pages|harmful pages)\b', # 2. Mask 'job application' 和 'harmful'
        #r'\bGo off duty. \b', # 3. Mask 'Go off duty'
    ]

mask_rate = 0.6
#act_mask_rate = 0.6 # 0.2

# 定义一个函数，按概率 MASK 内容
def conditional_attri_mask(match): # 行为属性掩蔽
    text = match.group(0)
    # 以 0.6 的概率 MASK 其他内容
    return "[MASK]" if random.random() < mask_rate else text

def conditional_act_mask(match): # 行为类型掩蔽 
    text = match.group(0)
    # 以 0.6 的概率 MASK 其他内容
    return "[MASK]" if random.random() < mask_rate else text 

def conditional_insert_mask(match): # 行为属性掩蔽02(mask插入)
    text = match.group(0)
    # 以 0.6 的概率 MASK 其他内容
    return  '[MASK] '+text if random.random() < mask_rate else text

def item_masking(content):
    masked_content = content

    # 添加纯MASK，用于about job application|job application pages|harmful pages
    masked_content = re.sub(r'\b(on own pc|on colleague\'s pc|on supervisor\'s pc)\b', conditional_insert_mask, masked_content)

    for pattern in patterns_attri:
        masked_content = re.sub(pattern, conditional_attri_mask, masked_content)

    for pattern in patterns_act:
        masked_content = re.sub(pattern, conditional_act_mask, masked_content)
    
    return masked_content

