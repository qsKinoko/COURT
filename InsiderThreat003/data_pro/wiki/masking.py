# Masking and Spliting

import json,argparse,os
import re
import random
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser("Data processing for experiments.")
    parser.add_argument('--data_path', type=str, default="", help='Path to Dataset.')
    parser.add_argument('--prod_data_path', type=str, default="", help='Path to processed data.')
    parser.add_argument('--text_data_path', type=str, default="", help='Path to processed paragraph.')

    return parser.parse_args()


## masking

patterns = [
    r'\b(reverted \d+ times?)\b',                     # 匹配 "reverted x time" 或 "reverted x times"
    r'\b(reverted \d+ times? and subsequently reverted)\b',  # 匹配完整的组合形式
    r'\b(subsequently reverted)\b',                  # 匹配 "subsequently reverted"
    r'\b(\d+ hops?)\b',                              # 匹配 "x hop" 或 "x hops"
    r'\b(\d+ (second|minute|hour|seconds|minutes|hours))\b',  # 匹配单数或复数时间单位
    r'\b(meta page)\b',                              # 匹配 "meta page"
]

# 定义一个函数，按概率 MASK 内容
def conditional_mask(match):
    text = match.group(0)
    #if "meta page" in text:  # 强制 MASK meta page
    #    return "[MASK]"
    # 以 0.8 的概率 MASK 其他内容
    return "[MASK]" if random.random() < 0.6 else text

def item_masking(content):
    masked_content = content
    for pattern in patterns:
        masked_content = re.sub(pattern, conditional_mask, masked_content)
    return masked_content


