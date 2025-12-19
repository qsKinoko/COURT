from GPT_API.GPT import GPT
from GPT_API.load_config import gpt4_config
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os,argparse,re,json,time,random
from data_pro.wiki.masking import item_masking


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--text_data_path', type=str, default="", help='Path to processed paragraph.')
    
    parser.add_argument('--output_dir', type=str, default="", help='Path to processed paragraph.')


    parser.add_argument('--max_length', type=int, default=500, help='Path to processed paragraph.')

    return parser.parse_args()


class LLM_GAN:
    def __init__(self,strategy=None):
        
        instr_de = [{"role": "system",  "content": f'''You are now a Wiki user edit log detector. Your task is to determine whether the Wiki user edit logs I provide are vandalic or not, based on the judgment strategy. You should respond with "Yes" or "No" and provide a brief explanation for your decision. 

Judgement Strategy:
{strategy}

Provide your answer in JSON format directly with the following keys: Judgement, Explanation. Let's work this out in a step by step way to be sure we have the right answer.'''}] # for detection
        
        #instr_de = [{"role": "system",  "content": '''Determine whether the logs I provide are abnormal or not. You should respond with "Yes" or "No". Provide your answer in JSON format directly with the following keys: Judgement.'''}] # for gpt only
        
        instr_d = [{"role": "system",  "content": '''You are now a Wiki user edit log detector. Your task is to determine whether the Wiki user edit logs I provide are normal or not, based on the judgment strategy. You should respond with "Yes" or "No" and provide a brief explanation for your decision. If you make an incorrect judgment, I will inform you. You should then analyze the reason for your error and summarize the patterns of benign user edit behavior again as concisely as possible for your updated judgment strategies.
Provide your detection result only if I ask for it in JSON format (e.g. ```json\n \{Your Answer\} \n```) directly with the following keys: Judgement, Explanation. Provide your updated judgment strategy only if I ask for it in JSON format (e.g. ```json\n \{Your Answer\} \n```) directly with the following keys: Strategy. Do not provide both unless explicitly requested. Let's work this out in a step by step way to be sure we have the right answer.'''}] # for discriminator

        instr_g = [{"role": "system",  "content": '''You are now a generator of Wiki user edit logs. Your task is to generate Wiki user edit logs by filling in the masked portions of the log entries I provide, based on the generation strategy. Your goal is to make the generated Wiki user edit logs as close to authentic Wiki edit logs as possible.  I will provide you with feedback and suggestions to help you create more realistic logs.  Based on my feedback, you should briefly update your generation strategy accordingly. 

Primary Generation Strategy:
The masked portions may include the following elements: 1) Edit Reversion Status: reverted \d times, reverted \d times and subsequently reverted, subsequently reverted; 2) Page Hops: \d hops; 3) Time Interval: \d seconds, \d minutes, \d hours; 4) Keyword: 'page'.

Provide your generation result only if I ask for it in JSON format (e.g. ```json\n \{Your Answer\} \n```) directly with the following keys: Generation. Provide your updated generation strategy only if I ask for it in JSON format (e.g. ```json\n \{Your Answer\} \n```) directly with the following keys: Strategy. Do not provide both unless explicitly requested. Let's work this out in a step by step way to be sure we have the right answer.'''}]

        self.gpt_g = GPT(gpt4_config,instr_g)
        self.gpt_d = GPT(gpt4_config,instr_d)
        self.gpt_de = GPT(gpt4_config,instr_de)
        self.gpt_naive = GPT(gpt4_config)
        
        self.dataset = None
        self.assis_id = None
        self.thread_id = None

    def __del__(self):
        if self.assis_id is not None:
            while not self.gpt.assistant_delete(self.assis_id):
                pass
            self.assis_id = None
        if self.thread_id is not None:
            while self.gpt.thread_delete(self.thread_id):
                pass
            self.thread_id = None

    def detection(self,content):
        response_text = ""
        prompts = "Test Log:\n%s\n\nYour Answer:\n"#"Judgment Strategy:\n%s\n\nTest Log:\n%s\n\nYour Answer:\n"
        #print(prompts)
        while response_text=="":
            response_text = self.gpt_de.query(prompts % content)#(prompts % (strategy,content))
            if response_text!="":
                #self.gpt_de.clean_history() # 清空历史
                self.gpt_de.roll_update_history() # 滚动更新历史
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                print("Non-formatted output")
                response_text = ""
                continue

    def detection_naive(self,content,strategy):
        response_text = ""
        prompts = "Determine whether the logs I provide are abnormal or not. You should respond with \"Yes\" or \"No\". Provide your answer in JSON format directly with the following keys: Judgement.\n\nJudgment Strategy:\n%s\n\nTest Log:\n%s\n\nYour Answer:\n"
        #print(prompts)
        while response_text=="":
            response_text = self.gpt_naive.query(prompts % (strategy,content))
            if response_text!="":
                self.gpt_naive.clean_history() # 清空历史
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match and 'Judgement' in response_text:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                print("Non-formatted output")
                response_text = ""
                continue
        
    '''
    def discriminator_assistant_init(self,strategy="None"):
        self.assis_id = None
        self.thread_id = None
        instruction = f"""You are now a Wiki user edit log detector. Your task is to determine whether the Wiki user edit logs I provide are vandalic or not. You should respond with "Yes" or "No" and provide a brief explanation for your decision. Provide your answer in JSON format directly with the following keys: Judgement, Explanation. Let's work this out in a step by step way to be sure we have the right answer.

Judgment Strategy:
{strategy}
        """
        while self.assis_id is None or self.assis_id=='':
            self.assis_id = self.gpt.assistant_start(instruction)
        while self.thread_id is None or self.thread_id=='':
            self.thread_id = self.gpt.thread_start()
    
    def discriminator_assistant(self,msg):
        run_id = None
        status = None
        response_text = None
        # STEP 01
        while not self.gpt.messages_add(msg,self.thread_id):
            pass
        # STEP 02
        while run_id is None or run_id=='':
            run_id = self.gpt.run_start(self.assis_id,self.thread_id)
        # STEP 03
        while status != "completed":
            status = self.gpt.run_check(self.hread_id,run_id)
        # STEP 04
        while response_text is None or response_text=='':
            response_text = self.gpt.get_message(self.thread_id)
            
        print(response_text)
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        else:
            print("Non-formatted output")
            #raise ValueError("Non-formatted output")
            return None
    '''

    def load_dataset(self,data_path,args):
        # Configure data loader
        #prompt = "下面这段编辑操作日志中是否存在异常操作？\n%s\n使用Yes或No来回答："
        df = pd.read_json(os.path.join(args.text_data_path,data_path), lines=True)
        # 过滤掉 'content' 列中长度超过 MAX_LENGTH 的行
        df = df[df['content'].apply(lambda x: len(x.strip().split(' ')) <= args.max_length)]
        print(len(df))
        #df['content'] = df['content'].apply(lambda x: prompt % x.strip())
        self.dataset = df
    
    def dataloader(self, max_iterations=-1):
        count = 0  # 初始化计数器
        for index, row in self.dataset.iterrows():
            if count >= max_iterations and max_iterations>0:  # 如果达到最大迭代次数，停止循环
                break
            yield index, row  # 返回行索引和行数据
            count += 1  # 增加计数器

    def generator(self,content,strategy="None"):
        response_text = ""
        prompts = "Generation Strategy:\n%s\n\nMasked Log:\n%s\n\nGeneration Result Only:\n"
        while response_text=="":
            response_text = self.gpt_g.query(prompts % (strategy,content))
            if response_text!="":
                #self.gpt_d.clear_history() # 清空历史
                self.gpt_g.roll_update_history() # 滚动更新历史
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                print("Non-formatted output")
                response_text = ""
                continue

    def discriminator(self,content,strategy="None"):
        response_text = ""
        prompts = "Judgment Strategy:\n%s\n\nTest Log:\n%s\n\nDetection Result Only:\n"
        while response_text=="":
            response_text = self.gpt_d.query(prompts % (strategy,content))
            if response_text!="":
                #self.gpt_d.clear_history() # 清空历史
                self.gpt_d.roll_update_history() # 滚动更新历史
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                print("Non-formatted output")
                response_text = ""
                continue
    
    def generator_update(self,feedback):
        response_text = ""
        prompts = "Feedback:\n%s\n\nUpdated Generation Strategy Only:\n"
        while response_text=="":
            response_text = self.gpt_g.query(prompts % feedback)
            if response_text!="":
                #self.gpt_g.clear_history() # 清空历史
                self.gpt_g.roll_update_history() # 滚动更新历史
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                print("Non-formatted output")
                response_text = ""
                continue
    
    def discriminator_update(self,feedback):
        response_text = ""
        prompts = "Feedback:\n%s\n\nShort benign user edit behavior pattern summarization only:\n"
        while response_text=="":
            response_text = self.gpt_d.query(prompts % feedback)
            if response_text!="":
                #self.gpt_d.cldear_history() # 清空历史
                self.gpt_d.roll_update_history() # 滚动更新历史
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                print("Non-formatted output")
                response_text = ""
                continue


def common_test(strategy=''):
    args = get_parser()
    
    model = LLM_GAN(strategy)
    model.load_dataset("new_test.jsonl",args)
    ground_truth = []
    pred = []
    current_time = time.strftime("%Y-%m-%d_%H%M%S")
    results_record = open(os.path.join(args.output_dir,f"test_{current_time}.jsonl"),'w')
    i = 0
    #model.discriminator_assistant_init(strategy)
    for index, row in tqdm(model.dataloader(),ncols=100):
        ground_truth.append(row["label"])
        result = model.detection(row["content"])
        #result = model.discriminator_assistant(row["content"])
        judgement = result["Judgement"]
        print(judgement,row["label"])
        pred.append(1 if "Yes" in judgement else 0)
        i += 1
        result["Evaluation"] = "Correct" if pred[-1]==ground_truth[-1] else "Wrong"
        result["Content"] = row["content"]
        json.dump(result, results_record)
        results_record.write('\n')
        if i%10==0:
            print('acc:',accuracy_score(ground_truth, pred))
            print('f1:',f1_score(ground_truth, pred))

    results_record.close()
    accuracy = accuracy_score(ground_truth, pred)
    precision = precision_score(ground_truth, pred)
    recall = recall_score(ground_truth, pred)
    f1 = f1_score(ground_truth, pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    
def anomaly_gan():
    args = get_parser()
    model = LLM_GAN()
    model.load_dataset("new_train.jsonl",args)

    current_time = time.strftime("%Y-%m-%d_%H%M%S")
    results_record_d = open(os.path.join(args.output_dir,f"test_strategy_d_{current_time}.jsonl"),'w')
    results_record_g = open(os.path.join(args.output_dir,f"test_strategy_g_{current_time}.jsonl"),'w')
    gen_data_record = open(os.path.join(args.output_dir,f"test_gen_data_{current_time}.jsonl"),'w')

    strategy_d = "None"
    strategy_g = "None"
    for index, row in tqdm(model.dataloader(250),ncols=100):
        content = row["content"]
        gen_data = model.generator(item_masking(content),strategy_g)["Generation"]
        result_d4g = model.discriminator(gen_data,strategy_d)
        # Judgement, Explanation
        if result_d4g['Judgement']=="Yes": # 认为生成日志正常，更新判别器
            strategy_d = model.discriminator_update("Incorrect.")["Strategy"]
        elif result_d4g['Judgement']=="No" and random.random() < 0.03: # 生成日志被判别为异常，更新生成器
            strategy_g = model.generator_update(result_d4g["Explanation"])["Strategy"]
        
        result_d4c = model.discriminator(content,strategy_d)
        if result_d4c['Judgement']=="No": # 认为正常日志异常，更新判别器
            strategy_d = model.discriminator_update("Incorrect.")["Strategy"]

        model.gpt_d.clean_history()
        model.gpt_g.clean_history()
        json.dump(strategy_d, results_record_d)
        json.dump(strategy_g, results_record_g)
        json.dump(gen_data, gen_data_record)
        results_record_d.write('\n')
        results_record_g.write('\n')
        gen_data_record.write('\n')
        results_record_d.flush()
        results_record_g.flush()
        gen_data_record.flush()
    
    results_record_d.close()
    results_record_g.close()
    gen_data_record.close()
    print('strategy_g:',strategy_g)
    print('strategy_d:',strategy_d)

    common_test(strategy_d)
    
def naive_test():
    args = get_parser()
    
    strategy = ""
    
    model = LLM_GAN()
    model.load_dataset("new_test.jsonl",args)
    ground_truth = []
    pred = []
    current_time = time.strftime("%Y-%m-%d_%H%M%S")
    results_record = open(os.path.join(args.output_dir,f"test_{current_time}.jsonl"),'w')
    i = 0
    #model.discriminator_assistant_init(strategy)
    for index, row in tqdm(model.dataloader(),ncols=100):
        ground_truth.append(row["label"])
        result = model.detection_naive(row["content"],strategy)
        #result = model.discriminator_assistant(row["content"])
        judgement = result["Judgement"]
        print(judgement,row["label"]  )
        results_record.write('\n')
        pred.append(1 if "Yes" in judgement else 0)
        i += 1
        result["Evaluation"] = "Correct" if pred[-1]==ground_truth[-1] else "Wrong"
        result["Content"] = row["content"]
        json.dump(result, results_record)
        if i%10==0:
            print('acc:',accuracy_score(ground_truth, pred))
            print('f1:',f1_score(ground_truth, pred))

    results_record.close()
    accuracy = accuracy_score(ground_truth, pred)
    precision = precision_score(ground_truth, pred)
    recall = recall_score(ground_truth, pred)
    f1 = f1_score(ground_truth, pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")

if __name__=="__main__":
    #anomaly_gan()
    naive_test()
    #strategy = ""
    #common_test(strategy)