from GPT_API.GPT import GPT
from GPT_API.load_config import gpt4_config
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os,argparse,re,json,time,random,glob


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--text_data_path', type=str, default="", help='Path to processed paragraph.')
    
    parser.add_argument('--output_dir', type=str, default="", help='Path to processed paragraph.')

    parser.add_argument('--max_length', type=int, default=500, help='Path to processed paragraph.')

    parser.add_argument('--user', type=str, default='', help='Users to be tested.')

    return parser.parse_args()


class ARBITER:
    def __init__(self,strategy=None):
        instr = "You are a professional arbiter for employee work log detection. I have two detection systems: one based on anomaly detection and the other based on misuse detection. These systems have produced conflicting results. I will provide you with the known threat scenarios, the original test log, the anomaly detection system's result and explanation, and the misuse detection system's result and explanation. Using this information, your task is to confirm the final detection result. Note that while insights from the anomaly detection system are still valuable, the misuse detection system's input should carry greater weight in your decision-making process.\n"
        instr += "You need to determine which of the following categories the test log belongs to:\n1) High Threat: Logs that align with known threat scenarios and have sufficient evidence of threatening activities; 2) Low Threat: Logs that do not fully align with known threat scenarios but potentially containing threatening activities; 3)Benign Anomaly: Logs that show anomalies but no signs of threat; 4) Benign Activity: Logs with no anomalies or signs of threat; 5) Unknown Threat: Logs that contain threats but do not align with known threat scenarios. This category should be considered when the misuse detection system classifies the log as benign, while the anomaly detection system classifies it as a threat.\n\n"
        instr += f"The known threat scenarios:\n{strategy}\n\n"
        instr += "Provide your answer directly along with an breif explanation. Present your answer in JSON format using the following keys: Judgement, Explanation. Let's work this out in a step by step way to be sure we have the right answer."

        self.gpt = GPT(gpt4_config,[{"role": "system",  "content": instr}])

           
    def arbit_experiments(self):
        args = get_parser()

        prompts = "The original test log:\n%s\n\n"
        prompts += "The anomaly detection system's result and explanation:\n%s\n%s\n\n"
        prompts += "The misuse detection system's result and explanation:\n%s\n%s\n\n"
        prompts += "Your answer:\n"
        #print(prompts)

        test_result_file = open(glob.glob(os.path.join(args.output_dir,args.user+"_test*"))[0],'r')
        gan_test_result_file = open(glob.glob(os.path.join(args.output_dir,args.user+"_gan_test*"))[0],'r')
        ground_truth_file = open(os.path.join(args.text_data_path,args.user+"_test.jsonl"),'r')

        current_time = time.strftime("%Y-%m-%d_%H%M%S")
        arbitration_result_record = open(os.path.join(args.output_dir,args.user+f"_arbitration_{current_time}.jsonl"),'w')


        test_results = test_result_file.readlines()
        gan_test_results = gan_test_result_file.readlines()
        ground_truths = ground_truth_file.readlines()

        results = []
        preds = []
        preds_strict = []
        threat_keyword = ["High Threat","Low Threat"]
        threat_keyword_strict = ["High Threat"]

        for test_result,gan_test_result,ground_truth in tqdm(zip(test_results,gan_test_results,ground_truths),ncols=100):
            test_result = json.loads(test_result)
            gan_test_result = json.loads(gan_test_result)
            ground_truth = json.loads(ground_truth)
            #print(len(results),len(preds))
            assert gan_test_result["Content"] == test_result["Content"], "Content not match!"
            if gan_test_result["Judgement"] != test_result["Judgement"]:
                # 开启仲裁
                response_text = ""
                anomaly_result = "Threat" if gan_test_result["Judgement"]=="Yes" else "Benign"
                misuse_result = "Threat" if test_result["Judgement"]=="Yes" else "Benign"
                while response_text=="":
                    response_text = self.gpt.query(prompts % (ground_truth["content"],anomaly_result,gan_test_result["Explanation"],misuse_result,test_result["Explanation"]))
                    if response_text!="":
                        #self.gpt.clear_history() # 清空历史
                        self.gpt.roll_update_history() # 滚动更新历史
                    json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        arbit_result = json.loads(json_str)
                        print(arbit_result)
                        if arbit_result["Judgement"] in threat_keyword:
                            preds.append(1)
                        else:
                            preds.append(0)
                        if arbit_result["Judgement"] in threat_keyword_strict:
                            preds_strict.append(1)
                        else:
                            preds_strict.append(0)
                        
                        arbit_result["content"] = ground_truth["content"]
                        arbit_result["label"] = ground_truth["label"]
                        json.dump(arbit_result, arbitration_result_record)
                        arbitration_result_record.write('\n')
                        arbitration_result_record.flush()
                        break

                    else:
                        print("Non-formatted output")
                        response_text = ""
                        continue

            else:
                preds.append(1 if test_result["Judgement"]=="Yes" else 0)
                preds_strict.append(1 if test_result["Judgement"]=="Yes" else 0)

            results.append(int(ground_truth["label"]))
            
        
        arbitration_result_record.close()
        accuracy = accuracy_score(results, preds)
        precision = precision_score(results, preds)
        recall = recall_score(results, preds)
        f1 = f1_score(results, preds)
        print("Relaxed Results:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
        print('\n')

        accuracy = accuracy_score(results, preds_strict)
        precision = precision_score(results, preds_strict)
        recall = recall_score(results, preds_strict)
        f1 = f1_score(results, preds_strict)
        print("Strict Results:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
    
    def arbit_single(self,sample):

        prompts = "The original test log:\n%s\n\n"
        prompts += "The anomaly detection system's result and explanation:\n%s\n%s\n\n"
        prompts += "The misuse detection system's result and explanation:\n%s\n%s\n\n"
        prompts += "High-priority special note: The employee is approved to work overtime today. All after-hours activities, including the use of USB devices, are authorized and compliant with company regulations.\nThis is unforeseen information that the anomaly detection system and misuse detection system are not aware of." if sample["Experimental Type"] == "Special Note" else ""
        prompts += "Your answer:\n"

        response_text = self.gpt.query(prompts % (sample["Content"],sample["Anomaly Detection"],sample["Anomaly Explanation"],sample["Misuse Detection"],sample["Misuse Explanation"]))
        return response_text

    def adaptability_evaluate(self):
        args = get_parser()

        prompts = "The original test log:\n%s\n\n"
        prompts += "The anomaly detection system's result and explanation:\n%s\n%s\n\n"
        prompts += "The misuse detection system's result and explanation:\n%s\n%s\n\n"
        prompts += "Your answer:\n"
        #print(prompts)

        gan_test_result_file = open(glob.glob(os.path.join(args.output_dir,args.user+"_gan_test*"))[0],'r')

        current_time = time.strftime("%Y-%m-%d_%H%M%S")
        arbitration_result_record = open(os.path.join(args.output_dir,args.user+f"_adapt_{current_time}.jsonl"),'w')

        gan_test_results = gan_test_result_file.readlines()

        results = []
        preds = []
        threat_keyword = ["Unknown Threat"]

        for gan_test_result in tqdm(gan_test_results,ncols=100):
            gan_test_result = json.loads(gan_test_result)
            #print(len(results),len(preds))
            if gan_test_result["Judgement"] == "Yes":# and gan_test_result["Evaluation"] == "Correct":
                # 开启仲裁
                response_text = ""
                anomaly_result = "Threat" if gan_test_result["Judgement"]=="Yes" else "Benign"
                while response_text=="":
                    response_text = self.gpt.query(prompts % (gan_test_result["Content"],anomaly_result,gan_test_result["Explanation"],"Benign","None"))
                    if response_text!="":
                        #self.gpt.clear_history() # 清空历史
                        self.gpt.roll_update_history() # 滚动更新历史
                    json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        arbit_result = json.loads(json_str)
                        print(arbit_result)
                        if arbit_result["Judgement"] in threat_keyword:
                            preds.append(1)
                        else:
                            preds.append(0)
                        
                        arbit_result["content"] = gan_test_result["Content"]
                        arbit_result["label"] = 1 if gan_test_result["Evaluation"] == "Correct" else 0
                        json.dump(arbit_result, arbitration_result_record)
                        arbitration_result_record.write('\n')
                        arbitration_result_record.flush()
                        break

                    else:
                        print("Non-formatted output")
                        response_text = ""
                        continue

                results.append(1 if gan_test_result["Evaluation"] == "Correct" else 0)
            
        
        arbitration_result_record.close()
        accuracy = accuracy_score(results, preds)
        precision = precision_score(results, preds)
        recall = recall_score(results, preds)
        f1 = f1_score(results, preds)
        print("Relaxed Results:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")


strategy = ""

arbiter = ARBITER(strategy)
arbiter.arbit_experiments()
