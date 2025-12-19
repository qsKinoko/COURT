from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os,argparse,re,json,time,random,glob


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--text_data_path', type=str, default="", help='Path to processed paragraph.')
    
    parser.add_argument('--output_dir', type=str, default="", help='Path to processed paragraph.')

    parser.add_argument('--max_length', type=int, default=500, help='Path to processed paragraph.')

    parser.add_argument('--mask_rate', type=float, default=0.7, help='Users to be tested.')
    parser.add_argument('--update_rate', type=float, default=0.03, help='Users to be tested.')

    parser.add_argument('--users', type=str, default=[], help='Users to be tested.')

    return parser.parse_args()

def cert_evaluation_arbitration():
    args = get_parser()

    results = []
    preds = []
    preds_strict = []
    threat_keyword = ["High Threat","Low Threat"]
    threat_keyword_strict = ["High Threat"]
    for user in args.users:
        print(user)
        test_result_file = open(glob.glob(os.path.join(args.output_dir,user+"_test*"))[0],'r')
        gan_test_result_file = open(glob.glob(os.path.join(args.output_dir,user+"_gan_test*"))[0],'r')
        arbit_result_file = open(glob.glob(os.path.join(args.output_dir,user+"_arbitration*"))[0],'r')
        ground_truth_file = open(os.path.join(args.text_data_path,user+"_test.jsonl"),'r')


        test_results = test_result_file.readlines()
        gan_test_results = gan_test_result_file.readlines()
        arbit_results = arbit_result_file.readlines()
        arbit_idx = 0
        ground_truths = ground_truth_file.readlines()

        for test_result,gan_test_result,ground_truth in tqdm(zip(test_results,gan_test_results,ground_truths),ncols=100):
            test_result = json.loads(test_result)
            gan_test_result = json.loads(gan_test_result)
            ground_truth = json.loads(ground_truth)
            #print(len(results),len(preds))
            assert gan_test_result["Content"] == test_result["Content"], "Not Align!"
            assert test_result["Content"] == ground_truth["content"], "Not Align!"
            
            if gan_test_result["Judgement"] != test_result["Judgement"]:
                # 开启仲裁
                arbit_result = json.loads(arbit_results[arbit_idx])
                arbit_idx += 1

                assert test_result["Content"] == arbit_result["content"], "Not Align!"

                if arbit_result["Judgement"] in threat_keyword:
                    preds.append(1)
                else:
                    preds.append(0)
                if arbit_result["Judgement"] in threat_keyword_strict:
                    preds_strict.append(1)
                else:
                    preds_strict.append(0)
                
            else:
                preds.append(1 if test_result["Judgement"]=="Yes" else 0)
                preds_strict.append(1 if test_result["Judgement"]=="Yes" else 0)

            results.append(int(ground_truth["label"]))    
        
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


def cert_evaluation_gan():
    args = get_parser()

    results = []
    preds = []
    for user in args.users:
        print(user)
        #test_result_file = open(glob.glob(os.path.join(args.output_dir,user+"_test*"))[0],'r')
        gan_test_result_file = open(glob.glob(os.path.join(args.output_dir,user+f"_gan_test_update_{args.update_rate}*"))[0],'r')
        ground_truth_file = open(os.path.join(args.text_data_path,user+"_test.jsonl"),'r')


        #test_results = test_result_file.readlines()
        gan_test_results = gan_test_result_file.readlines()
        ground_truths = ground_truth_file.readlines()

        for gan_test_result,ground_truth in tqdm(zip(gan_test_results,ground_truths),ncols=100):
            #test_result = json.loads(test_result)
            gan_test_result = json.loads(gan_test_result)
            ground_truth = json.loads(ground_truth)
            #print(len(results),len(preds))

            assert gan_test_result["Content"] == ground_truth["content"], "Not Align!"

            preds.append(1 if gan_test_result["Judgement"]=="Yes" else 0)
            results.append(int(ground_truth["label"]))
            
      
        
    accuracy = accuracy_score(results, preds)
    precision = precision_score(results, preds)
    recall = recall_score(results, preds)
    f1 = f1_score(results, preds)
    #print("Relaxed Results:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print('\n')

def cert_evaluation_misuse():
    args = get_parser()

    results = []
    preds = []
    for user in args.users:
        print(user)
        test_result_file = open(glob.glob(os.path.join(args.output_dir,user+"_test*"))[0],'r')
        #gan_test_result_file = open(glob.glob(os.path.join(args.output_dir,user+"_gan_test*"))[0],'r')
        ground_truth_file = open(os.path.join(args.text_data_path,user+"_test.jsonl"),'r')


        test_results = test_result_file.readlines()
        #gan_test_results = gan_test_result_file.readlines()
        ground_truths = ground_truth_file.readlines()

        for test_result,ground_truth in tqdm(zip(test_results,ground_truths),ncols=100):
            test_result = json.loads(test_result)
            #gan_test_result = json.loads(gan_test_result)
            ground_truth = json.loads(ground_truth)
            #print(len(results),len(preds))

            assert test_result["Content"] == ground_truth["content"], "Not Align!"

            preds.append(1 if test_result["Judgement"]=="Yes" else 0)
            results.append(int(ground_truth["label"]))
            
      
        
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

def cert_evaluation_gpt():
    args = get_parser()

    results = []
    preds = []
    for user in args.users:
        print(user)
        test_result_file = open(glob.glob(os.path.join(args.output_dir,""+user+"_test*"))[0],'r')
        #gan_test_result_file = open(glob.glob(os.path.join(args.output_dir,user+"_gan_test*"))[0],'r')
        ground_truth_file = open(os.path.join(args.text_data_path,user+"_test.jsonl"),'r')


        test_results = test_result_file.readlines()
        #gan_test_results = gan_test_result_file.readlines()
        ground_truths = ground_truth_file.readlines()

        for test_result,ground_truth in tqdm(zip(test_results,ground_truths),ncols=100):
            test_result = json.loads(test_result)
            #gan_test_result = json.loads(gan_test_result)
            ground_truth = json.loads(ground_truth)
            #print(len(results),len(preds))

            assert test_result["Content"] == ground_truth["content"], "Not Align!"

            preds.append(1 if test_result["Judgement"]=="Yes" else 0)

            results.append(int(ground_truth["label"]))
            
      
        
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

def cert_evaluation_adapt():
    args = get_parser()

    results = []
    preds = []
    threat_keyword = ["Unknown Threat"]
    for user in args.users:
        print(user)
        #test_result_file = open(glob.glob(os.path.join(args.output_dir,user+"_test*"))[0],'r')
        gan_test_result_file = open(glob.glob(os.path.join(args.output_dir,user+"_adapt*"))[0],'r')
        #ground_truth_file = open(os.path.join(args.text_data_path,user+"_test.jsonl"),'r')


        #test_results = test_result_file.readlines()
        gan_test_results = gan_test_result_file.readlines()
        #ground_truths = ground_truth_file.readlines()

        for test_result in tqdm(gan_test_results,ncols=100):
            test_result = json.loads(test_result)
            #print(test_result["Judgement"],threat_keyword,(1 if test_result["Judgement"] in threat_keyword else 0))
            preds.append(1 if test_result["Judgement"] in threat_keyword else 0)
            results.append(int(test_result["label"]))
            
      
    print(preds)
    print(results)
    accuracy = accuracy_score(results, preds)
    precision = precision_score(results, preds)
    recall = recall_score(results, preds)
    f1 = f1_score(results, preds)
    print("Relaxed Results:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")


#cert_evaluation_adapt()
#cert_evaluation_arbitration()
cert_evaluation_gan()
#cert_evaluation_misuse()
#cert_evaluation_gpt()