from simpletransformers.ner import NERModel
import pandas as pd
import numpy as np
import torch
from scipy.special import softmax
import os
import shutil
import argparse
from pathlib import Path
from os import listdir
from os.path import join
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True) #specifies the dataset for the experiments, e.g., data.tsv
parser.add_argument('--gpu', type=int, required=True) #allows to choose the specific GPU for experiments; depending on the hardware architecture, this can be removed

def get_classification_model(model):
    return NERModel(model[0], model[1], use_cuda=torch.cuda.is_available(), cuda_device=parser.parse_args().gpu)

def main(params, model_config, path_out):
    Path(path_out).mkdir(parents=True, exist_ok=True)

    probs = []
    preds = []
    
    train_df = pd.read_csv(parser.parse_args().input_dir, sep='\t', encoding ='utf-8')
#    test_df = pd.read_csv(join(parser.parse_args().input_dir, 'test_conll.tsv'), sep='\t', encoding ='utf-8')
#    test_df = test_df.loc[test_df['asp'] == 1]
#    to_predict = test_df.clean.tolist()
    
    sentence_id = np.array(list(set(train_df.sentence_id.tolist())))
    kf = KFold(n_splits=10)
    
    split = 1
    for train, test in kf.split(sentence_id):
        model = get_classification_model(model_config)
        to_predict = list()
        train_data_split = train_df[train_df.sentence_id.isin(train)]
        test_data_split = train_df[train_df.sentence_id.isin(test)]
        for i in sorted(list(set(test_data_split.sentence_id.tolist()))):
            question = list()
            for _, rows in test_data_split.iterrows():
                if rows.sentence_id == i:
                    question.append(rows.words)
            to_predict.append(' '.join(question))
    
        model.train_model(train_data_split, eval_df=train_data_split, args=args)       
        predictions, raw_outputs = model.predict(to_predict)
        
        c = 0
        ids = sorted(list(set(test_data_split.sentence_id.tolist())))
        examples = list()
        for n, (preds, outs) in enumerate(zip(predictions, raw_outputs)):
#        print("\n___________________________")
#        print("Sentence: ", to_predict[n])
            for pred, out in zip(preds, outs):
                key = list(pred.keys())[0]
                new_out = out[key]
                preds = list(softmax(np.mean(new_out, axis=0)))
                example = list()
                example.append(ids[c])
                example.append(key)
                example.append(pred[key])
                example.append(preds[np.argmax(preds)])
                example.append(preds)
                examples.append(example)
            c+=1
            
        df_out = pd.DataFrame(examples, columns=["sentence_id", "words", "labels", "prob", "raw_probabilities"])      
#            print(key, pred[key], preds[np.argmax(preds)], preds)
        df_out.to_csv(path_out + '/results_' + str(split) +  '.tsv', sep='\t', index=False)
        split+=1

# for the available transformer models refer to the simple transformers documentation
models =  [["roberta", "roberta-large"]]
#models = [["camembert", "camembert-base"], ["distilbert", "distilbert-base-uncased"], ["electra", "google/electra-large-discriminator"]]
#models = [["roberta", "roberta-large"], ["bert","bert-large-uncased"], ["camembert", "camembert-base"], ["distilbert", "distilbert-base-uncased"], ["electra", "google/electra-large-discriminator"]]
#models = [["electra", "google/electra-base-discriminator"], ["albert","albert-large-v2"], ["roberta", "roberta-large"], ["bert","bert-large-uncased"], ["xlnet", "xlnet-large-cased"], ["camembert", "camembert-base"], ["distilbert", "distilbert-base-uncased"]]

# configuration for the model training (for the particular model configuration refer to the paper)
# important to specify the "labels_list", e.g., if a dataset contains only two labels "O" and "OBJ": "labels_list": ["O", "OBJ"]

args = {"overwrite_output_dir": True, "num_train_epochs": 10, 
        "fp16": False, "train_batch_size": 8, 
        "gradient_accumulation_steps": 4, 
        "evaluate_during_training": False, 
        "learning_rate": 3e-5, 
        "labels_list": ["O", "OBJ", "ASP", "PRED"], 
        "reprocess_input_data": True, 
        "output_dir": "OUTPUT_PATH", 
        "max_seq_length": 64, 
        "use_early_stopping": True}

if __name__ == "__main__":
   for model in models:
       for lr in [3e-5]: #, 4e-5, 6e-5]: #1e-5, 3e-5, 5e-5]:
           args["learning_rate"] = lr
           main(parser.parse_args(), model, "directory-name-to-store-results/{}{}".format(model[0], '10epochs_CV'))
