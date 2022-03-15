from simpletransformers.classification import ClassificationModel
import pandas as pd
import numpy as np
import torch
from scipy.special import softmax
import os
import argparse
import shutil
from sklearn.metrics import precision_recall_curve
from pathlib import Path
from os import listdir
from os.path import join
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True) # specifies the path to the data directory
parser.add_argument('--label', type=str, required=True) # specifies the target classification label, e.g., comp
parser.add_argument('--gpu', type=int, required=True) # specifies the GPU number (depending on the hardware architecture, the line can be removed)

def get_classification_model(model):
    return ClassificationModel(model[0], model[1], use_cuda=torch.cuda.is_available(), cuda_device=parser.parse_args().gpu) # num_labels=2

def main(params, model_config, output_predictions, path_out):
    Path(path_out).mkdir(parents=True, exist_ok=True)

    train_probs = []
    train_preds = []
    ground_truth = []
    ground_questions = []
    
    FILENAME = '' # specifies the file name with the very hard comparative questions
    df = pd.read_csv(join(parser.parse_args().input_dir, FILENAME), sep='\t')
    #df = df[['clean', label]]
    #df.columns = ['text', 'labels']
    train_questions = np.array(df.clean.tolist())
    y_train = np.array(df[parser.parse_args().label].tolist())
    
    kf = StratifiedKFold(n_splits=10)
    for train_index, test_index in kf.split(train_questions, y_train):
        quest_train, quest_test = train_questions[train_index], train_questions[test_index]
        y_tr, y_ts = y_train[train_index], y_train[test_index]
        ground_truth.extend(y_ts)
        ground_questions.extend(quest_test)
        
        #print(len(train_questions))
        
        train_df = pd.DataFrame({'text': quest_train, 'labels': y_tr})
        #test_df = pd.DataFrame({'text': quest_test, 'labels': y_ts})
        #print(len(train_df))
        tmp_model = get_classification_model(model_config)
        tmp_model.train_model(train_df, eval_df=train_df, args=args)
        #result, raw_outputs, _ = tmp_model.eval_model(eval_df)
        
        predictions, raw_outputs = tmp_model.predict(quest_test)
        train_preds.extend(predictions)
        
        #print(predictions)
        train_probs.extend(softmax(raw_outputs, axis=1))
        
        #train_preds.extend(np.argmax(raw_outputs, axis=1))
        
        #print(model[0], str(split))
        #shutil.rmtree("outputs_aspect_classification_direct_CV")
        #shutil.rmtree("runs")
        #shutil.rmtree("cache_dir")
        #print("Directory with models is deleted")

    neg_prob = [p[0] for p in train_probs]
    pos_prob = [p[1] for p in train_probs]
    df_out = pd.DataFrame({'clean': ground_questions,'neg_prob': neg_prob, 'pos_prob': pos_prob, 'predictions': train_preds, 'true_label': ground_truth})
    df_out.to_csv(path_out + '/results.tsv', sep='\t', index=False)
    
# for the available transformer models refer to the simple transformers documentation
#models = [["roberta", "roberta-large"]]
#models = [["roberta", "roberta-large"] ,["bert","bert-large-uncased"], ["albert","albert-large-v2"]]
#models = [["albert","albert-large-v2"], ["bert","bert-large-uncased"]]
models = [["electra", "google/electra-large-discriminator"], ["roberta", "roberta-large"], ["xlnet", "xlnet-large-cased"] , ["bert","bert-large-uncased"], ["albert","albert-large-v2"]]
#models = [["electra", "google/electra-base-discriminator"], ["roberta", "roberta-base"], ["xlnet", "xlnet-base-cased"] , ["bert","bert-base-uncased"], ["albert","albert-base-v2"]]

# configuration for the model training (for the particular model configuration refer to the paper)
args = {"overwrite_output_dir": True, "num_train_epochs": 10, "fp16": False, "train_batch_size": 8, "gradient_accumulation_steps": 4, "evaluate_during_training": True, "max_seq_length": 64, "learning_rate": 2e-5, "output_dir": "OUTPUT_PATH", "early_stopping_consider_epochs": True}
#use_weights = True

if __name__ == "__main__":
   for model in models:
       for lr in [2e-5]: #, 4e-5, 6e-5]: #1e-5, 3e-5, 5e-5]:
           args["learning_rate"] = lr
           main(parser.parse_args(), model, False, "directory-name-to-store-results/{}{}".format(model[1], '_10epochs'))
