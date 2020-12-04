import pandas as pd
import numpy as np
import nltk
import sys
sys.path.append('../../src/process_data_res')
import preprocess_data as prep_data
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="preprocess data")
    parser.add_argument("--original_data",
                        type=str,
                        required=True,
                        help="original_data")
    parser.add_argument("--dir_gen_marta",
                        type=str,
                        required=True,
                        help="Directory of generated data")

    args = parser.parse_args()
    return args

args = parse_args()



original_data = pd.read_csv(args.original_data).dropna(subset=['text']).dropna(subset=['rationale']).dropna(subset=['label']).reset_index(drop=True)

original_data_textual = original_data.drop_duplicates(subset=['doc_id']).reset_index(drop=True)
#generate textual data
textual_data = original_data_textual[['doc_id','text','label']]

longform = pd.DataFrame(columns=['doc_id', 'text','sentence','label'])

for idx, doc_id, text , label, in textual_data.itertuples():
    sentences_summary = nltk.tokenize.sent_tokenize(text)

    longform = longform.append(
        [{'doc_id': doc_id, 'text': text,'sentence':sent,'label':label} for sent in sentences_summary],
        ignore_index=True
    )
longform['sentence_id'] = np.arange(longform.shape[0])
longform[['doc_id', 'text', 'sentence', 'label', 'sentence_id']].to_csv(args.dir_gen_marta+'textual_data.csv',index=False)


#generate worker_answer matrix
worker_answer_link =  original_data[['doc_id','WorkerId','worker_label']]
worker_answer_link = worker_answer_link[worker_answer_link['worker_label']!='maybe']
worker_answer_link['worker_label'] = worker_answer_link['worker_label'].astype(int)
worker_answer_link.to_csv(args.dir_gen_marta+'workers_answers.csv',index=False)


#generate worker_sentence matrix
data_rationale = original_data[['doc_id','WorkerId','worker_label','text','rationale']]
worker_sentence_label = pd.DataFrame(columns=['doc_id','WorkerId','worker_label','sentence','rationale','sent_label','rat_label'])

for idx, doc_id, WorkerId,worker_label, text, rationale  in data_rationale.itertuples():
    sentences_summary = nltk.tokenize.sent_tokenize(text)
    # sentences_rationale = nltk.tokenize.sent_tokenize(rationale)
    worker_sentence_label = worker_sentence_label.append(
        [{'doc_id': doc_id,'WorkerId': WorkerId ,'worker_label':worker_label,'sentence':sent,'rationale':rationale,'sent_label': prep_data.clean_text(sent).find(prep_data.clean_text(rationale)), 'rat_label': prep_data.clean_text(rationale).find(prep_data.clean_text(sent))} for sent in sentences_summary],
        ignore_index=True
    )

worker_sentence_label['sent_label'] = np.where(worker_sentence_label['sent_label']!= -1, 1, 0)
worker_sentence_label['rat_label'] = np.where(worker_sentence_label['rat_label']!= -1, 1, 0)
worker_sentence_label['sentence_label'] = worker_sentence_label['sent_label']  + worker_sentence_label['rat_label']
worker_sentence_label['sentence_label'] = np.where(worker_sentence_label['sentence_label']!= 0, 1, 0)
worker_sentence_label.drop_duplicates(subset=['rationale'])
worker_sentence_label[['doc_id','WorkerId','worker_label','sentence','rationale','sentence_label']].to_csv(args.dir_gen_marta+'workers_sentence_label.csv',index=False)
