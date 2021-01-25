import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy.special import digamma
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import Model
import argparse
import sys
sys.path.append('./data_process/')
sys.path.append('./helpers')
import preprocess_data as prep_data
import results_summary as rs
import attent_model as mil
import e_step as e_step
import text_embed as text_embed
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="MARTA")
    parser.add_argument("--data_path",
                        type=str,
                        required=True,
                        help="data_path")
    parser.add_argument("--compute_embedding",
                        type=int,
                        required=True,
                        help="if first time running, set to True")
    parser.add_argument("--shuffle",
                        type=int,
                        required=True,
                        help="if first time running, set to True")
    parser.add_argument("--embedding_type",
                        type=str,
                        default='albert',
                        help="type of embedding to use")
    parser.add_argument("--epochs_mil",
                        type=int,
                        required=True,
                        help="epochs_mil")
    parser.add_argument("--epochs_mil_retrain",
                        type=int,
                        required=True,
                        help="epochs_mil_retrain")
    parser.add_argument("--evaluation_file",
                        type=str,
                        required=True,
                        help="evaluation result after each iteration")
    parser.add_argument("--evaluation_file_val",
                        type=str,
                        required=True,
                        help="evaluation result after each iteration")
    parser.add_argument("--weight_dir",
                        type=str,
                        required=True,
                        help="weights directory")
    parser.add_argument("--sup_degree",
                        type=float,
                        default=0.5,
                        help="sup_degree")
    parser.add_argument("--rdm",
                        type=int,
                        default=0,
                        help="rdm")
    parser.add_argument("--max_iter",
                        type=int,
                        default=20,
                        help="max_iter")
    parser.add_argument("--n_updates",
                        type=int,
                        default=5,
                        help="n_updates")
    args = parser.parse_args()
    return args

args = parse_args()
data_path= args.data_path
evaluation_file = args.evaluation_file
evaluation_file_val =args.evaluation_file_val
weights_dir = args.weight_dir
MAX_ITER = args.max_iter

workers_answers = pd.read_csv(data_path + 'workers_answers.csv')
worker_answer_matrix = workers_answers[['WorkerId', 'doc_id', 'worker_label']].sort_values(by=['WorkerId'])
all_workers = worker_answer_matrix.WorkerId.unique()
#initialize worker reliability
phi = pd.DataFrame(columns=['WorkerId', 'reliability_m','reliability_n'])
phi['WorkerId'] = all_workers
phi['reliability_m'] = 2.5
phi['reliability_n'] = 2


textual_data = pd.read_csv(data_path+'textual_data.csv')
textual_data_text = textual_data.drop_duplicates(subset='doc_id').reset_index(drop=True)[['doc_id','text','label']]
labels_wiki_doc_ids = textual_data_text[['doc_id','label']]


MAX_SIZE =text_embed.average_sentence_length(textual_data.sentence.str)
BATCH_SIZE = 50
embedding_dim = 768
MAX_SENTENCE_NUM =  textual_data['doc_id'].value_counts().max()
all_doc_ids = textual_data.doc_id.unique()
compute_embedding = args.compute_embedding
start_att = time.time()
if compute_embedding==1:
    if args.embedding_type=='scibert':
        from transformers import AutoTokenizer,AutoModel
        tokeni = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    else:
        from transformers import AlbertTokenizer, AlbertModel
        tokeni = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = AlbertModel.from_pretrained('albert-base-v2')
    input_ids,unbatched_train_np = text_embed.embedding_vectors_sent(tokeni,model,MAX_SIZE,BATCH_SIZE,textual_data['sentence'],)
    embedding_df = pd.DataFrame(unbatched_train_np)
    embedding_df['doc_id'] = textual_data.doc_id
    input_X = text_embed.matrix_sentences(all_doc_ids,embedding_df,MAX_SENTENCE_NUM,embedding_dim)
    with open(data_path + args.embedding_type +'embedding.npy', 'wb') as f:
        np.save(f, input_X)
else:
    with open(data_path +args.embedding_type + '_embedding.npy', 'rb') as f:
        input_X = np.load(f)


y=labels_wiki_doc_ids['label']

#split dataset
train_idxs,val_idxs,test_idxs = prep_data.split_data_sup(textual_data_text,args.sup_degree,args.rdm,args.shuffle)
X_train, X_val, X_test = input_X[train_idxs], input_X[val_idxs], input_X[test_idxs]
y_train, y_val, y_test = y[train_idxs], y[val_idxs], y[test_idxs]


train_val_input = np.append(X_train,X_val,axis=0)
index_predictions = np.append(y_train.index.values,y_val.index.values)


#initialize the prediction with an attention based model
model = mil.define_model(32,embedding_dim,MAX_SENTENCE_NUM,2)
history = model.fit(X_train, pd.get_dummies(y_train), batch_size=BATCH_SIZE, epochs=args.epochs_mil , validation_data=(X_val,pd.get_dummies(y_val)),callbacks=[EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)])
#predict
Y_pred_val = model.predict(X_val)
Y_pred = model.predict(X_test)

Y_pred_val_bin = np.where(Y_pred_val[:,1]>Y_pred_val[:,1].mean(),1,0)
y_pred_test_eval = np.where(Y_pred[:,1]>Y_pred[:,1].mean(),1,0)

rs.save_results(evaluation_file_val,y_val,Y_pred_val_bin)
rs.save_results(evaluation_file,y_test,y_pred_test_eval)

end_att  = time.time()


print("######elapsed time Attention#######", end_att -start_att )
with open(evaluation_file, 'a') as f:
    f.write('elapsed time Attention, %s\n' % (end_att -start_att))

#initialize z_i
predictions_label = pd.DataFrame(columns=['doc_id', 'pred_0', 'pred_1'], index=index_predictions)
predictions_label[['pred_0', 'pred_1']] = pd.get_dummies(y_train).rename(
    columns={0: 'pred_0', 1: 'pred_1'}).append(
    pd.DataFrame(Y_pred_val, columns=['pred_0', 'pred_1'],
                 index=y_val.index)).append(
    pd.DataFrame(Y_pred, columns=['pred_0', 'pred_1'], index=y_test.index))
predictions_label['doc_id'] = all_doc_ids[index_predictions]

#extract the attention scores
activation_1 = Model(inputs=model.input, outputs=model.get_layer("weights").output)
weights_sent_train = activation_1.predict(X_train)
weights_sent_val= activation_1.predict(X_val)
weights_sent_test = activation_1.predict(X_test)
#initialize the weight alpha_s
weight_sent_test = pd.DataFrame(weights_sent_test)
weight_sent_test['id'] = y_test.index.values

weight_sent_all = pd.DataFrame(np.append(weights_sent_train,weights_sent_val,axis=0))
weight_sent_all['id'] = index_predictions

workers_answers_sent = pd.read_csv(data_path+'workers_sentence_label.csv')
all_sentences_dict = dict(zip(workers_answers_sent.sentence.unique(),np.arange(workers_answers_sent.sentence.unique().shape[0])))
workers_answers_sent['sentence_id'] = workers_answers_sent['sentence'].map(all_sentences_dict)

doc_id_sentence_id = workers_answers_sent[['doc_id','sentence_id']].drop_duplicates(subset=['sentence_id'])
weights_doc_id_init = e_step.reconstruct_weights(weight_sent_all,doc_id_sentence_id,index_predictions,all_doc_ids)
weights_doc_id_init.to_csv(weights_dir+'/weight_init_0.csv')

weights_doc_id_test_init = e_step.reconstruct_weights(weight_sent_test,doc_id_sentence_id,y_test.index.values,all_doc_ids)
weights_doc_id_test_init.to_csv(weights_dir+'/weight_init_test_0.csv')

weight = pd.DataFrame(columns=['sentence_id', 'weight_s_0', 'weight_s_1'])
B_workers_answers_sent = workers_answers_sent[['WorkerId', 'sentence_id', 'sentence_label']]
B_workers_answers_sent.to_csv(data_path+'B_matrix.csv')

doc_id_sentence_id_train= doc_id_sentence_id[doc_id_sentence_id['doc_id'].isin(all_doc_ids[index_predictions])]
all_sentences = doc_id_sentence_id_train.sentence_id
worker_sent_article_label = workers_answers_sent[['WorkerId','doc_id','worker_label', 'sentence', 'sentence_label']].dropna(subset=['sentence'])
worker_sent_article_label['sentence_id'] = worker_sent_article_label['sentence'].map(all_sentences_dict)





model_retrain = mil.retrain_model_weights(32,embedding_dim,MAX_SENTENCE_NUM,2)

start_val = X_train.shape[0]
end_val = X_train.shape[0]+X_val.shape[0]


iter=0
start_marta = time.time()
while iter<MAX_ITER:
    change_phi = 1
    n_updates_rj = args.n_updates
    norm_diff_old_phi =1
    phi_old = phi.copy()
    start_e = time.time()
    while (change_phi>0.01) and (n_updates_rj>0):
        phi = e_step.worker_reliability(all_workers,phi,worker_sent_article_label,weights_doc_id_init,predictions_label,MAX_SENTENCE_NUM)
        n_updates_rj = n_updates_rj - 1
        change_phi, norm_diff_old_phi = e_step.update(phi, phi_old,norm_diff_old_phi)
        # print ("*******change phi******",change_phi)
        phi_old = phi.copy()

    change_zi=1
    norm_diff_old_zi = 1
    n_updates_zi = args.n_updates
    predictions_label_old = predictions_label.copy()

    while (change_zi > 0.01) and (n_updates_zi > 0):
        predictions_label = e_step.compute_zI(all_doc_ids[index_predictions], predictions_label, phi, worker_answer_matrix)
        n_updates_zi = n_updates_zi - 1
        change_zi, norm_diff_old_zi = e_step.update(predictions_label_old,predictions_label,norm_diff_old_zi)
        predictions_label_old = predictions_label.copy()

    pred_e_step_val = predictions_label.pred_1.values[start_val:end_val]
    y_pred_e_step_val = np.where(pred_e_step_val>pred_e_step_val.mean(),1,0)
    end_e = time.time()
    weights_sentences = e_step.compute_weight_i(all_sentences, weights_doc_id_init, phi, workers_answers_sent,MAX_SENTENCE_NUM)



    print("********E step**********")
    with open(evaluation_file_val, 'a') as f:
        f.write('E step val, %s\n' % iter)
    rs.save_results(evaluation_file_val, y_val, y_pred_e_step_val)
    with open(evaluation_file, 'a') as f:
        f.write('elapsed time MARTA E, %s\n' % (end_e - start_e))

    weights_e_steps =weight_sent_all.values[:,:-1]
    idx_doc_id=0
    for doc_id in predictions_label.doc_id.values:
        weights_doc_id = weights_sentences[weights_sentences['doc_id']==doc_id]
        weights_e_steps[idx_doc_id,:weights_doc_id.shape[0]] = weights_doc_id.weight.values
        idx_doc_id = idx_doc_id+1
    y_pred =pd.get_dummies(y_train).rename(
        columns={0: 'pred_0', 1: 'pred_1'}).append(
        pd.DataFrame(np.array([1-y_pred_e_step_val,y_pred_e_step_val]).T, columns=['pred_0', 'pred_1'], index=y_val.index))

    history = model_retrain.fit(train_val_input, {"pred": y_pred, "weights": weights_e_steps},
                                batch_size=BATCH_SIZE, epochs=args.epochs_mil_retrain,
                                validation_data=(
                                X_val, [pd.get_dummies(y_pred_e_step_val), weights_e_steps[-y_val.shape[0]:]]),
                                callbacks=[EarlyStopping(monitor='val_pred_loss', mode='min', verbose=1, patience=2)])

    # labels retrained
    Y_pred_val = model_retrain.predict(X_val)[0]
    print(Y_pred_val[0])
    Y_pred_retrain = model_retrain.predict(X_test)[0]
    y_pred_m_step = np.where(Y_pred_retrain[:, 1] > Y_pred_retrain[:, 1].mean(), 1, 0)
    print("********M step with retrain **********")
    with open(evaluation_file_val, 'a') as f:
        f.write('M step retrain, %s\n' % iter)
    Y_pred_val_bin = np.where(Y_pred_val[:, 1] > Y_pred_val[:, 1].mean(), 1, 0)
    rs.save_results(evaluation_file_val, y_val, Y_pred_val_bin)
    with open(evaluation_file, 'a') as f:
        f.write('M step retrain, %s\n' % iter)
    rs.save_results(evaluation_file, y_test, y_pred_m_step)

    Y_pred_val_bin = np.where(Y_pred_val[:, 1] > Y_pred_val[:, 1].mean(), 1, 0)
    predictions_label[['pred_0', 'pred_1']] = pd.get_dummies(y_train).rename(
        columns={0: 'pred_0', 1: 'pred_1'}).append(
        pd.DataFrame(Y_pred_val, columns=['pred_0', 'pred_1'], index=y_val.index)).append(
        pd.DataFrame(Y_pred_retrain, columns=['pred_0', 'pred_1'], index=y_test.index))

    # weights retrained

    weights_retrained = model_retrain.predict(train_val_input)[1]
    weights_retrained_df = pd.DataFrame(weights_retrained)
    weights_retrained_df['id'] = index_predictions
    weights_doc_id_init = e_step.reconstruct_weights(weights_retrained_df, doc_id_sentence_id, index_predictions, all_doc_ids)

    weights_test = model_retrain.predict(X_test)[1]
    weights_test_df = pd.DataFrame(weights_test)
    weights_test_df['id'] = y_test.index.values
    weights_doc_id_test = e_step.reconstruct_weights(weights_test_df, doc_id_sentence_id, y_test.index.values,
                                                     all_doc_ids)
    weights_doc_id_test.to_csv(weights_dir + '/weight_test_' + str(iter) + '.csv')
    iter = iter + 1

end_marta = time.time()
with open(evaluation_file_val, 'a') as f:
    f.write('M step retrain, %s\n' % iter)
rs.save_results(evaluation_file_val, y_val, Y_pred_val_bin)

with open(evaluation_file, 'a') as f:
    f.write('M step retrain, %s\n' % iter)
rs.save_results(evaluation_file, y_test, y_pred_m_step)

print("######elapsed time MARTA#######", end_marta -start_marta )
with open(evaluation_file, 'a') as f:
    f.write('elapsed time MARTA, %s\n' % (end_marta -start_marta))
with open(evaluation_file, 'a') as f:
    f.write('elapsed time MARTA E step, %s\n' % (end_e - start_e))
