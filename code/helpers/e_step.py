import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy.special import digamma
import pdb


def update(phi, phi_old,norm_diff_old):
    mean_old = phi_old.iloc[:, 1] / (phi_old.iloc[:, 1] + phi_old.iloc[:, 2])
    mean = phi.iloc[:, 1] / (phi.iloc[:, 1] + phi.iloc[:, 2])
    norm_diff = LA.norm(mean_old-mean)
    change = np.abs(norm_diff-norm_diff_old)
    return change,norm_diff

def compute_zI(all_doc_ids,predictions_label,phi,worker_answer_matrix):
    for doc_id in all_doc_ids:
        p_zi_1 = predictions_label[predictions_label['doc_id'] == doc_id].pred_1.iloc[0]
        p_zi_0 = predictions_label[predictions_label['doc_id'] == doc_id].pred_0.iloc[0]
        index_doc_id = predictions_label[predictions_label['doc_id'] == doc_id].index[0]
        worker_labeling_doc_id = worker_answer_matrix[worker_answer_matrix['doc_id'] == doc_id]
        worker_pos_doc_id = worker_labeling_doc_id[worker_labeling_doc_id['worker_label'] == 1]
        phi_pos = phi[phi['WorkerId'].isin(worker_pos_doc_id.WorkerId.unique())]
        p_zi_1 = p_zi_1 * np.exp(phi_pos.reliability_m.apply(digamma).sum() - (phi_pos.reliability_m + phi_pos.reliability_n).apply(
            digamma).sum())
        p_zi_0 = p_zi_0 * np.exp(phi_pos.reliability_n.apply(digamma).sum() - (phi_pos.reliability_m + phi_pos.reliability_n).apply(
            digamma).sum())
        worker_neg_doc_id = worker_labeling_doc_id[worker_labeling_doc_id['worker_label'] == 0]
        phi_neg = phi[phi['WorkerId'].isin(worker_neg_doc_id.WorkerId.unique())]
        p_zi_1 = p_zi_1 * np.exp(phi_neg.reliability_n.apply(digamma).sum() - (phi_neg.reliability_m + phi_neg.reliability_n).apply(
            digamma).sum())
        p_zi_0 = p_zi_0 * np.exp(phi_neg.reliability_m.apply(digamma).sum() - (phi_neg.reliability_m + phi_neg.reliability_n).apply(
            digamma).sum())
        predictions_label.loc[index_doc_id, ['pred_1']] = p_zi_1 * 1.0 / (p_zi_0 + p_zi_1)
        predictions_label.loc[index_doc_id, ['pred_0']] = p_zi_0 * 1.0 / (p_zi_0 + p_zi_1)
    return predictions_label


def compute_weight_i(all_sentences, weights_doc_id_init, phi, workers_answers_sent,MAX_SENTENCE_NUM):
    for index,sent in all_sentences.iteritems():
        weight_init = weights_doc_id_init[weights_doc_id_init['sentence_id'] == sent].weight.iloc[0]
        index_sent = weights_doc_id_init[weights_doc_id_init['sentence_id'] == sent].index[0]

        worker_labeling_sent = workers_answers_sent[workers_answers_sent['sentence_id'] == sent]
        # positively labeled sentences
        worker_pos_sent = worker_labeling_sent[worker_labeling_sent['sentence_label'] == 1]
        phi_pos = phi[phi['WorkerId'].isin(worker_pos_sent.WorkerId.unique())]
        weight_pos = weight_init * np.exp(phi_pos.reliability_m.apply(digamma).sum() - (phi_pos.reliability_m + phi_pos.reliability_n).apply(
            digamma).sum())
        # negatively labeled sentences
        worker_neg_sent = worker_labeling_sent[worker_labeling_sent['sentence_label'] == 0]
        phi_neg = phi[phi['WorkerId'].isin(worker_neg_sent.WorkerId.unique())]
        weight_neg = weight_init * np.exp(phi_neg.reliability_n.apply(digamma).sum() - (phi_neg.reliability_m + phi_neg.reliability_n).apply(
            digamma).sum())
        weights_doc_id_init.loc[index_sent, ['weight']] = weight_pos + weight_neg
    for index, sent in all_sentences.iteritems():
        weight_sent = weights_doc_id_init[weights_doc_id_init['sentence_id'] == sent].weight.iloc[0]
        doc_id_sent = weights_doc_id_init[weights_doc_id_init['sentence_id'] == sent].doc_id.iloc[0]
        other_weights = weights_doc_id_init[weights_doc_id_init['doc_id'] == doc_id_sent].weight
        index_sent = weights_doc_id_init[weights_doc_id_init['sentence_id'] == sent].index[0]
        weights_doc_id_init.loc[index_sent, ['weight']] = np.exp(weight_sent) / np.sum(np.exp(other_weights))
    return weights_doc_id_init

def worker_reliability(all_workers,phi,worker_sent_article_label,weights_sentences,predictions_label,MAX_SENTENCE_NUM):
    for worker in all_workers:
        phi_m_w = phi[phi['WorkerId'] == worker].reliability_m.iloc[0]
        phi_n_w = phi[phi['WorkerId'] == worker].reliability_n.iloc[0]
        index_w = phi[phi['WorkerId'] == worker].index[0]

        answers_w = worker_sent_article_label[worker_sent_article_label['WorkerId']==worker]
        answers_w_A_pos = answers_w[answers_w['worker_label']==1]
        doc_ids_pos_A = answers_w_A_pos['doc_id'].unique()


        theta_I_doc_ids_w_pos = predictions_label[predictions_label['doc_id'].isin(doc_ids_pos_A)].pred_1.sum()
        theta_I_0_doc_ids_w_pos = predictions_label[predictions_label['doc_id'].isin(doc_ids_pos_A)].pred_0.sum()


        answers_w_B_pos_A_pos = answers_w_A_pos[answers_w_A_pos['sentence_label'] == 1]
        sentences_pos_B_pos_A = answers_w_B_pos_A_pos['sentence_id'].unique()

        weight_i_sent_w_B_pos_A_pos = weights_sentences[weights_sentences['sentence_id'].isin(sentences_pos_B_pos_A)].weight.sum()
        weight_i_0_sent_w_B_pos_A_pos = (((1 / MAX_SENTENCE_NUM) * 2) - weights_sentences[weights_sentences['sentence_id'].isin(sentences_pos_B_pos_A)].weight).sum()

        phi_m_w = phi_m_w + theta_I_doc_ids_w_pos + weight_i_sent_w_B_pos_A_pos
        phi_n_w = phi_n_w + theta_I_0_doc_ids_w_pos + weight_i_0_sent_w_B_pos_A_pos

        answers_w_B_neg_A_pos = answers_w_A_pos[answers_w_A_pos['sentence_label'] == 0]
        sentences_neg_B_pos_A = answers_w_B_neg_A_pos['sentence_id'].unique()

        weight_i_sent_w_B_neg_A_pos = weights_sentences[weights_sentences['sentence_id'].isin(sentences_neg_B_pos_A)].weight.sum()
        weight_i_0_sent_w_B_neg_A_pos  = (((1 / MAX_SENTENCE_NUM) * 2) - weights_sentences[weights_sentences['sentence_id'].isin(sentences_pos_B_pos_A)].weight).sum()

        phi_m_w = phi_m_w + weight_i_0_sent_w_B_neg_A_pos
        phi_n_w = phi_n_w + weight_i_sent_w_B_neg_A_pos


        answers_w_A_neg = answers_w[answers_w['worker_label']==0]
        doc_ids_neg_A = answers_w_A_neg['doc_id'].unique()

        theta_I_doc_ids_w_neg = predictions_label[predictions_label['doc_id'].isin(doc_ids_neg_A)].pred_1.sum()
        theta_I_0_doc_ids_w_neg = predictions_label[predictions_label['doc_id'].isin(doc_ids_neg_A)].pred_0.sum()

        answers_w_B_pos_A_neg = answers_w_A_neg[answers_w_A_neg['sentence_label'] == 1]
        sentences_pos_B_A_neg = answers_w_B_pos_A_neg['sentence_id'].unique()

        weight_i_sent_w_B_pos_A_neg = weights_sentences[weights_sentences['sentence_id'].isin(sentences_pos_B_A_neg)].weight.sum()
        weight_i_0_sent_w_B_pos_A_neg = (((1 / MAX_SENTENCE_NUM) * 2) - weights_sentences[weights_sentences['sentence_id'].isin(sentences_pos_B_A_neg)].weight).sum()

        phi_m_w = phi_m_w + theta_I_0_doc_ids_w_neg + weight_i_0_sent_w_B_pos_A_neg
        phi_n_w = phi_n_w + theta_I_doc_ids_w_neg + weight_i_sent_w_B_pos_A_neg

        answers_w_B_neg_A_neg = answers_w_A_neg[answers_w_A_neg['sentence_label'] == 0]
        sentences_neg_B_A_neg = answers_w_B_neg_A_neg['sentence_id'].unique()

        weight_i_sent_w_neg_B_A_neg = weights_sentences[weights_sentences['sentence_id'].isin(sentences_neg_B_A_neg)].weight.sum()
        weight_i_0_sent_w_neg_B_A_neg = (((1 / MAX_SENTENCE_NUM) * 2) - weights_sentences[weights_sentences['sentence_id'].isin(sentences_neg_B_A_neg)].weight).sum()

        phi_m_w = phi_m_w + weight_i_0_sent_w_neg_B_A_neg
        phi_n_w = phi_n_w + weight_i_sent_w_neg_B_A_neg

        phi.loc[index_w, ['reliability_m']] = phi_m_w
        phi.loc[index_w, ['reliability_n']] = phi_n_w
    return phi


def reconstruct_weights(weight_sent_all,doc_id_sentence_id,index_predictions,all_doc_ids):
    weights_doc_id_init = pd.DataFrame(columns=['doc_id','sentence_id','weight'])
    for idx in index_predictions:
        doc_id = all_doc_ids[idx]
        sentences_ids = doc_id_sentence_id[doc_id_sentence_id['doc_id'] == doc_id].sentence_id.values
        nbr_sentences = sentences_ids.shape[0]
        weights = weight_sent_all[weight_sent_all['id']==idx].values
        sentences_weights = weights[0,:nbr_sentences]
        idx_sent_weight = pd.DataFrame(columns=['doc_id','sentence_id','weight'])
        idx_sent_weight['sentence_id']=sentences_ids
        idx_sent_weight['weight']=sentences_weights
        idx_sent_weight['doc_id'] = doc_id
        weights_doc_id_init = weights_doc_id_init.append(idx_sent_weight)

    weights_doc_id_init = weights_doc_id_init.reset_index(drop=True)
    return weights_doc_id_init