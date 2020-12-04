import numpy as np
import torch
from tqdm import tqdm
def average_sentence_length(sentence_column):
    sents_tokens = sentence_column.split()
    sents_length = [len(s) for s in sents_tokens]
    MAX_SIZE = int(np.max(sents_length))
    return MAX_SIZE



def embedding_vectors_sent(tokeni,model,MAX_SIZE,BATCH_SIZE,train_X):
    tokenized_input = train_X.apply((lambda x: tokeni.encode(x, add_special_tokens=True)))

    padded_tokenized_input = np.zeros((0, MAX_SIZE))
    for i in tokenized_input.values:
        if len(i) < MAX_SIZE:
            padded_tokenized_input = np.append(padded_tokenized_input, [i + [0] * (MAX_SIZE - len(i))], axis=0)
        else:
            padded_tokenized_input = np.append(padded_tokenized_input, [i[:MAX_SIZE]], axis=0)

    # padded_tokenized_input = np.array([i + [0]*(MAX_SIZE-len(i)) if len(i)< MAX_SIZE else i for i in tokenized_input.values])
    attention_masks = np.where(padded_tokenized_input != 0, 1, 0)
    # print(attention_masks[0])

    input_ids = torch.tensor(padded_tokenized_input)
    attention_masks = torch.tensor(attention_masks)

    all_train_embedding = []
    step_size = BATCH_SIZE
    input_ids = torch.tensor(input_ids).to(torch.int64)
    with torch.no_grad():
        for i in tqdm(range(0, len(input_ids), step_size)):
            last_hidden_states = model(input_ids[i:min(i + step_size, len(train_X))],
                                       attention_mask=attention_masks[i:min(i + step_size, len(train_X))])[0][:, 0,
                                 :].numpy()
            all_train_embedding.append(last_hidden_states)

    unbatched_train = []
    for batch in all_train_embedding:
        for seq in batch:
            unbatched_train.append(seq)

    unbatched_train_np = np.array(unbatched_train)
    return input_ids,unbatched_train_np

def matrix_sentences(all_doc_ids,embedding_df,MAX_SENTENCE_NUM,embedding_dim):
    input_X = np.zeros((all_doc_ids.shape[0], MAX_SENTENCE_NUM, embedding_dim))
    i = 0
    for doc_id in all_doc_ids:
        all_sentences_embedding = embedding_df[embedding_df['doc_id']==doc_id]
        sent_nb = all_sentences_embedding.shape[0]
        input_X[i,:sent_nb] = all_sentences_embedding.drop(columns=['doc_id']).values
        i = i+1
    return input_X