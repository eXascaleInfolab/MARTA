import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, StratifiedKFold

def split_data_sup(dset,sup,rdm,shuffle):
    # Split dataset
    if shuffle==0:
        skf = StratifiedKFold(n_splits=2)
        input_X = dset['text']
        y=dset['label']
        for train_index, test_index in skf.split(input_X, y):
            x_train, X_val_test = input_X[train_index], input_X[test_index]
            y_train, y_val_test = y[train_index], y[test_index]
        x_val,x_test,y_val,y_test=train_test_split(X_val_test,y_val_test,stratify=y_val_test,test_size=0.5,random_state=rdm)
    else:
        x_train,x_val_test,y_train,y_val_test = train_test_split(
            dset['text'],
            dset['label'],
            stratify=dset['label'],
            test_size=1-sup,
            shuffle=True, random_state=rdm)

        # Further split into val and test
        x_val, x_test, y_val, y_test = train_test_split(
            x_val_test,
            y_val_test,
            stratify=y_val_test,
            test_size=0.5,
            shuffle=True, random_state=11)
    return x_train.index,  x_val.index, x_test.index


def split_data(dset):
    # Split dataset
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=11) # xi
    split_gen = skf.split(dset['text'], dset['label'])

    # Split the train and val+test indices
    for train_idxs, val_test_idxs in split_gen: pass # we only use the last

    # Further split into val and test
    val_idxs, test_idxs, _, _ = train_test_split(
        val_test_idxs,
        dset['label'].iloc[val_test_idxs],
        stratify=dset['label'].iloc[val_test_idxs],
        test_size=0.5,
        shuffle=True, random_state=11) # xi
    return train_idxs,val_idxs,test_idxs

def clean_text(text):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    # lower text
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    text = text.lower()
    # stemmer = SnowballStemmer("english")
    text = BAD_SYMBOLS_RE.sub(' ',text)
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = text.replace('\n', ' ')
    # text = text.replace('\\n',' ')
    text = text.replace('\\',' ')
    # tokenize text
    text = text.split(" ")
    # remove stop words
    # stop = stopwords.words('english')
    # text = [x for x in text if x not in stop]
    # remove words with only one letter or empty
    text = [t for t in text if len(t) > 1]
    # stems = []
    # for t in text:
    #     stems.append(stemmer.stem(t))
    # join all
    text = " ".join(text)
    return (text)



