import glob, os, pickle
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def calc_ent(x):
    """
        calculate shanno ent of x
    """

    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent

def melody_id(pitch_roll):
    score1 = [0] * (len(pitch_roll))
    score2 = [0] * (len(pitch_roll))
    for i in range(pitch_roll.shape[0]):
        notes = []
        for j in range(pitch_roll.shape[1]):
            if pitch_roll[i][j] != -2:
                if pitch_roll[i][j] == -1:
                    notes.append(note)
                else:
                    note = pitch_roll[i][j]
                    notes.append(note)
        if notes:
            score1[i] = np.mean(notes)
        else:
            score1[i] = -2

        score2[i] = calc_ent(np.array(notes))

    result1 = np.argmax(score1)
    result2 = np.argmax(score2)
    return result1, result2

def create_list(root):
    with open(root,'rb') as f:
        _,test_list = pickle.load(f)

        # _, train_list = train_test_split(train_list, test_size = 0.01, random_state=SEED)
        # _, val_list = train_test_split(val_list, test_size = 0.01, random_state=SEED)
    return test_list

test_files = create_list('npy/kfolds/train_test_ids_kfolds_5.pkl')
# train, test = train_test_split(files, test_size=0.2, random_state=42)
# np.save('testnpy.npy', test)
# # out = np.load('E:\\1\AI_lyricists\data\\npy\original\o.3920.c.0.npy')
# out = np.load('E:\\1\AI_lyricists\data\\npy\swap\s.5834.c.1.npy')
# data_dict = out[()]
# pitch_roll = data_dict['data']
# print(data_dict['data'])

correct1, correct2 = 0, 0
with open('npy/summary/shuffle_records_rev.pkl','rb') as f:
    rev = pickle.load(f)

for i, file in enumerate(test_files):
    fpath = rev[file]
    label = np.load('npy/original/{}'.format(fpath),allow_pickle=True).reshape(1)[0]['label']
    pitch_roll = np.load('npy/original/{}'.format(fpath),allow_pickle=True).reshape(1)[0]['data']
    result1, result2 = melody_id(pitch_roll)
    if result1 == label:
        correct1 += 1
    if result2 == label:
        correct2 += 1

accuracy1 = correct1 / len(test_files)
accuracy2 = correct2 / len(test_files)
print("highest average pitch accuracy", correct1, accuracy1)
print("highest entropy accuracy", correct2, accuracy2)