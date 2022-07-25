import sys
import os, pickle
import shutil
import glob
import pretty_midi
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.stats import norm
import argparse

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kfold', type=int, default=0)
    parser.add_argument('--job_no', type=int, default=0)

    args = parser.parse_args()

    return args

def create_list(root):
    d = {}
    for fpth in os.listdir('npy/midi/mid'):
        id = fpth.split('.')[1]
        d[id] = os.path.join('npy/midi/mid',fpth)
    with open(root,'rb') as f:
        (train_list, val_list), test_list = pickle.load(f)
        train_list = [d[fpth.split('.')[1]] for fpth in train_list]
        val_list = [d[fpth.split('.')[1]] for fpth in val_list]
        test_list = [d[fpth.split('.')[1]] for fpth in test_list]

        # _, train_list = train_test_split(train_list, test_size = 0.01, random_state=42)
        # _, test_list = train_test_split(test_list, test_size = 0.01, random_state=42)
    return train_list, val_list, test_list

# Input: midi file path, save folder, window_size:a measure contains 8s, fs: 8 timesteps per sencond
# Output: save 2D feature matrix for that file, x axis denoting measure (), 
# y axis denoting features(velocity mean, IOI mean, IOI std)
def feature_mid(midifile, window_size=8, fs=8, N=2):
    features = []
    pm = pretty_midi.PrettyMIDI(midifile)
    end = pm.get_end_time()
    num_window = int(np.ceil(end / window_size))
    time_seq = np.arange(0, num_window*window_size, window_size)
    time_seq = np.append(time_seq, end)
    time_seq = np.around(time_seq, 3).tolist()
    times = np.arange(0, end, 1./fs)
    ts = window_size*fs
    
    if num_window <= 2*N:
        # with open("invalid_music.csv", 'w') as f:
        #     f.writelines(midifile+','+'too short,'+'\n')
        return
    else:
        for instrument in pm.instruments:
            j, v, n = 0, 0, 0
            ioi_mean, ioi_std, vel_mean = [], [], []
            pitch_mean, pitch_std, note_den = [], [], []
            onsets = instrument.get_onsets()
            piano_roll = instrument.get_piano_roll(fs=fs, times=times)
            pitch_line = []
            # get pitches at each timestep 
            for step in range(piano_roll.shape[1]):
                pitches = np.nonzero(piano_roll[:, step])[0].tolist()
                pitch_line.append(pitches)
            
            for i in range(N, num_window-N):
                ioi_stats, vel_stats, pitch_stats, duration = [], [], [], 0
                for p in range((i-N)*ts, (i+N)*ts):
                    if pitch_line[p]:
                        duration += 1
                note_den.append(np.around(duration/(ts*(2*N+1)), 6))

                while v<len(onsets) and onsets[v]<time_seq[i+N] and onsets[v]>=time_seq[i-N]:
                    vel = instrument.notes[v].velocity
                    vel_stats.append(vel)
                    pitch = instrument.notes[v].pitch
                    pitch_stats.append(pitch)
                    v+=1

                while j<len(onsets) and onsets[j]<time_seq[i+N] and onsets[j]>=time_seq[i-N]:
                    k = j+1
                    while k<len(onsets) and onsets[k]-onsets[k-1]<0.075 and onsets[k]<time_seq[i+N]:
                        k+=1
                    if k<len(onsets) and onsets[k]<time_seq[i+N]:
                        IOI = onsets[k]-onsets[j]
                        ioi_stats.append(IOI)
                    j = k

                if pitch_stats:
                    pitch_mean.append(np.around(np.mean(pitch_stats), 6))
                    pitch_std.append(np.around(np.std(pitch_stats), 6))
                else:
                    pitch_mean.append([])
                    pitch_std.append([])

                if vel_stats:
                    vel_mean.append(np.around(np.mean(vel_stats), 6))
                else:
                    vel_mean.append([])

                if ioi_stats:
                    ioi_mean.append(np.around(np.mean(ioi_stats), 6))
                    ioi_std.append(np.around(np.std(ioi_stats), 6))
                else:
                    ioi_mean.append([])
                    ioi_std.append([])


            fea_instru = list(zip(note_den, pitch_mean, pitch_std, vel_mean, ioi_mean, ioi_std))
            features.append(fea_instru)
#     print(fea_instru)
    
    # filename = save_dir+'mid_'+midifile.split('/')[-1][:-4]+'.npy'
    # np.save(filename, features)
    return features
# feature_mid('61401_01.mid')

# Input: features files
# Return: 12个mu, sigma (mu_i,k, sigma_i,k)
def compute_mu_sigma(root, fea_files):
    global train_list
    mlocs, mscales = [], []
    nlocs, nscales = [], []
    m_nd, m_pm, m_ps, m_vm, m_im, m_is = [],[],[],[],[],[]
    n_nd, n_pm, n_ps, n_vm, n_im, n_is = [],[],[],[],[],[]
    error = 0
    for i in tqdm(range(len(fea_files))):
        # file = fea_files[i]
        # features = np.load(root + file)
        features = fea_files[i]
        # features = np.array(features)
        try:
            if len(features)>0:
                label = int(train_list[i].split('.')[-2])
                m_feature = np.array(features[label])
                # print(m_feature)
                m_nd.extend(list(filter(lambda x: (x > 0), list(m_feature[:, 0]))))
                m_pm.extend(list(filter(lambda x: x, list(m_feature[:, 1]))))
                m_ps.extend(list(filter(lambda x: x, list(m_feature[:, 2]))))
                m_vm.extend(list(filter(lambda x: x, list(m_feature[:, 3]))))
                m_im.extend(list(filter(lambda x: x, list(m_feature[:, 4]))))
                m_is.extend(list(filter(lambda x: x, list(m_feature[:, 5]))))
                for instru in range(len(features)):
                    if instru != label:
                        features_instru = np.array(features[instru])
                        n_nd.extend(list(filter(lambda x: (x > 0), list(features_instru[:, 0]))))
                        n_pm.extend(list(filter(lambda x: x, list(features_instru[:, 1]))))
                        n_ps.extend(list(filter(lambda x: x, list(features_instru[:, 2]))))
                        n_vm.extend(list(filter(lambda x: x, list(features_instru[:, 3]))))
                        n_im.extend(list(filter(lambda x: x, list(features_instru[:, 4]))))
                        n_is.extend(list(filter(lambda x: x, list(features_instru[:, 5]))))
        except:
            error += 1
            print('[{}/{}] error'.format(error,len(fea_files)))
            continue
    mlocs.append(np.mean(m_nd))
    mlocs.append(np.mean(m_pm))
    mlocs.append(np.mean(m_ps))
    mlocs.append(np.mean(m_vm))
    mlocs.append(np.mean(m_im))
    mlocs.append(np.mean(m_is))
    mscales.append(np.std(m_nd))
    mscales.append(np.std(m_pm))
    mscales.append(np.std(m_ps))
    mscales.append(np.std(m_vm))
    mscales.append(np.std(m_im))
    mscales.append(np.std(m_is))
    
    nlocs.append(np.mean(n_nd))
    nlocs.append(np.mean(n_pm))
    nlocs.append(np.mean(n_ps))
    nlocs.append(np.mean(n_vm))
    nlocs.append(np.mean(n_im))
    nlocs.append(np.mean(n_is))
    nscales.append(np.std(n_nd))
    nscales.append(np.std(n_pm))
    nscales.append(np.std(n_ps))
    nscales.append(np.std(n_vm))
    nscales.append(np.std(n_im))
    nscales.append(np.std(n_is))

    return mlocs, mscales, nlocs, nscales
# input: each measure features：x(6个值的列表), num_channel
# output: Mscore for that measure
def compute_Mscore(x, mlocs, mscales, nlocs, nscales, num_channel):
    P_c0x = 1/num_channel
    P_c1x = 1-P_c0x
    Mscore = []
    for i in range(6):
        P_c0x *= norm.pdf(x[i], loc = mlocs[i], scale = mscales[i])
    for i in range(6):
        P_c1x *= norm.pdf(x[i], loc = nlocs[i], scale = nscales[i])
    P_mc = (P_c0x / (P_c0x + P_c1x))
    Mscore = np.around(np.log(P_mc), 6) 
    if Mscore:
        return Mscore
    else:
        return np.nan

def select_melody(root, fea_file, mlocs, mscales, nlocs, nscales, SP=36):
    # features = np.load(root+fea_file)
    features = fea_file
    num_c = len(features)
    num_m = len(features[0])
    A = np.full((num_m, num_c), np.nan)
    B = np.full((num_m, num_c), np.nan)
    #     forward
    for i in range(num_c):
        A[0, i] = compute_Mscore(features[i][0], mlocs, mscales, nlocs, nscales, num_c)
    for m in range(num_m):
        for c in range(num_c):
            x = A[m-1, c] + compute_Mscore(features[c][m], mlocs, mscales, nlocs, nscales, num_c)
            if np.isnan(x):
                B[m, c] = np.nan
                continue
            else:
                B[m, c] = c
            for i in range(num_c):
                y = A[m-1, i] + compute_Mscore(features[c][m], mlocs, mscales, nlocs, nscales, num_c) - SP
                if c!=i and y>x:
                    x = y
                    B[m, c] = i
            A[m, c] = x
     
    #     backtrack
    best_path = np.zeros((num_m))
    #排除全是nan的情况
    if np.isnan(A[-1]).all():
        best_path[-1] = np.nan
    else:
        best_path[-1] = np.nanargmax(A[-1])

    for m in range(num_m-2, -1, -1):
        if np.isnan(best_path[m+1]):
            if np.isnan(A[m]).all():
                best_path[m] = np.nan
            else:
                best_path[m] = np.nanargmax(A[m])
        else:
            idx = int(best_path[m+1])
            best_path[m] = B[m][idx]

    return best_path

    

config = vars(parse_arg())
print('kfold:',config['kfold'])
kfold_list = 'npy/kfolds/train_test_ids_kfolds_{}.pkl'.format(config['kfold'])
train_list, val_list, test_list = create_list(kfold_list)
feature_list = [feature_mid(fpath) for fpath in train_list]

mlocs, mscales, nlocs, nscales = compute_mu_sigma('',feature_list)
print('mlocs',mlocs)
print('mscales',mscales)
print('nlocs',nlocs)
print('nscales',nscales)

# rand_index = np.random.randint(len(fea_files), size = 20)
correct_by_measure, correct_by_song, num_m = 0, 0, 0
test_feature_list = [feature_mid(fpath) for fpath in test_list]
result_lists = {'correct':[],'wrong':[]}

# by measure
for i in tqdm(range(len(test_feature_list))):
    features = test_feature_list[i]
    try:
        results = select_melody('', features, mlocs, mscales, nlocs, nscales)
        n_nan = np.isnan(results).sum()
        num_m += len(results) - n_nan
    #     print(results)
        label = int(test_list[i].split('.')[-2])
    #     print(label)
        correct_by_measure += list(results).count(label)
        results_clean = [x for x in results if np.isnan(x) == False]
        if results_clean:
            maxlabel = max(results_clean, key=results_clean.count)
            if maxlabel == label:
                correct_by_song += 1
                result_lists['correct'].append(test_list[i])
            else:
                result_lists['wrong'].append(test_list[i])
    except:
        continue

acc_by_measure = correct_by_measure/num_m
acc_by_song = correct_by_song/len(test_list)
print('by measure',correct_by_measure, num_m, acc_by_measure)
print('by song',correct_by_song, len(test_list), acc_by_song)
with open('codes/case_analysis/dgm.pkl','wb') as f:
    pickle.dump(result_lists,f)