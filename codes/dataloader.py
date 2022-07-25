from os import listdir
from torch.utils.data import Dataset
import numpy as np
import os, pickle
# import pretty_midi
# from midi2pianoroll import get_piano_rolls
# from pretty_midi import PrettyMIDI, TimeSignature, KeySignature


class MIDIDataset(Dataset):
    def __init__(self, song_dirs=None, frame_shuffle=True, min_len=705, max_len=9000,
     model='cnn',mode='shuffled',root='npy/shuffle'):
        self.song_dirs = song_dirs
        self.frame_shuffle = frame_shuffle #used only when model is plain cnn
        self.model = model
        self.mode = mode
        self.min_len = min_len
        self.max_len = max_len
        self.root = root
        if self.mode == 'test':
            with open('npy/summary/shuffle_records_rev.pkl','rb') as f:
                self.rev = pickle.load(f)
        self.songs = self.load_songs_from_dir()
        self.x, self.y, self.mapping, self.lens, self.channel_masks = self.load_frames_from_songs()
           
    def load_frames_from_songs(self):
        # CNN: (\sum_N {length_i}, 16, 16), (\sum_N {length_i}, ),  (\sum_N {length_i}, 16)
        # CRNN: (N, length_i, 16, 16), (N,), (N, length_i, 16)
        frames_all, labels_all, mapping_all, lengths_all, channel_mask_all = [], [], [], [], [] 
        if self.model in ['cnn','resnet']:
            for song, label, channel_mask in self.songs:
                frames = frame_func(song)   
                frames_tmp, labels_tmp, mapping_tmp, lens_tmp = [], [], [], []
                for f in frames:
                    if np.all(f==0):
                        continue
                    indices = [x for x in range(16)]
                    if self.frame_shuffle:
                        np.random.shuffle(indices)
                        label = indices.index(label)
                        channel_mask[[x for x in range(16)]] = channel_mask[indices]
                    frames_tmp.append(f[indices])
                    labels_tmp.append(label)
                    mapping_tmp.append(np.array(indices))
                
                frames_all.append(np.stack(frames_tmp))
                labels_all.append(np.array(labels_tmp)) 
                mapping_all.append(np.stack(mapping_tmp))
                channel_mask_all.append(channel_mask)

            data = np.expand_dims(np.concatenate(frames_all,axis=0), 1) / 129   
            mapping = np.concatenate(mapping_all,axis=0)
            labels = np.concatenate(labels_all,axis=0)
            lens = np.zeros_like(labels)
            channel_masks = np.zeros_like(labels) #channel_mask_all

        elif self.model in ['crnn']:
            for song, label, channel_mask in self.songs:
                length = song.shape[1]
                frames = frame_func(song)
                if song.shape[1] > self.max_len:
                    idx = [i for i,f in enumerate(frames) if np.all(f==0)]
                    nonzeros = [i for i in range(len(frames)) if i not in idx]
                    skip = self.max_len / (length - len(idx) - self.max_len)
                    idx.extend([nonzeros[int(round(i))] for i in np.arange(0, len(nonzeros),skip+1)])
                    frames = np.array([f for i,f in enumerate(frames) if i not in idx])

                frames_all.append(np.expand_dims(frames,1)/129)
                labels_all.append(label)
                lengths_all.append(frames.shape[0])
                channel_mask_all.append(channel_mask)
            data, labels, mapping, lens, channel_masks = frames_all, labels_all, labels_all, lengths_all, channel_mask_all

        elif self.model in ['bigbird']:
            for song, label, channel_mask in self.songs:
                length = song.shape[1]
                if song.shape[1] < self.min_len:
                    song = np.pad(song,((0,0), (0,self.min_len-song.shape[1])), constant_values = -2)
                frames = frame_func(song)
                if song.shape[1] > self.max_len:
                    idx = [i for i,f in enumerate(frames) if np.all(f==0)]
                    nonzeros = [i for i in range(len(frames)) if i not in idx]
                    skip = self.max_len / (length - len(idx) - self.max_len)
                    idx.extend([nonzeros[int(round(i))] for i in np.arange(0, len(nonzeros),skip+1)])
                    frames = np.array([f for i,f in enumerate(frames) if i not in idx])
                    length = frames.shape[0]

                frames_all.append(np.expand_dims(frames,1)/129)
                labels_all.append(label)
                lengths_all.append(length)
                channel_mask_all.append(channel_mask)
            data, labels, mapping, lens, channel_masks = frames_all, labels_all, labels_all, lengths_all, channel_mask_all

        elif self.model in ['rnn']:
            
            for song, label, channel_mask in self.songs:
                channel, timestep = song.shape
                song[np.where(song == -2)] = 129
                song[np.where(song == -1)] = 128
            
                song = song + 1
                song[np.where(song == 130)] = 0
                song = song.T
                song = song[:min(1024,song.shape[0])]
                frames_all.append(song/129)
                channel_mask_all.append(channel_mask)
                labels_all.append(label)
                lengths_all.append(timestep)
            
            data, labels, mapping, lens, channel_masks = frames_all, labels_all, labels_all, lengths_all, channel_mask_all
        
        return data, labels, mapping, lens, channel_masks


            


    def load_songs_from_dir(self):
        data = []
        for fpath in self.song_dirs:
            if self.mode == 'unshuffled':
                fpath = self.rev[fpath]
                m = np.load('npy/original/{}'.format(fpath),allow_pickle=True).reshape(1)[0]['data']
            else:
                m = np.load(os.path.join(self.root,fpath))
            
            if len(fpath.split('.')) == 5:
                _,_,_,label,_ = fpath.split('.') 
            else:
                label,_ = fpath.split('.') # infer
            channel_mask = np.zeros(16)
            channel_mask[np.where(np.sum(m,axis=1)!=129*m.shape[1])] = 1
            label = int(label)
            data.append([m,label,channel_mask])
        return data

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx], self.mapping[idx], self.lens[idx], self.channel_masks[idx]

def frame_func(mid, window_size=16):
    '''
    Generate frames of (channel, window_size) from a standardized channel-time matrix 

    Args:
        mid: channel-time matrix, where each item M(i,j) is a pitch value at i-th time in j-th channel.
            Currently the resolution of these matrices is 16-th note, that is, each item represents 
            a 16-th note (1/4 beat).
        window_size: the length of each frame. Default value is 16, i.e., 2 beats
    '''
    channel, timestep = mid.shape
    mid[np.where(mid == -2)] = 129
    mid[np.where(mid == -1)] = 128
   
    mid = mid + 1
    mid[np.where(mid == 130)] = 0

    mid_pad = np.pad(mid,((0,16-channel), (7,8)), constant_values = 0)

    row_stride, col_stride = mid_pad.strides
    frame_shape = (16, timestep, 1, window_size)
    frame_strides = (row_stride, col_stride, row_stride, col_stride)
            
    frames = np.lib.stride_tricks.as_strided(mid_pad,shape=frame_shape,strides=frame_strides)
    frames = np.squeeze(np.moveaxis(frames,0,1))

    return frames


# def get_pitch_rolls(midifile, transfer=False, beat_resolution=4):
#     label = int(midifile.split('.')[-2])
#     pm = pretty_midi.PrettyMIDI(midifile)
#     if transfer:
#         pm =  transfer_key(pm)
#     piano_rolls, onset_rolls, info = get_piano_rolls(pm, beat_resolution=beat_resolution)
#     num_channel = len(piano_rolls)
#     num_time = piano_rolls[0].shape[0]
#     pitch_matrix = np.full((num_channel, num_time), fill_value=-2)
#     for i, instrument in enumerate(piano_rolls):
#         for t in range(num_time):
#             pitch = np.nonzero(instrument[t, :])[0]
#             #print(pitch)
#             if pitch.size > 0:
#                 pitch_matrix[i, t] = max(max(pitch), pitch_matrix[i, t])

#         for t in range(num_time):
#             if pitch_matrix[i, t] != -2 and not onset_rolls[i][t]:
#                 pitch_matrix[i, t] = -1

#     out_dict = {'data': pitch_matrix, 'label': label}

#     return out_dict

# def get_pitch_class_histogram(notes, use_duration=True, use_velocity=True, normalize=True):
#     weights = np.ones(len(notes))
#     # Assumes that duration and velocity have equal weight
#     if use_duration:
#         weights *= [note.end - note.start for note in notes]
#     if use_velocity:
#         weights *= [note.velocity for note in notes]

#     histogram, _ = np.histogram([n.pitch % 12 for n in notes],
#                                 bins=np.arange(13),
#                                 weights=weights,
#                                 density=normalize)
#     if normalize:
#         histogram /= (histogram.sum() + (histogram.sum() == 0))
#     return histogram

# def time_group(start, time_list):
#     for time in time_list[::-1]:
#         if start - time >= 0:
#             return time_list.index(time)

# def transfer_key(midi_data):
#     # key_signature_changes
#     key_profile = pickle.load(open('npy/summary/key_profile.pickle', 'rb'))

#     key_signature_changes = midi_data.key_signature_changes
                    
#     if key_signature_changes == []:
#         key_signature_changes.append(KeySignature(key_number=0, time=0.0))
#         K_times = [0.0]
#         # estimate the real key signature changes
#         note_groups = [[] * len(K_times)]
#         for instrument in midi_data.instruments:
#             if not instrument.is_drum:
#                 for note in instrument.notes:
#                     if note.end > note.start:
#                         note_group = time_group(note.start, K_times)
#                         note_groups[note_group].append(note)
#         for i, notes in enumerate(note_groups):
#             histogram = get_pitch_class_histogram(notes)
#             key_candidate = np.dot(key_profile, histogram)
#             key_temp = np.where(key_candidate == max(key_candidate))
#             major_index = key_temp[0][0]
#             minor_index = key_temp[0][1]
#             major_count = histogram[major_index]
#             minor_count = histogram[minor_index % 12]
#             if major_count < minor_count:
#                 key_signature_changes[i].key_number = minor_index
#             else:
#                 key_signature_changes[i].key_number = major_index
#     else:
#         K_times = [key.time for key in key_signature_changes]

#     for instrument in midi_data.instruments:
#         if instrument.is_drum:
#             continue
#         else:
#             for note in instrument.notes:
#                 start = note.start
#                 k_group = time_group(start, K_times)
#                 real_key = key_signature_changes[k_group].key_number
                
#                 # transposite to C major or A minor
#                 if real_key <= 11:
#                     trans = 0 - real_key
#                 else:
#                     trans = 21 - real_key
#                 note.pitch += trans
#     return midi_data
