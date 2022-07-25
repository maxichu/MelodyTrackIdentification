import os, time, datetime, json, pickle
from model import *
from dataloader import *
from trainer import *
import torch, transformers
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BigBirdConfig


SEED = 42

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_dataloader(train_ids, val_ids, test_ids, batch_size, model, mode, min_len, max_len=9000):

    def pad_collate(batch):
        xx, yy, mm, ll, channel_masks = zip(*batch)
        xx = [torch.from_numpy(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        return xx_pad, torch.tensor(yy), torch.tensor(mm), torch.tensor(ll), torch.tensor(channel_masks)

    def att_mask_collate(batch):
        xx, yy, mm, ll, channel_masks = zip(*batch)
        xx = [torch.from_numpy(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        bs, seq, _ ,_ ,_ = xx_pad.size()
        mask = np.zeros((bs, seq))
        for i, length in enumerate(ll):
            mask[i,:length] = 1
        return xx_pad, torch.tensor(yy), torch.tensor(mm), torch.from_numpy(mask).float(), torch.tensor(channel_masks)

    collate_funcs = {'cnn':None, 'resnet':None, 'crnn':pad_collate, 'bigbird':att_mask_collate, 'rnn':pad_collate}
    train_loader, val_loader = None, None
    
    if mode == 'train':
        train_dataset = MIDIDataset(song_dirs=train_ids, model=model)
        val_dataset = MIDIDataset(song_dirs=val_ids, model=model)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, \
            num_workers=0, worker_init_fn=np.random.seed(SEED),collate_fn=collate_funcs[model])
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, \
            num_workers=0, worker_init_fn=np.random.seed(SEED),collate_fn=collate_funcs[model])
        
    if model in ['cnn','resnet']:
        test_loader = []
        for song_id in test_ids:
            test_dataset = MIDIDataset(song_dirs=[song_id], frame_shuffle=False, model=model)
            test_loader.append(DataLoader(test_dataset, batch_size=batch_size, shuffle=False, \
                num_workers=0, worker_init_fn=np.random.seed(SEED)))
    elif model in ['crnn','bigbird','rnn']:
        test_dataset = MIDIDataset(song_dirs=test_ids, frame_shuffle=False, model=model)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, \
                num_workers=0, worker_init_fn=np.random.seed(SEED),collate_fn=collate_funcs[model])
        

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}

def select_model(config, device):
    if config['model'] == 'cnn':
        net = ConvNet().float().to(device)    

    elif config['model'] == 'resnet':
        net = ResConvNet(resblock_opt=config['option'],dropout1=config['dropout1'],dropout2=config['dropout2']).float().to(device)
    
    elif config['model'] == 'crnn':
        net = MelodyNet(resblock_opt=config['option'], hidden_dim=config['hidden_dim'], rnn_layer=config['rnn_layer'],
        dropout=config['dropout_after_rnn'], rnn_bi=config['bidirectional'], 
        device=config['device'], pretrained=config['pretrained']).float().to(device)

    elif config['model'] == 'bigbird':
        bigbirdconfig = BigBirdConfig(num_hidden_layers=config['bigbird_layer'], hidden_size=config['hidden_dim'],
            num_attention_heads=config['num_att_head'], num_random_blocks=config['num_random_blocks'],
            hidden_dropout_prob=config['hidden_dropout_prob'], classifier_dropout=config['classifier_dropout'], 
            block_size=config['block_size'], num_labels=16, max_position_embeddings=13000)
        net = MelodyNet(resblock_opt=config['option'],seq_head='bigbird', pretrained=config['pretrained'],
        bigbirdconfig=bigbirdconfig, device=config['device']).float().to(device)
    elif config['model'] == 'rnn':
        net = MelodyRNN().float().to(device)
    return net

def train(config):
     #create folder
    print(config)
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
    suffix = config['option'] if config['model'] != 'bigbird' else {'cosine':1,'exp':2}[config['lr_scheduler']]
    model_path = 'model_checkpoints/{}_{}'.format(st,config['job_no']) if config['data_prop'] != -2 \
        else 'model_checkpoints/kfolds/{}_{}_{}/{}'.format(config['model'],suffix,config['job_no'],config['kfolds'])
    if config['mode'] == 'train':
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        with open(os.path.join(model_path,'config.json'), 'w') as f:
            json.dump(config, f)

    batch_size = config['batch_size']
    learning_rate = config['lr']
    num_epochs = config['epoch']
    #os.environ['CUDA_VISIBLE_DEVICES'] = config['cude_device_number']
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    config['device'] = device
    print('using ', device)
    load_prop = config['data_prop']
    weight_decay = config['weight_decay']
    warmup_ratio = config['warmup_ratio']

    # set seed
    set_seed(SEED)

    # load list of file names of samples
    if load_prop == -1:
        print('testing idea')
        data_dir = 'npy/summary/{}.pkl'.format('train_test_ids_partial') 
    elif load_prop == -2:
        print('k folds experiment')
        data_dir = 'npy/kfolds/train_test_ids_kfolds_{}.pkl'.format(config['kfolds']) 
    else:
        data_dir = 'npy/summary/{}.pkl'.format('train_test_ids') 

    with open(data_dir,'rb') as f:
        train_data_dirs, test_data_dirs = pickle.load(f)
        f.close()

    # load songs for training
    # split dataset into training and validation set by 8:1
    # create dataloaders
    if load_prop < 1 and load_prop > 0:
        _, train_data_dirs = train_test_split(train_data_dirs, test_size = load_prop, random_state=SEED)
        _, test_data_dirs = train_test_split(test_data_dirs, test_size = load_prop, random_state=SEED)
    
    train_ids, val_ids = train_data_dirs if load_prop == -2 \
        else train_test_split(train_data_dirs, test_size = 0.125, random_state=SEED)
 
    min_len = (5+2*config['num_random_blocks'])*config['block_size']
    dataloaders = get_dataloader(train_ids, val_ids, test_data_dirs, batch_size, config['model'], config['mode'], min_len)
    print('Dataset Loaded.')

    # select a model, corresponding train and test function     
    net = select_model(config, device)
    loss_fn = nn.CrossEntropyLoss()

    if config['mode'] == 'train':
        if config['optim'] == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay) 
        elif config['optim'] == 'adadelta':
            optimizer = torch.optim.Adadelta(net.parameters())
        elif config['optim'] == 'RMSprop':
            optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif config['optim'] == 'adamW':
            optimizer = transformers.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        if config['lr_scheduler'] == 'exp' and not config['lr_warmup']:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
        # elif config['lr_scheduler'] == 'reduce' and not config['lr_warmup']:
        #     lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5)
        elif config['lr_scheduler'] == 'cosine' and config['lr_warmup'] and not config['hard_restart']:
            lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer,
                num_warmup_steps=int(num_epochs*warmup_ratio),num_training_steps=num_epochs)
        elif config['lr_scheduler'] == 'cosine' and config['lr_warmup'] and config['hard_restart']:
            lr_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer=optimizer, num_warmup_steps=int(num_epochs*warmup_ratio),num_training_steps=num_epochs)
        elif config['lr_scheduler'] == 'linear' and config['lr_warmup']:
            lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer, 
                num_warmup_steps=int(num_epochs*warmup_ratio),num_training_steps=num_epochs)

        trainer = basicTrainer(net, config['model'],  model_path, dataloaders, device, 
        num_epochs, loss_fn, optimizer, lr_scheduler, save_every=5) 
        trainer.run()

    elif config['mode'] == 'test':
        trainer = basicTrainer(net, config['model'], config['test_path'], dataloaders, 
        device, num_epochs, loss_fn, save_every=5)   
        for fp in os.listdir(config['test_path']):
            if fp.split('.')[-1] in ['pt']:
                trainer.test(fp.split('.')[0])
        result_lists = {'correct':[],'wrong':[]}
        for i, result in enumerate(trainer.test_cases):
            if result == 1:
                result_lists['correct'].append(test_data_dirs[i])
            else:
                result_lists['wrong'].append(test_data_dirs[i])
        with open('codes/case_analysis/bigbird.pkl','wb') as f:
            test_data_dirs
            pickle.dump(result_lists, f)
            
    print('Work Done.')