import torch, os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataloader import *

class TBPTT():
    # implemented by Alban Desmaison
    # link: https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500/4
    # a well-description of TBPTT: http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf 
    def __init__(self, one_step_module, loss_module, k1, k2, optimizer):
        self.one_step_module = one_step_module
        self.loss_module = loss_module
        self.k1 = k1
        self.k2 = k2
        # every k1 steps, backward from t to t-k2
        # if k1 < k2, two contiguous updates will overlap on [t-k2, t-k1]
        # so we need to retain the graph in this case
        self.retain_graph = k1 < k2 
        # You can also remove all the optimizer code here, and the
        # train function will just accumulate all the gradients in
        # one_step_module parameters
        self.optimizer = optimizer

    def train(self, input_sequence, init_state):
        states = [(None, init_state)]
        for j, (inp, target) in enumerate(input_sequence):

            state = states[-1][1].detach()
            state.requires_grad=True
            output, new_state = self.one_step_module(inp, state)
            states.append((state, new_state))

            while len(states) > self.k2:
                # Delete stuff that is too old
                del states[0]

            if (j+1)%self.k1 == 0:
                loss = self.loss_module(output, target)

                optimizer.zero_grad()
                # backprop last module (keep graph only if they ever overlap)
                start = time.time()
                loss.backward(retain_graph=self.retain_graph)
                # run BPTT from t to t-k2
                for i in range(self.k2-1):
                    # if we get all the way back to the "init_state", stop
                    if states[-i-2][0] is None:
                        break
                    curr_grad = states[-i-1][0].grad
                    states[-i-2][1].backward(curr_grad, retain_graph=self.retain_graph)
                #print("bw: {}".format(time.time()-start))
                optimizer.step()

class basicTrainer():
    def __init__(self, model, model_type, model_path, dataloaders, device, num_epochs=1, 
    loss_fn=None, optimizer=None, lr_scheduler=None, save_every=5, last_ckpt_dir=None):
        self.model = model
        self.model_type = model_type
        self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss()
        self.optimizer = optimizer 
        self.lr_scheduler = lr_scheduler
        self.dataloaders = dataloaders
        self.device = device
        self.model_path = model_path
        self.num_epochs = num_epochs
        self.last_ckpt_dir = last_ckpt_dir
        self.save_every = save_every
        self.records = []
        self.max_val_epoch = 0
        self.test_cases = []
        self.clip = 1.0 if self.model_type in ['rnn'] else None
    def _one_epoch(self, mode, dataloader):
        hit, losses, iters, hit_per_song = 0, 0, 0, 0
        pred_res = []
        sm = nn.Softmax()

        for batch in dataloader:
            x, y, m, l, channel_masks = batch
            y = y.to(self.device) if mode != 'infer' else y
            l = l.to(self.device) if self.model_type == 'bigbird' else l
            channel_masks = channel_masks.to(self.device)
            x = x.float().to(self.device)
            # print(l.detach().cpu().tolist())
    
            # if self.model_type == 'bigbird':
            #     output = self.model([x,m,l,channel_masks])
            #     pred = output.logits
            #     sm_out = torch.max(sm(pred),dim=1)
            #     #sm_pred = torch.argmax(sm(output.logits),dim=1)
            #     sm_pred = sm_out[1]
            # else:
            #     pred = self.model([x,m,l,channel_masks])
            #     sm_out = torch.max(sm(pred),dim=1)
            #     sm_pred = sm_out[1]
            pred = self.model([x,m,l,channel_masks])
            sm_out = torch.max(pred,dim=1)
            sm_pred = sm_out[1]
            if mode == 'infer' and self.model_type not in ['cnn','resnet']:
                sm_pred = sm_out[1].detach().cpu().tolist()
                sm_scores = sm_out[0].detach().cpu().tolist()
                ids = y.tolist()
                pred_res.extend(list(zip(ids,sm_pred,sm_scores)))

            else:
                if mode != 'infer':
                    loss = self.loss_fn(pred, y)  
                    hit += torch.where(sm_pred == y)[0].shape[0]
                    # print(list((sm_pred == y).int().cpu().numpy()))
                    self.test_cases += list((sm_pred == y).int().cpu().numpy())
                    losses += loss.item()
                    iters += 1
                pred_res.append(sm_pred.cpu().numpy())

            if mode == 'train':
                loss.backward()
                if self.clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
        # print('test')
        if mode in ['test','infer'] and self.model_type in ['cnn','resnet']:
            data_size = len(dataloader.dataset)
            pred_res = torch.from_numpy(np.concatenate(pred_res)).to(self.device)
            occur, occur_cnt = torch.unique(pred_res,return_counts=True)#.to(self.device)
            pred = occur[torch.argmax(occur_cnt)]
            if mode == 'test':
                hit_per_song += torch.where(pred == y[0])[0].shape[0]
                return data_size, hit, hit_per_song
            elif mode == 'infer':
                return (y[0].detach().cpu().tolist(), pred.detach().cpu().tolist(), max(occur_cnt.detach().cpu().tolist())/data_size)

        elif mode == 'infer':
            return pred_res

        else:
            data_size = len(dataloader.dataset)
            avg_loss = losses / iters
            avg_acc = hit / data_size
            return avg_loss, avg_acc
            

    def train_one_epoch(self, epoch):
        one_epoch_results = []
        for mode in ['train', 'val']:

            if mode == 'train':
                self.model.train()
            else:
                self.model.eval()

            with torch.set_grad_enabled(mode == 'train'):
                avg_loss, avg_acc = self._one_epoch(mode, self.dataloaders[mode])

                one_epoch_results.extend([avg_loss,avg_acc])

            print('[{}] Epoch{}: loss {}, acc {}'\
                        .format(mode,epoch,round(avg_loss,3),round(avg_acc,3)))
        return one_epoch_results
        
    def train(self):

        max_val_acc, max_val_epoch, start = 0, 0, 0

        if self.last_ckpt_dir:
            checkpoint = torch.load('model_checkpoints/{}'.format(self.last_ckpt_dir))
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.records = list(checkpoint['records'])
            start = checkpoint['epoch']
            max_val_epoch = start 
            max_val_acc = self.records[-1][-1] 
            num_epochs += start

        for epoch in range(start, self.num_epochs):
            
            #train_loss, train_acc, val_loss, val_acc = results
            results = self.train_one_epoch(epoch)
            self.records.append(np.array(results))

            if results[-1] > max_val_acc:
                self.save_checkpoint(epoch,'best')
                max_val_epoch = epoch if results[-1] > max_val_acc else max_val_epoch
                max_val_acc = results[-1]

            self.save_records()
            self.lr_scheduler.step()

        self.max_val_epoch = max_val_epoch

    def test(self,epoch):
        checkpoint = torch.load(os.path.join(self.model_path,'{}.pt'.format(epoch)),map_location=self.device)
        epoch = checkpoint['epoch'] if epoch == 'best' else epoch
        self.model.load_state_dict(checkpoint['state_dict'])
        frame_cnt, hit_per_song, hit_per_frame = 0, 0, 0
        self.model.eval()
        with torch.no_grad():
            if self.model_type in ['cnn','resnet']:
                for test_loader in self.dataloaders['test']:
                    _frame_cnt, _hit_per_frame, _hit_per_song = self._one_epoch('test',test_loader)
                    frame_cnt += _frame_cnt
                    hit_per_frame += _hit_per_frame
                    hit_per_song += _hit_per_song   
                    
                test_avg_acc = hit_per_song / len(self.dataloaders['test'])
                frame_avg_acc = hit_per_frame / frame_cnt
                print('Test Accuracy per song:', epoch, test_avg_acc)
                print('Test Accuracy per frame:', epoch, frame_avg_acc)
                test_acc_per_frame, test_acc_per_song =  frame_avg_acc, test_avg_acc
            
            elif self.model_type in ['crnn', 'bigbird', 'rnn']:              
                _, test_avg_acc = self._one_epoch('test',self.dataloaders['test'])
                print('Test Accuracy per song:', epoch, test_avg_acc)
                test_acc_per_frame, test_acc_per_song = 0, test_avg_acc
            
        self.records.append(np.array([test_acc_per_frame, test_acc_per_song, epoch, 0]))

    def infer(self):
        checkpoint = torch.load(self.model_path,map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        with torch.no_grad():       
            if self.model_type in ['cnn','resnet']:
                return [self._one_epoch('infer',test_loader)\
                for test_loader in self.dataloaders['infer']]
            elif self.model_type in ['crnn', 'bigbird']:              
                return self._one_epoch('infer',self.dataloaders['infer'])

    def save_records(self):
        np.savetxt(os.path.join(self.model_path,'records.csv'), 
        np.stack(self.records).squeeze(), delimiter=",")

    def save_checkpoint(self, epoch, filename):
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'records': np.stack(self.records).squeeze()
            }
        torch.save(checkpoint, os.path.join(self.model_path,'{}.pt'.format(filename)))
    
    def run(self):
        self.train()
        self.test('best')
        self.save_records()


# def pack_seq(seq, seq_lengths, batch_first=True):
#     pass
#     #return packed_seq


# class for CNN trainer