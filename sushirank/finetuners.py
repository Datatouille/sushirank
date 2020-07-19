import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import (
    AdamW, 
    get_linear_schedule_with_warmup, 
)
#emb_sz_rule from fastai: https://github.com/fastai/fastai/blob/master/fastai/tabular/data.py
def emb_sz_rule(n_cat:int)->int: return min(600, round(1.6 * n_cat**0.56))

class PointwiseFinetuner(pl.LightningModule):
    def __init__(self, hparams,train_dataset,valid_dataset,test_dataset):
        super(PointwiseFinetuner, self).__init__()
        
        self.hparams = hparams
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        
        #construct layers
        self.n_cat = sum([emb_sz_rule(i) for i in self.hparams.cat_dims])
        self.n_num = self.hparams.n_num
        self.emb = torch.nn.ModuleList([torch.nn.Embedding(i, emb_sz_rule(i)) for i in self.hparams.cat_dims])
        self.emb_droupout = torch.nn.Dropout(self.hparams.emb_drop)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.n_cat + self.n_num, self.hparams.num_hidden),
            torch.nn.Dropout(p=self.hparams.drop),
            torch.nn.Linear(self.hparams.num_hidden, 1),
            torch.nn.Sigmoid()
        )
        
        #loss
        self.loss_fn = torch.nn.MSELoss()

    def forward(self,inp):
        cat_x = inp['cat_feature']
        cat_x = [e(cat_x[:,i]) for i,e in enumerate(self.emb)]
        cat_x = torch.cat(cat_x, 1)
        cat_x = self.emb_droupout(cat_x)
        x = torch.cat([cat_x,inp['num_feature']],1)
        x = self.head(x)
        # x = (self.hparams.y_range[1]-self.hparams.y_range[0]) * x + self.hparams.y_range[0]
        return x
    
    def _step(self, batch):
        preds = self.forward(batch)
        loss = self.loss_fn(preds, batch['label'])
        return loss, preds

    def training_step(self, batch, batch_nb):
        loss, _ = self._step(batch)
        tensorboard_logs = {'train_loss': loss.cpu()}
        return {'loss': loss.cpu(), 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        loss, preds = self._step(batch)
        tensorboard_logs = {'train_loss': loss.cpu()}
        return {'loss': loss.cpu(), 'log': tensorboard_logs}
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = np.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_val_loss,}
        return {'val_loss': avg_val_loss,
                'log': tensorboard_logs,
                'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        loss, preds = self._step(batch)
        tensorboard_logs = {'train_loss': loss.cpu()}
        return {'loss': loss.cpu(), 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_test_loss = np.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_test_loss,}
        return {'test_loss': avg_test_loss,
                'log': tensorboard_logs,
                'progress_bar': tensorboard_logs}
    
    def configure_optimizers(self):
        no_decay = ["bias"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None,
        on_tpu=False, using_native_amp=False, using_lbfgs=False
    ):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.per_device_train_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=0,
        )
        
        #calculate total timesteps
        t_total = (
            (
                len(dataloader.dataset)
                // (self.hparams.per_device_train_batch_size * max(1, self.hparams.n_gpu))
            )
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        #create scheduler
        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=t_total,
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=self.hparams.per_device_eval_batch_size, num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.hparams.per_device_eval_batch_size, num_workers=0
        )
    
class PairwiseFinetuner(pl.LightningModule):
    def __init__(self, hparams,train_dataset,valid_dataset,test_dataset):
        super(PairwiseFinetuner, self).__init__()
        
        self.hparams = hparams
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        
        #construct layers
        self.n_cat = sum([emb_sz_rule(i) for i in self.hparams.cat_dims])
        self.n_num = self.hparams.n_num
        self.emb = torch.nn.ModuleList([torch.nn.Embedding(i, emb_sz_rule(i)) for i in self.hparams.cat_dims])
        self.emb_droupout = torch.nn.Dropout(self.hparams.emb_drop)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.n_cat + self.n_num, self.hparams.num_hidden),
            torch.nn.Dropout(p=self.hparams.drop),
            torch.nn.Linear(self.hparams.num_hidden, 1),
        )
        
        #loss
        self.loss_fn = torch.nn.BCEWithLogitsLoss()


    def predict(self,inp):
        cat_i = inp['cat_feature_i']
        cat_i = [e(cat_i[:,idx]) for idx,e in enumerate(self.emb)]
        cat_i = torch.cat(cat_i, 1)
        cat_i = self.emb_droupout(cat_i)
        x_i = torch.cat([cat_i,inp['num_feature_i']],1)
        x_i = self.head(x_i)
        return x_i

    def forward(self,inp):
        #i
        cat_i = inp['cat_feature_i']
        cat_i = [e(cat_i[:,idx]) for idx,e in enumerate(self.emb)]
        cat_i = torch.cat(cat_i, 1)
        cat_i = self.emb_droupout(cat_i)
        x_i = torch.cat([cat_i,inp['num_feature_i']],1)
        x_i = self.head(x_i)

        #j
        cat_j = inp['cat_feature_j']
        cat_j = [e(cat_j[:,idx]) for idx,e in enumerate(self.emb)]
        cat_j = torch.cat(cat_j, 1)
        cat_j = self.emb_droupout(cat_j)
        x_j = torch.cat([cat_j,inp['num_feature_j']],1)
        x_j = self.head(x_j)

        return x_i-x_j
    
    def _step(self, batch):
        preds = self.forward(batch)
        loss = self.loss_fn(preds, batch['label'])
        return loss, preds

    def training_step(self, batch, batch_nb):
        loss, _ = self._step(batch)
        tensorboard_logs = {'train_loss': loss.cpu()}
        return {'loss': loss.cpu(), 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        loss, preds = self._step(batch)
        tensorboard_logs = {'train_loss': loss.cpu()}
        return {'loss': loss.cpu(), 'log': tensorboard_logs}
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = np.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_val_loss,}
        return {'val_loss': avg_val_loss,
                'log': tensorboard_logs,
                'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        loss, preds = self._step(batch)
        tensorboard_logs = {'train_loss': loss.cpu()}
        return {'loss': loss.cpu(), 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_test_loss = np.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_test_loss,}
        return {'test_loss': avg_test_loss,
                'log': tensorboard_logs,
                'progress_bar': tensorboard_logs}
    
    def configure_optimizers(self):
        no_decay = ["bias"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None,
        on_tpu=False, using_native_amp=False, using_lbfgs=False
    ):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.per_device_train_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=0,
        )
        
        #calculate total timesteps
        t_total = (
            (
                len(dataloader.dataset)
                // (self.hparams.per_device_train_batch_size * max(1, self.hparams.n_gpu))
            )
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        #create scheduler
        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=t_total,
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=self.hparams.per_device_eval_batch_size, num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.hparams.per_device_eval_batch_size, num_workers=0
        )