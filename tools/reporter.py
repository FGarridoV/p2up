import os
import pandas as pd

class Reporter:
    def __init__(self, results_dir):
        self.models = f'{results_dir}/models.csv'
        self.indicators = f'{results_dir}/indicators.csv'


    def add_model(self, dict_model):
        if not os.path.exists(self.models):
            cols = ['trainer_name', 'model_name', 'base_model', 'base_pretrained', 
                    'pooling', 'encoder_layers', 'projection_layers', 'use_dropout', 
                    'dropout_rate', 'loss_kind', 'loss_dist', 'loss_margin', 'loss_swap', 
                    'count_corrects', 'optimizer', 'lr', 'lr_embedder',
                    'lr_scheduler_step', 'lr_scheduler_gamma',
                    'weight_decay', 'n_epochs', 'n_batches', 'n_params', 'batch_size', 
                    'memory_batch_size', 'backward_freq', 'img_transform', 'train_split',
                    'val_split', 'test_split', 'data_seed', 'seed', 'device', 'best_epoch',
                    'best_val_loss', 'best_val_acc', 'elapsed_time_at_best', 'best_model_path']
            df = pd.DataFrame(columns=cols)
            df['best_model_path'] = df['best_model_path'].astype(object)
            df.loc[0] = dict_model
        else:
            df = pd.read_csv(self.models)
            newrow = pd.DataFrame(dict_model, index=[0])
            df = pd.concat([df, newrow], ignore_index=True)
        df.to_csv(self.models, index=False)


    def add_best_model(self, trainer_name, best_epoch, best_val_loss, best_val_acc, elapsed_time_at_best, best_model_path):
        df = pd.read_csv(self.models)
        cols = ['best_epoch', 'best_val_loss', 'best_val_acc', 'elapsed_time_at_best', 'best_model_path']
        df['best_model_path'] = df['best_model_path'].astype(object)
        df.loc[df['trainer_name'] == trainer_name, cols] = [int(best_epoch), round(best_val_loss,3), round(best_val_acc,3), round(elapsed_time_at_best,3), best_model_path]
        df.to_csv(self.models, index=False)


    def add_batch_indicator(self, trainer_name, epoch, batch, train_loss, train_acc, elapsed_time):
        new_indicator = {
            'trainer_name': trainer_name,
            'epoch': int(epoch),
            'batch': int(batch),
            'train_loss': round(train_loss,3),
            'train_acc': round(train_acc,3),
            'elapsed_time': round(elapsed_time,3)
        }
        if not os.path.exists(self.indicators):
            cols = ['trainer_name', 'epoch', 'batch', 'train_loss', 'train_acc', 
                    'val_loss', 'val_acc', 'elapsed_time']
            df = pd.DataFrame(columns=cols)
            df.loc[0] = new_indicator
        else:
            df = pd.read_csv(self.indicators)
            newrow = pd.DataFrame(new_indicator, index=[0])
            df = pd.concat([df, newrow], ignore_index=True)
        df.to_csv(self.indicators, index=False)

    
    def add_epoch_indicator(self, trainer_name, epoch, train_loss, train_acc, val_loss, val_acc, elapsed_time):
        new_indicator = {
            'trainer_name': trainer_name,
            'epoch': int(epoch),
            'batch': 'full',
            'train_loss': round(train_loss,3),
            'train_acc': round(train_acc,3),
            'val_loss': round(val_loss,3),
            'val_acc': round(val_acc,3),
            'elapsed_time': round(elapsed_time,3)
        }
        if not os.path.exists(self.indicators):
            cols = ['trainer_name', 'epoch', 'batch', 'train_loss', 'train_acc', 
                    'val_loss', 'val_acc', 'elapsed_time']
            df = pd.DataFrame(columns=cols)
            df.loc[0] = new_indicator
        else:
            df = pd.read_csv(self.indicators)
            newrow = pd.DataFrame(new_indicator, index=[0])
            df = pd.concat([df, newrow], ignore_index=True)
        df.to_csv(self.indicators, index=False)
