import os
import time

class Logger:
    def __init__(self, model_dir, verbose):
        self.filename = f'{model_dir}/log.txt'
        self.verbose = verbose
        os.system(f'touch {self.filename}')


    def log(self, message):
        now = time.strftime('%H:%M:%S %d/%m/%Y')

        if self.verbose:
            print(f'[{now}] {message}')

        with open(self.filename, 'a') as file:
            file.write(f'[{now}] {message}\n')


    def log_info(self, message):
        if self.verbose:
            print(message)

        with open(self.filename, 'a') as file:
            file.write(f'{message}\n')


    def log_header(self, trainer_name, device, use_tensorboard):
        self.log_info('-'*(29 + len(trainer_name)))
        self.log_info(f'| PLACE EMBEDDING TRAINER: {trainer_name} |')
        self.log_info('-'*(29 + len(trainer_name)))
        self.log_info(f'Working on device: {device}')
        self.log_info(f'Using tensorboard: {use_tensorboard}')
        self.log_info('')
    

    def log_data(self, n_triplets, n_trainset, n_valset, batch_size, memory_batch_size):
        self.log_info(f'Dataset:')
        self.log_info(f'    Total triplets: {n_triplets:,}')
        self.log_info(f'    Training set size: {n_trainset:,} triplets ({n_trainset/n_triplets:.3%})')
        self.log_info(f'    Validation set size: {n_valset:,} triplets ({n_valset/n_triplets:.3%})')
        self.log_info(f'    Reserved for testing: {n_triplets - n_trainset - n_valset:,} triplets ({(n_triplets - n_trainset - n_valset)/n_triplets:.3%})')
        if batch_size == memory_batch_size or memory_batch_size is None:
            self.log_info(f'    Batch size: {batch_size} (no memory batching)')
        else:
            self.log_info(f'    Batch size: {batch_size} (looping on {batch_size//memory_batch_size} memory batches - {memory_batch_size} instances)')
        self.log_info('')


    def log_model(self, name, base_model, emb_size, pretrained, img2vec_encoder_layers, pooling, 
                   encoder_layers, projection_layers, use_dropout, dropout_rate, n_params):
        self.log_info(f'Model:') 
        self.log_info(f'    Name: {name}')
        if pretrained:
            self.log_info(f'    Image embedding model: {base_model} -> {emb_size} D embedding space')
        else:
            self.log_info(f'    Image embedding model: {base_model} -> {emb_size} D embedding space')
        self.log_info(f'    Image emb. pretrained: {pretrained}')

        if img2vec_encoder_layers is not None:
            self.log_info(f'    Image2Vec encoder layers: {emb_size} -> {img2vec_encoder_layers}')
            emb_size = img2vec_encoder_layers[-1] # OJO
        
        if pooling == 'concat':
            self.log_info(f'    Pooling rule: {pooling} -> {emb_size * 5} D embedding space')
            if encoder_layers is not None:
                self.log_info(f'    Encoder layers: {emb_size * 5} -> {encoder_layers}')
        else:
            size = emb_size * (pooling.count('-') + 1)
            self.log_info(f'    Pooling rule: {pooling} -> {size} D embedding space')
            if encoder_layers is not None:
                self.log_info(f'    Encoder layers: {size} -> {encoder_layers}')
                
        if projection_layers is not None:
            self.log_info(f'    Projection layers: {encoder_layers[-1]} -> {projection_layers}')

        if use_dropout:
            self.log_info(f'    Dropout rate: {dropout_rate}')
        else:
            self.log_info(f'    Dropout: False')

        self.log_info(f'    Number of parameters: {n_params:,}')
        self.log_info('')


    def log_loss(self, loss_kind, loss_dist, loss_margin, loss_swap, count_corrects):
        self.log_info(f'Loss function:') 
        self.log_info(f'    Kind: {loss_kind} loss')
        self.log_info(f'    Distance metric: {loss_dist}')
        self.log_info(f'    Margin: {loss_margin}')
        self.log_info(f'    Swap: {loss_swap}')
        self.log_info(f'    Count corrects: {count_corrects}')
        self.log_info('')


    def log_optimizer(self, optimizer, lr, wd, backward_freq, lr_embedder, lr_scheduler_step, lr_scheduler_gamma):
        self.log_info(f'Optimizer:')
        self.log_info(f'    Name: {optimizer}')
        self.log_info(f'    Learning rate: {lr}')
        self.log_info(f'    Weight decay: {wd}')
        self.log_info(f'    Backpropagation frequency: {backward_freq}')
        if lr_embedder != lr:
            if lr_embedder == 'no_train':
                self.log_info(f'    Learning rate embedder: No training')
            else:
                self.log_info(f'    Learning rate img embedder: {lr_embedder}')
        if lr_scheduler_step is not None and lr_scheduler_gamma is not None:
            adjust_lr = f'Every {lr_scheduler_step} epochs, multiply by {lr_scheduler_gamma}'
        self.log_info(f'    Learning rate scheduler: {adjust_lr}')
        self.log_info('')
        self.log_info('')


    def log_training(self, n_epochs, n_batches):
        self.log_info(f'MODEL TRAINING: {n_epochs} epochs | {n_batches} batches')
        self.log_info(f'{"-"*45}')
        self.log_info(f'')
    
    def log_epoch_times(self, epoch, epoch_time, total_time):
        self.log_info(f'Epoch {epoch} Finished in {epoch_time/60:.3f} minutes')
        self.log_info(f'Total time: {total_time/60:.2f} minutes')
        self.log_info(f'')

    
    def log_finish(self):
        self.log_info(f'{"-"*45}')
        self.log_info(f'{"TRAINING FINISHED":^45}')
        self.log_info(f'{"-"*45}')
        self.log_info(f'')

    




    

        