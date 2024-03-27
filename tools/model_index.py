import os
import pandas as pd

class ModelIndex:

    M_COLUMNS = ['model_index', 
               'model_name', 
               'n_parameters',
               'n_images', 
               'base_model', 
               'base_pretrained',
               'pooling', 
               'encoder_layers', 
               'projection_layers', 
               'use_dropout',
               'dropout_rate',
               'loss_kind', 
               'loss_dist', 
               'loss_margin',
               'loss_swap',
               'loss_reduction',                      
               'optimizer',
               'lr',
               'adjust_lr', 
               'weight_decay', 
               'batch_size', 
               'num_epochs']
    
    E_COLUMNS = ['model_index', 
                 'epoch', 
                 'train_loss', 
                 'val_loss', 
                 'train_accuracy', 
                 'val_accuracy']
    
    B_COLUMNS = ['model_index',
                 'epoch',
                 'batch_set',
                 'batch_i',
                 'batch_f',
                 'train_loss',
                 'val_loss',
                 'train_accuracy',
                 'val_accuracy']

    def __init__(self, root = 'results/', models = 'models.csv', epochs = 'epochs.csv', batches = 'batches.csv'):
        self.root = root
        self.models = models
        self.epochs = epochs
        self.batches = batches

        if not os.path.exists(os.path.join(root, models)):
            self._create_new_index_csv('models')

        if not os.path.exists(os.path.join(root, epochs)):
            self._create_new_index_csv('epochs')

        if not os.path.exists(os.path.join(root, batches)):
            self._create_new_index_csv('batches')


    def _create_new_index_csv(self, index):
        columns, file = ((ModelIndex.M_COLUMNS, self.models) if index == 'models' else 
                   (ModelIndex.E_COLUMNS, self.epochs) if index == 'epochs' else
                   (ModelIndex.B_COLUMNS, self.batches))[0]
        df = pd.DataFrame(columns = columns)
        df.to_csv(os.path.join(self.root, file), index = False)

                    
    def register_model(self, model_name, 
                             n_parameters,
                             n_images, 
                             base_model, 
                             base_pretrained,
                             pooling, 
                             encoder_layers, 
                             projection_layers, 
                             use_dropout,
                             dropout_rate,
                             loss_kind, 
                             loss_dist, 
                             loss_margin,
                             loss_swap,
                             loss_reduction,                      
                             optimizer,
                             lr,
                             adjust_lr, 
                             weight_decay, 
                             batch_size, 
                             num_epochs):
        df = pd.read_csv(os.path.join(self.root, self.index))
        model_index = len(df) + 1
        new_row = pd.DataFrame({'model_index': [model_index],
                                'model_name': [model_name],
                                'n_parameters': [n_parameters],
                                'n_images': [n_images],
                                'base_model': [base_model],
                                'base_pretrained': [base_pretrained],
                                'pooling': [pooling],
                                'encoder_layers': [encoder_layers],
                                'projection_layers': [projection_layers],
                                'use_dropout': [use_dropout],
                                'dropout_rate': [dropout_rate],
                                'loss_kind': [loss_kind],
                                'loss_dist': [loss_dist],
                                'loss_margin': [loss_margin],
                                'loss_swap': [loss_swap],
                                'loss_reduction': [loss_reduction],                      
                                'optimizer': [optimizer],
                                'lr': [lr],
                                'adjust_lr': [adjust_lr],
                                'weight_decay': [weight_decay],
                                'batch_size': [batch_size],
                                'num_epochs': [num_epochs]})
        
        df = pd.concat([df, new_row], ignore_index = True)
        df.to_csv(os.path.join(self.root, self.index), index = False)
        return model_index
    

    def register_epoch(self, model_index, epoch, train_loss, val_loss, train_accuracy, val_accuracy):
        df = pd.read_csv(os.path.join(self.root, self.index))
        new_row = pd.DataFrame({'model_index': [model_index],
                                'epoch': [epoch],
                                'train_loss': [train_loss],
                                'val_loss': [val_loss],
                                'train_accuracy': [train_accuracy],
                                'val_accuracy': [val_accuracy]})
        
        df = pd.concat([df, new_row], ignore_index = True)
        df.to_csv(os.path.join(self.root, self.index), index = False)


    def register_batch(self, model_index, epoch, batch_set, batch_i, batch_f, train_loss, val_loss, train_accuracy, val_accuracy):
        df = pd.read_csv(os.path.join(self.root, self.index))
        new_row = pd.DataFrame({'model_index': [model_index],
                                'epoch': [epoch],
                                'batch_set': [batch_set],
                                'batch_i': [batch_i],
                                'batch_f': [batch_f],
                                'train_loss': [train_loss],
                                'val_loss': [val_loss],
                                'train_accuracy': [train_accuracy],
                                'val_accuracy': [val_accuracy]})
        
        df = pd.concat([df, new_row], ignore_index = True)
        df.to_csv(os.path.join(self.root, self.index), index = False)