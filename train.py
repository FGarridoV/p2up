from tools.trainer import PlaceEmbeddingTrainer
import pandas as pd
import ast

# Hyperparameters
batch_size = 40
memory_batch_size = 8
num_epochs = 3
img_transform = 'augmentation'    # 'default' or 'augmentation'
base_model = 'resnet18'      # many options available
pooling = 'mean'             # 'mean', 'std', 'max', 'min', 'median', 'concat' or any combination
encoder_layers = [512]       # None or any list of integers
projection_layers = [256]    # None or any list of integers
use_dropout = True
dropout_rate = 0.3
loss_dist = 'euclidean'      # 'euclidean' or 'cosine'
loss_margin = 0.2
loss_swap = True
lr = 0.001
lr_embedder = 'no_train'          # any value or 'no_train'
lr_scheduler_step = 10       # None or any value
lr_scheduler_gamma = 0.1     # None or any value
weight_decay = 0.001         # 0.0 or any value
backward_freq = 'batch'      # 'mbatch' or 'batch'


if __name__ == '__main__':

    # Keep fixed
    seed = 46  
    data_splits = {'train': 0.8, 
                'val': 0.1, 
                'test': 0.1}
    base_pretrained = True
    loss_kind = 'triplet'
    count_corrects = True 
    optimizer = 'adam'
    verbose = True              # Print the progress
    gpu = True                  # Try to use GPU
    use_tensorboard = True      # Use tensorboard for logging 

    df = pd.read_csv('train_scheduler.csv', sep = ';')

    for i, row in df.iterrows():
        if row['status'] == 'pending':

            batch_size = int(row['batch_size'])
            memory_batch_size = int(row['memory_batch_size']) if row['memory_batch_size'] != 'None' else None
            num_epochs = int(row['num_epochs'])
            img_transform = row['img_transform']
            base_model = row['base_model']
            pooling = row['pooling']
            encoder_layers = ast.literal_eval(row['encoder_layers'])
            projection_layers = ast.literal_eval(row['encoder_layers'])
            use_dropout = row['use_dropout']
            dropout_rate = float(row['dropout_rate'])
            loss_dist = row['loss_dist']
            loss_margin = float(row['loss_margin'])
            loss_swap = row['loss_swap']
            lr = float(row['lr'])
            lr_embedder = float(row['lr_embedder']) if row['lr_embedder'] != 'no_train' else 'no_train'
            lr_scheduler_step = int(row['lr_scheduler_step']) if row['lr_scheduler_step'] != 'None' else None
            lr_scheduler_gamma = float(row['lr_scheduler_gamma']) if row['lr_scheduler_gamma'] != 'None' else None
            weight_decay = float(row['weight_decay'])
            backward_freq = row['backward_freq']

            trainer = PlaceEmbeddingTrainer(name = None,
                                            use_gpu = gpu,
                                            use_tensorboard = use_tensorboard, 
                                            verbose = verbose)

            # Set the data
            trainer.set_data(triplets_path = 'data/triplets.csv',
                            img_transform = img_transform, 
                            data_splits = data_splits, 
                            batch_size = batch_size, 
                            memory_batch_size = memory_batch_size)

            # Define the model
            trainer.set_model(base_model = base_model, base_pretrained = base_pretrained, pooling = pooling, 
                            encoder_layers = encoder_layers, projection_layers = projection_layers,
                            use_dropout = use_dropout, dropout_rate = dropout_rate)

            # Define the loss
            trainer.set_loss(loss_kind = loss_kind, loss_dist = loss_dist, loss_margin = loss_margin, 
                            loss_swap = loss_swap, count_corrects = count_corrects)

            # Define the optimizer
            trainer.set_optimizer(optimizer = optimizer, learning_rate = lr, lr_embedder = lr_embedder, 
                                lr_scheduler_step = lr_scheduler_step, lr_scheduler_gamma = lr_scheduler_gamma, 
                                weight_decay = weight_decay, backward_freq = backward_freq)

            # Train the model
            trainer.train(num_epochs = num_epochs)

            # Change status to done
            df.loc[i, 'status'] = 'done'
            df.to_csv('train_scheduler.csv', index = False, sep = ';')
