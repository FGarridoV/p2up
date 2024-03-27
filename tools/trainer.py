import os
import time
import math
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from models.dataset import TripletDataset
from models.models import PlaceEmbedding
from tools.transforms import get_transform
from tools.logger import Logger
from tools.reporter import Reporter


class PlaceEmbeddingTrainer(object):

    RESULTS_DIR = 'results'
    def __init__(self, name, criterion = 'loss', seed = 42, use_gpu = True, use_tensorboard = True, verbose = True):

        # General parameters
        self.name = time.strftime('%Y%m%d_%H%M%S')
        if name is not None:
            self.name = self.name + '_' + name
        self.seed = seed
        PlaceEmbeddingTrainer.set_seed(seed)
        self.trainer_dir = PlaceEmbeddingTrainer.create_trainer_folder(self.name)
        self.device = PlaceEmbeddingTrainer.get_device(use_gpu)
        
        # Logger and tensorboard
        self.logger = Logger(self.trainer_dir, verbose)
        self.reporter = Reporter(PlaceEmbeddingTrainer.RESULTS_DIR)
        self.use_tensorboard = use_tensorboard
        self.writer = PlaceEmbeddingTrainer.get_tensorboard_writer(self.trainer_dir) if use_tensorboard else None

        # Data
        self.batch_size = None
        self.memory_batch_size = None
        self.train_loader = None
        self.val_loader = None

        # Model, loss, optimizer
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.lr_scheduler = None
        self.backward_freq = None

        # Extras
        self.n_batches = None
        self.unix_0 = None
        self.epoch_times = []
        self.trainer_dict = {}
        self.criterion = criterion
        self.best = {'epoch': 0, 'val_loss': np.inf, 'val_acc': 0, 'elapsed_time': 0, 'model_path': None} 

        self.logger.log_header(self.name, self.device, use_tensorboard)

        
    def set_data(self, triplets_path, img_transform, data_splits, batch_size, memory_batch_size, data_seed = 21, num_workers = 0):

        if memory_batch_size is None:
            memory_batch_size = batch_size

        if batch_size % memory_batch_size != 0:
            raise ValueError('Memory batch size must be a divisor of batch size')

        transform = get_transform(img_transform)
        dataset = TripletDataset(data = triplets_path, transform = transform)
        trs, vls = int(data_splits['train'] * len(dataset)), int(data_splits['val'] * len(dataset))
        trainset, valset, _ = random_split(dataset, [trs, vls, len(dataset) - trs - vls], generator = torch.Generator().manual_seed(data_seed))
        
        self.batch_size = batch_size
        self.memory_batch_size = memory_batch_size

        batch = batch_size if memory_batch_size is None else memory_batch_size
        self.train_loader = DataLoader(trainset, batch_size = batch, shuffle=True, num_workers=num_workers) 
        self.val_loader = DataLoader(valset, batch_size = batch, shuffle=False, num_workers=num_workers)

        if memory_batch_size == batch_size:
            self.n_batches = len(self.train_loader)
        else:
            self.n_batches = int(math.ceil(len(self.train_loader)/(batch_size / memory_batch_size)))

        self.trainer_dict.update({'img_transform': str(img_transform),
                                  'train_split': len(self.train_loader.dataset),
                                  'val_split': len(self.val_loader.dataset),
                                  'test_split': len(dataset) - len(self.train_loader.dataset) - len(self.val_loader.dataset),
                                  'data_seed': data_seed})

        self.logger.log_data(len(dataset), len(trainset), len(valset), batch_size, memory_batch_size)
    

    def set_model(self, base_model, base_pretrained, pooling,     
                     encoder_layers, projection_layers, use_dropout, 
                     dropout_rate, n_images = 5):
        
        model_name = PlaceEmbeddingTrainer.model_name(base_model, base_pretrained, pooling, 
                                                      encoder_layers, projection_layers, 
                                                      use_dropout, dropout_rate)
        
        self.model = PlaceEmbedding(model_name, n_images,
                                    base_model, base_pretrained, pooling, 
                                    encoder_layers, projection_layers, 
                                    use_dropout, dropout_rate)
        
        self.model = self.model.to(self.device)

        self.trainer_dict.update({'base_model': str(base_model),
                                  'base_pretrained': str(base_pretrained),
                                  'pooling': str(pooling),
                                  'encoder_layers': str(encoder_layers),
                                  'projection_layers': str(projection_layers),
                                  'use_dropout': str(use_dropout),
                                  'dropout_rate': float(dropout_rate)})

        self.logger.log_model(self.model.name, base_model, self.model.vec_size,
                              base_pretrained, pooling, encoder_layers, projection_layers, use_dropout, dropout_rate,
                              self.model.num_params)

    
    def set_loss(self, loss_kind, loss_dist, loss_margin, loss_swap, count_corrects):

        if loss_kind == 'triplet':
            from models.models import TripletLoss
            self.loss_fn = TripletLoss(dist = loss_dist, margin = loss_margin, 
                                       swap = loss_swap, 
                                       count_corrects = count_corrects)
        else:
            raise ValueError(f'Loss function {loss_kind} not implemented')
        
        self.trainer_dict.update({'loss_kind': str(loss_kind),
                                  'loss_dist': str(loss_dist),
                                  'loss_margin': float(loss_margin),
                                  'loss_swap': str(loss_swap),
                                  'count_corrects': str(count_corrects)})
        
        self.logger.log_loss(loss_kind, loss_dist, loss_margin, loss_swap, count_corrects)
        

    def set_optimizer(self, optimizer, learning_rate, lr_embedder, lr_scheduler_step, lr_scheduler_gamma,
                      weight_decay, backward_freq):
        if self.model is None:
            raise ValueError('Model not set') 
        self.backward_freq = backward_freq

        if optimizer == 'adam':
            from torch.optim import Adam
            self.optimizer = Adam(self.model.parameters(), lr = learning_rate, weight_decay = weight_decay)
        else:
            raise ValueError(f'Optimizer {optimizer} not implemented')
        
        if lr_scheduler_step is not None and lr_scheduler_gamma is not None:
            from torch.optim.lr_scheduler import StepLR
            self.lr_scheduler = StepLR(self.optimizer, step_size = lr_scheduler_step, gamma = lr_scheduler_gamma)
        
        if lr_embedder == 'no_train':
            for param in self.model.img2vec.parameters():
                param.requires_grad = False

        self.trainer_dict.update({'optimizer': str(optimizer),
                                  'lr': float(learning_rate),
                                  'lr_embedder': lr_embedder,
                                  'lr_scheduler_step': int(lr_scheduler_step),
                                  'lr_scheduler_gamma': float(lr_scheduler_gamma),
                                  'weight_decay': float(weight_decay),
                                  'backward_freq': str(backward_freq)})
        
        self.logger.log_optimizer(optimizer, learning_rate, weight_decay, backward_freq, lr_embedder, lr_scheduler_step, lr_scheduler_gamma)


    def train(self, num_epochs):
            
        if self.model is None:
            raise ValueError('Model not set')
        if self.loss_fn is None:
            raise ValueError('Loss function not set')
        if self.optimizer is None:
            raise ValueError('Optimizer not set')
        if self.train_loader is None or self.val_loader is None:
            raise ValueError('Data not set')
        
        self.trainer_dict.update({'n_epochs': num_epochs})
        self.reporter.add_model(self.trainer_to_dict())
        self.logger.log_training(num_epochs, self.n_batches)

        self.unix_0 = time.time()
        for epoch in range(1, num_epochs + 1):

            train_loss, train_accuracy = self._train_one_epoch(epoch)
            val_loss, val_accuracy = self._eval_one_epoch(epoch)
            epoch_time = time.time() - self.unix_0
            self.reporter.add_epoch_indicator(self.name, epoch, train_loss, train_accuracy, val_loss, val_accuracy, epoch_time)

            if self.criterion == 'loss':
                if val_loss < self.best['val_loss']:
                    self.new_best(epoch, val_loss, val_accuracy, epoch_time)
                    self.reporter.add_best_model(self.name, epoch, val_loss, val_accuracy, epoch_time, self.best['model_path'])

            elif self.criterion == 'accuracy':
                if val_accuracy > self.best['val_acc']:
                    self.new_best(epoch, val_loss, val_accuracy, epoch_time)
                    self.reporter.add_best_model(self.name, epoch, val_loss, val_accuracy, epoch_time, self.best['model_path'])

            if self.use_tensorboard:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_accuracy, epoch)
                self.writer.add_scalar('Accuracy/val', val_accuracy, epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            self.epoch_times.append(time.time() - self.unix_0)
            total_time = sum(self.epoch_times)
            self.logger.log_epoch_times(epoch, self.epoch_times[-1], total_time)

        if self.use_tensorboard:
            self.writer.close()

        self.logger.log_finish()
            

    def _train_one_epoch(self, epoch):

        self.logger.log_info(f'EPOCH {epoch}')

        self.model.train()
        self.logger.log(f'Epoch {epoch} - lr: {self.optimizer.param_groups[0]["lr"]:.6f}')
                        
        mb_in_b = self.batch_size / self.memory_batch_size

        epoch_loss = 0
        epoch_corrects = 0    
        batch_loss = 0
        batch_corrects = 0
        batch_len = 0

        self.optimizer.zero_grad()
        for mbi, data in enumerate(self.train_loader, start=1):

            p1, p2, p3, choice = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device), data[3].to(self.device)

            # Forward pass
            _, pe1 = self.model(p1)
            _, pe2 = self.model(p2)
            _, pe3 = self.model(p3)

            # Loss
            mb_loss, mb_corrects = self.loss_fn(pe1, pe2, pe3, choice)
            batch_corrects += mb_corrects.item()
            batch_loss += mb_loss
            batch_len += choice.shape[0]

            # Backward pass at memory batch frequency
            if self.backward_freq == 'mbatch':
                mb_loss.backward()

            if mbi % mb_in_b == 0 or mbi == len(self.train_loader):

                batch = int(mbi // mb_in_b) if mbi % mb_in_b == 0 else int(mbi // mb_in_b) + 1
                batch_mean_loss = batch_loss.item() / batch_len
                batch_accuracy = batch_corrects / batch_len
                
                self.reporter.add_batch_indicator(self.name, epoch, batch, batch_mean_loss, batch_accuracy, time.time() - self.unix_0)
                self.logger.log(f'  Batch {batch} - Loss: {batch_mean_loss:.3f} - Accuracy: {batch_accuracy*100:.3f}% ({batch_len})')

                if self.use_tensorboard:
                    self.writer.add_scalar('Batch_loss/train', batch_mean_loss, batch + (epoch-1)*self.n_batches)
                    self.writer.add_scalar('Batch_Accuracy/train', batch_accuracy, batch + (epoch-1)*self.n_batches)


                # Epoch loss and corrects
                epoch_loss += batch_loss.item()
                epoch_corrects += batch_corrects
                
                # Backward pass at batch frequency
                if self.backward_freq == 'batch':
                    batch_loss.backward()

                # Optimize
                self.optimizer.step()

                # Zero gradients, batch loss and batch corrects
                self.optimizer.zero_grad()
                batch_loss = 0
                batch_corrects = 0
                batch_len = 0
        
        epoch_mean_loss = epoch_loss / len(self.train_loader.dataset)
        epoch_accuracy = epoch_corrects / len(self.train_loader.dataset)

        self.logger.log(f'Epoch {epoch} - Train Loss: {epoch_mean_loss:.3f} - Train Accuracy: {epoch_accuracy*100:.3f}%')
        
        return epoch_mean_loss, epoch_accuracy
        

    def _eval_one_epoch(self, epoch):
        
        self.model.eval()
        
        with torch.no_grad():
            epoch_loss = 0
            epoch_corrects = 0
            for _, data in enumerate(self.val_loader):
                p1, p2, p3, choice = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device), data[3].to(self.device)

                _, pe1 = self.model(p1)
                _, pe2 = self.model(p2)
                _, pe3 = self.model(p3)

                loss, corrects = self.loss_fn(pe1, pe2, pe3, choice) 
                epoch_loss += loss.item()
                epoch_corrects += corrects.item()

            epoch_mean_loss = epoch_loss / len(self.val_loader.dataset)
            epoch_accuracy = epoch_corrects / len(self.val_loader.dataset)
        
        self.logger.log(f'Epoch {epoch} - Validation Loss: {epoch_mean_loss:.3f} - Validation Accuracy: {epoch_accuracy*100:.3f}%')
        
        return epoch_mean_loss, epoch_accuracy
    
    def new_best(self, epoch, val_loss, val_acc, elapsed_time):
        self.best['epoch'] = epoch
        self.best['val_loss'] = val_loss
        self.best['val_acc'] = val_acc
        self.best['elapsed_time'] = elapsed_time
        self.best['model_path'] = f'{self.trainer_dir}/{self.name}.pth'
        self.save_model(self.best['model_path'])
        self.save_img2vec(f'{self.trainer_dir}/{self.name}_img2vec.pth')
        self.logger.log(f'New best model stored at epoch {epoch} - val_loss: {val_loss:.3f} - val_acc: {val_acc*100:.3f}% - elapsed_time: {elapsed_time:.3f} seconds')


    def save_model(self, path = None):
        torch.save(self.model.state_dict(), path)
        return path
    

    def save_img2vec(self, path = None):
        torch.save(self.model.img2vec.state_dict(), path)
        return path


    def trainer_to_dict(self):
        self.trainer_dict.update({'trainer_name': self.name,
                                  'model_name': self.model.name,
                                  'n_batches': self.n_batches,
                                  'n_params': self.model.num_params,
                                  'batch_size': self.batch_size,
                                  'memory_batch_size': self.memory_batch_size,
                                  'seed': self.seed,
                                  'device': self.device})
        return self.trainer_dict

    # Static methods
    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    @staticmethod
    def create_trainer_folder(trainer_name):
        if not os.path.exists(f'{PlaceEmbeddingTrainer.RESULTS_DIR}/{trainer_name}'):
            os.makedirs(f'{PlaceEmbeddingTrainer.RESULTS_DIR}/{trainer_name}')
        return f'{PlaceEmbeddingTrainer.RESULTS_DIR}/{trainer_name}'


    @staticmethod
    def get_device(gpu):
        if gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps"  if torch.backends.mps.is_available() else 
                                  "cpu")
        else:
            device = torch.device("cpu")
        return device
    

    @staticmethod
    def model_name(base_model, base_pretrained, pooling, 
                   encoder_layers, projection_layers, 
                   use_dropout, dropout_rate):

        name = f'model_{base_model}'
        name += 'p' if base_pretrained else 'r'
        name += f'_{pooling}_enc'

        if encoder_layers is not None:
            for layer in encoder_layers:
                name += f'_{layer}'
        else:
            name += 'None'

        name += f'_proj'
        if projection_layers is not None:
            for layer in projection_layers:
                name += f'_{layer}'
        else:
            name += 'None'

        name += f'_d{dropout_rate}' if use_dropout else '_nd'

        return name.replace(".","")
    

    @staticmethod
    def get_tensorboard_writer(dir_name):
        tensorboard_dir = f'{dir_name}/tensorboard'
        from tensorboardX import SummaryWriter
        return SummaryWriter(tensorboard_dir)


