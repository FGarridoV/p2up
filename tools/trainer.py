import os
import time
import math
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from models.dataset import TripletDataset
from models.dataset import PlaceDataset
from models.dataset import TripletMinerSampler
from models.models import PlaceEmbedding, TripletPlaceEmbedding 
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
        self.triplet_miner = None

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


    def clean_gpu(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


    def set_data(self, triplets_path, img_transform, data_splits, batch_size, memory_batch_size, triplet_miner, data_seed = 21, num_workers = 0):

        if memory_batch_size is None:
            memory_batch_size = batch_size

        if batch_size % memory_batch_size != 0:
            raise ValueError('Memory batch size must be a divisor of batch size')

        train_transform = get_transform(img_transform)
        val_transform = get_transform('default')

        trainset = TripletDataset(data_splits['train'], data = triplets_path, transform = train_transform, train = True, seed = data_seed)
        valset = TripletDataset(data_splits['val'], data = triplets_path, transform = val_transform, train = False, seed = data_seed)
        
        self.batch_size = batch_size
        self.memory_batch_size = memory_batch_size
        self.triplet_miner = triplet_miner

        batch = batch_size if memory_batch_size is None else memory_batch_size
        if self.triplet_miner:
            self.train_loader = DataLoader(trainset, batch_size = batch, shuffle=False, num_workers=num_workers)
        else:
            self.train_loader = DataLoader(trainset, batch_size = batch, shuffle=True, num_workers=num_workers) 
        self.val_loader = DataLoader(valset, batch_size = batch, shuffle=False, num_workers=num_workers)

        if memory_batch_size == batch_size:
            self.n_batches = len(self.train_loader)
        else:
            self.n_batches = int(math.ceil(len(self.train_loader)/(batch_size / memory_batch_size)))

        self.trainer_dict.update({'img_transform': str(img_transform),
                                  'train_split': len(self.train_loader.dataset),
                                  'val_split': len(self.val_loader.dataset),
                                  'test_split': trainset.full_len - len(self.train_loader.dataset) - len(self.val_loader.dataset),
                                  'data_seed': data_seed})

        self.logger.log_data(trainset.full_len, len(trainset), len(valset), batch_size, memory_batch_size)
    

    def set_model(self, base_model, base_pretrained, img2vec_encoder_layers, pooling,  
                     encoder_layers, projection_layers, L2_norm, use_dropout, act_f_encoder, act_f_projection,
                     dropout_rate, pth_state = None, n_images = 5):
        
        model_name = PlaceEmbeddingTrainer.model_name(base_model, base_pretrained, img2vec_encoder_layers, pooling, 
                                                      encoder_layers, projection_layers, L2_norm,
                                                      use_dropout, dropout_rate)
        
        self.model = TripletPlaceEmbedding(model_name, n_images,
                                    base_model, base_pretrained, img2vec_encoder_layers, pooling,
                                    encoder_layers, projection_layers, L2_norm, 
                                    use_dropout, dropout_rate, act_f_encoder, act_f_projection)
        
        self.model = self.model.to(self.device)

        if pth_state is not None:
            self.model.load_state_dict(torch.load(pth_state, map_location=self.device))

        self.trainer_dict.update({'base_model': str(base_model),
                                  'pth_state': str(pth_state),
                                  'base_pretrained': str(base_pretrained),
                                  'img2vec_encoder_layers': str(img2vec_encoder_layers) if img2vec_encoder_layers is not None else 'not',
                                  'pooling': str(pooling),
                                  'encoder_layers': str(encoder_layers) if encoder_layers is not None else 'not',
                                  'projection_layers': str(projection_layers) if projection_layers is not None else 'not',
                                  'L2_norm': str(L2_norm),
                                  'use_dropout': str(use_dropout),
                                  'dropout_rate': float(dropout_rate),
                                  'act_f_encoder': str(act_f_encoder),
                                  'act_f_projection': str(act_f_projection)})

        self.logger.log_model(self.model.name, base_model, self.model.imgemb_size,
                              base_pretrained, img2vec_encoder_layers, 
                              pooling, encoder_layers, projection_layers, L2_norm, 
                              use_dropout, dropout_rate, act_f_encoder, act_f_projection,
                              pth_state, self.model.num_params)
        
    def set_model_for_application(self, name, base_model, base_pretrained, img2vec_encoder_layers, pooling, 
                                  encoder_layers, projection_layers, use_dropout, act_f_encoder, act_f_projection, L2_norm):
        
        n_images = 5
        dropout_rate = 0
        pth_state = None

        self.model = PlaceEmbedding(name = name, 
                                    n_images=n_images,
                                    base_model = base_model,
                                    base_pretrained = base_pretrained,
                                    img2vec_encoder_layers = img2vec_encoder_layers,
                                    pooling = pooling,
                                    encoder_layers = encoder_layers,
                                    projection_layers = projection_layers,
                                    L2_norm = L2_norm, 
                                    use_dropout = use_dropout,
                                    dropout_rate = dropout_rate,
                                    act_f_last_encoder = act_f_encoder,
                                    act_f_last_projection = act_f_projection)
        
        self.model = self.model.to(self.device)


        self.logger.log_model(name = self.model.name, 
                              base_model = base_model,
                              emb_size = self.model.imgemb_size,
                              pretrained = base_pretrained, 
                              img2vec_encoder_layers = img2vec_encoder_layers,
                              pooling = pooling,
                              encoder_layers = encoder_layers,
                              projection_layers = projection_layers,
                              L2_norm = L2_norm,
                              use_dropout = use_dropout,
                              dropout_rate = dropout_rate,
                              act_f_encoder = act_f_encoder, 
                              act_f_projection = act_f_projection,
                              pth_state = pth_state,
                              n_params = self.model.num_params)
    
    def set_loss(self, loss_kind, loss_dist, loss_margin, loss_swap, loss_reduction, count_corrects):

        if loss_kind == 'triplet':
            from models.models import TripletLoss
            self.loss_fn = TripletLoss(dist = loss_dist, margin = loss_margin, 
                                       swap = loss_swap, reduction = loss_reduction,
                                       count_corrects = count_corrects)
            self.loss_fn_no_red = TripletLoss(dist = loss_dist, margin = loss_margin,
                                                    swap = loss_swap, reduction = 'none',
                                                    count_corrects = False)
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

        # Create parameter groups
        params = [{'params': self.model.parameters(), 'lr': learning_rate}]

        if lr_embedder == 'no_train':
            for n, p in self.model.named_parameters():
                if 'img2vec' in n:
                    p.requires_grad = False
        
        if lr_embedder == 'now_train':
            for n, p in self.model.named_parameters():
                if 'img2vec' in n:
                    p.requires_grad = True
        
        else:
            img2vec_params = [p for n, p in self.model.named_parameters() if 'img2vec' in n]
            other_params = [p for n, p in self.model.named_parameters() if 'img2vec' not in n]
            params = [
                {'params': other_params, 'lr': learning_rate},
                {'params': img2vec_params, 'lr': lr_embedder}
            ]

        if optimizer == 'adam':
            from torch.optim import Adam
            self.optimizer = Adam(params, weight_decay=weight_decay)
        else:
            raise ValueError(f'Optimizer {optimizer} not implemented')
        
        if lr_scheduler_step is not None and lr_scheduler_gamma is not None:
            from torch.optim.lr_scheduler import StepLR
            self.lr_scheduler = StepLR(self.optimizer, step_size = lr_scheduler_step, gamma = lr_scheduler_gamma)

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
            t0 = time.time()
            train_loss, train_accuracy = self._train_one_epoch(epoch)
            val_loss, val_accuracy = self._eval_one_epoch(epoch)
            epoch_time = time.time() - t0
            elapsed_time = time.time() - self.unix_0
            self.reporter.add_epoch_indicator(self.name, epoch, train_loss, train_accuracy, val_loss, val_accuracy, epoch_time)

            if self.criterion == 'loss':
                if val_loss < self.best['val_loss']:
                    self.new_best(epoch, val_loss, val_accuracy, elapsed_time)
                    self.reporter.add_best_model(self.name, epoch, val_loss, val_accuracy, elapsed_time, self.best['model_path'])

            elif self.criterion == 'accuracy':
                if val_accuracy > self.best['val_acc']:
                    self.new_best(epoch, val_loss, val_accuracy, elapsed_time)
                    self.reporter.add_best_model(self.name, epoch, val_loss, val_accuracy, elapsed_time, self.best['model_path'])

            if self.use_tensorboard:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_accuracy, epoch)
                self.writer.add_scalar('Accuracy/val', val_accuracy, epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            self.epoch_times.append(epoch_time)
            total_time = time.time() - self.unix_0
            self.logger.log_epoch_times(epoch, epoch_time, total_time)

        if self.use_tensorboard:
            self.writer.close()

        self.logger.log_finish()

    
    def _mining_one_epoch(self):
        losses_and_indices = []
        for mbi, data in enumerate(self.train_loader):
            p1, p2, p3, choice = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device), data[3].to(self.device)
            with torch.no_grad():
                pe1, pe2, pe3 = self.model(p1, p2, p3)
                mb_loss, _ = self.loss_fn_no_red(pe1, pe2, pe3, choice)
                # Iterate over each item in the batch
                for i in range(p1.size(0)):
                    losses_and_indices.append((mb_loss[i].item(), mbi * self.train_loader.batch_size + i))

        # Sort indices based on loss values
        losses_and_indices.sort(key=lambda x: x[0])
        sorted_indices = [idx for _, idx in losses_and_indices]

        return sorted_indices


    def _train_one_epoch(self, epoch):

        self.logger.log_info(f'EPOCH {epoch}')

       
        self.logger.log(f'Epoch {epoch} - lr: {self.optimizer.param_groups[0]["lr"]:.6f}')
                        
        mb_in_b = self.batch_size / self.memory_batch_size

        epoch_loss = 0
        epoch_corrects = 0    
        batch_loss = 0
        batch_corrects = 0
        batch_len = 0

        if self.triplet_miner:
            sorted_indices = self._mining_one_epoch()
            sorted_sampler = TripletMinerSampler(self.train_loader.dataset, sorted_indices)
            self.train_loader = DataLoader(self.train_loader.dataset, batch_size=self.train_loader.batch_size, sampler=sorted_sampler)
        
        self.model.train()
        self.optimizer.zero_grad()
        for mbi, data in enumerate(self.train_loader, start=1):

            p1, p2, p3, choice = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device), data[3].to(self.device)

            # Forward pass
            pe1, pe2, pe3 = self.model(p1, p2, p3)

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

                pe1, pe2, pe3 = self.model(p1, p2, p3)

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
        self.best['model_path'] = f'{self.trainer_dir}/{self.name}_e{epoch}.pth'
        self.save_model(self.best['model_path'])
        self.save_img2vec(f'{self.trainer_dir}/{self.name}_img2vec_e{epoch}.pth')
        self.logger.log(f'New best model stored at epoch {epoch} - val_loss: {val_loss:.3f} - val_acc: {val_acc*100:.3f}% - elapsed_time: {elapsed_time/60:.3f} min')


    def set_place_data(self, data, batch_size, num_workers = 0):
        eval_transform = get_transform('default')
        
        evalset = PlaceDataset(data = data, root = '../', transform = eval_transform)
        self.eval_dataloader = DataLoader(evalset, batch_size = batch_size, shuffle=False, num_workers=num_workers) 

    
    def apply_model(self, modelpth):
        if modelpth is not None:
            self.model.load_state_dict(torch.load(modelpth))
        self.model = self.model.to(self.device)
        self.logger.log(f'Model loaded from {modelpth}')
        self.model.eval()
        responses = []
        with torch.no_grad():
            b = 0
            self.logger.log(f'Applying model to {len(self.eval_dataloader.dataset)} places')
            for _, data in enumerate(self.eval_dataloader):
                self.logger.log(f"Processing batch {b}")
                h3_code, place = data[0], data[1].to(self.device)
                e, pe = self.model(place)
                responses.append((h3_code, e.to('cpu').numpy(), pe.to('cpu').numpy()))
                b += 1
        return responses 


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
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


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
    def model_name(base_model, base_pretrained, img2vec_encoder_layers, pooling,
                   encoder_layers, projection_layers, L2_norm,
                   use_dropout, dropout_rate):

        name = f'model_{base_model}'
        name += 'p' if base_pretrained else 'r'

        if img2vec_encoder_layers is not None:
            for layer in img2vec_encoder_layers:
                name += f'_{layer}'
        else:
            name += 'None'

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

        name += 'L2' if L2_norm else ''

        name += f'_d{dropout_rate}' if use_dropout else '_nd'

        return name.replace(".","")
    


    

    @staticmethod
    def get_tensorboard_writer(dir_name):
        tensorboard_dir = f'{dir_name}/tensorboard'
        from tensorboardX import SummaryWriter
        return SummaryWriter(tensorboard_dir)
    
    


    #def mineHard(model, anchor, positive, negative, semiHard=False):
    #cnn = model
    #cnn.eval()
    #margin = 0.3
#
    #anchor, positive, negative = Variable(anchor).cuda(), Variable(positive).cuda(), Variable(negative).cuda()
    #output1, output2, output3 = cnn(anchor, positive, negative)
    #
    #d_pos = F.pairwise_distance(output1, output2)
    #d_neg = F.pairwise_distance(output1, output3)
    #if semiHard:
    #    pred1 = (d_pos - d_neg).cpu().data
    #    pred2 = (d_pos + margin - d_neg).cpu().data
    #    indices = numpy.logical_and((pred1 < 0), (pred2 > 0))
    #else:
    #    pred = (d_pos - d_neg).cpu().data
    #    indices = pred > 0
    #
    #if indices.sum() == 0:
    #    return None, None, None, False
#
    #x = torch.arange(0, d_pos.size()[0]).view(d_pos.size()[0], 1)
    #indices = x.type(torch.cuda.FloatTensor) * indices.type(torch.cuda.FloatTensor)
    #
    #nonzero_indices = torch.nonzero(indices)
    #indices = indices[nonzero_indices[:, 0], :].view(nonzero_indices.size()[0]).type(torch.cuda.LongTensor)
    #
    #anchor = torch.index_select(anchor.data, 0, indices)
    #positive = torch.index_select(positive.data, 0, indices)
    #negative = torch.index_select(negative.data, 0, indices)
#
    #return anchor, positive, negative, True


