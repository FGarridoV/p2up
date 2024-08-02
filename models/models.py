import torch
from torch import nn
from torch.nn import functional as F
from models.vision_models import ImageEmbeddingModel

class MLP(nn.Module):
    def __init__(self, input_size, encoder_layers, use_dropout, dropout_rate):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(encoder_layers)):
            if i == 0:
                layers.append(nn.Linear(input_size, encoder_layers[i]))
            else:
                layers.append(nn.Linear(encoder_layers[i-1], encoder_layers[i]))
            
            if i < len(encoder_layers) - 1:
                layers.append(nn.LeakyReLU())
                if use_dropout:
                    layers.append(nn.Dropout(dropout_rate))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

class TripletLoss(nn.Module):
    def __init__(self, dist='euclidean', margin=1.0, p=2.0, swap = True, reduction = 'sum', count_corrects = True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.swap = swap
        self.reduction = reduction
        self.count_corrects = count_corrects

        if dist == 'euclidean':
            self.dist = lambda x1, x2: F.pairwise_distance(x1, x2, p=p)
        if dist == 'cosine':
            self.dist = lambda x1, x2: 1.0 - F.cosine_similarity(x1, x2)

        self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=self.dist, margin=margin, swap = swap, reduction=reduction)


    def forward(self, pe1, pe2, pe3, choice):
        pes = [pe1, pe2, pe3]

        anchor = torch.stack([pes[(c + 2) % 3][i] for i, c in enumerate(choice)])
        positive = torch.stack([pes[(c + 1) % 3][i] for i, c in enumerate(choice)])
        negative = torch.stack([pes[(c + 0) % 3][i] for i, c in enumerate(choice)])

        loss = self.triplet_loss(anchor, positive, negative)

        if self.count_corrects:
            dist_pos = self.dist(anchor, positive)
            dist_neg = self.dist(anchor, negative)
            if self.swap:
                dist_neg = torch.min(dist_neg, self.dist(positive, negative))
            corrects = (dist_pos < dist_neg).sum()
        else:
            corrects = None

        return loss, corrects
    

class TripletPlaceEmbedding(nn.Module):
    def __init__(self, name,
                       n_images = 5, 
                       base_model = 'resnet18',
                       base_pretrained = True,
                       img2vec_encoder_layers = [512], # None
                       pooling = 'max', # ['mean', 'std', 'max', 'min', 'median',  OR 'concat']
                       encoder_layers = [512], # None
                       projection_layers = [32], # None
                       L2_norm = True,
                       use_dropout = False,
                       dropout_rate = 0.3):
        super(TripletPlaceEmbedding, self).__init__()

        # General parameters
        self.name = name
        self.n_images = n_images
        self.L2_norm = L2_norm

        # Model parts
        self.img2vec = None
        self.img2vec_encoder = None
        self.encoder = None
        self.projection = None

        # Image embedding model
        self.img2vec, self.imgemb_size = ImageEmbeddingModel.get_model(base_model, base_pretrained)

        # Image2Vec encoder
        if img2vec_encoder_layers is not None:
            self.img2vec_encoder = MLP(self.imgemb_size, img2vec_encoder_layers, use_dropout, dropout_rate)
            self.vec_size = img2vec_encoder_layers[-1]
        else:
            self.vec_size = self.imgemb_size
        
        # Pooling
        self.pooling = pooling
        self.pooled_embedding_size = self.vec_size * n_images if pooling == 'concat' else self.vec_size * (pooling.count('-') + 1)

        # Encoder head
        if encoder_layers is not None:
            self.encoder = MLP(self.pooled_embedding_size, encoder_layers, use_dropout, dropout_rate)

        # Projection head
        if projection_layers is not None:
            if self.encoder is None:
                self.projection = MLP(self.pooled_embedding_size, projection_layers, use_dropout, dropout_rate)
            else:
                self.projection = MLP(encoder_layers[-1], projection_layers, use_dropout, dropout_rate)
        
        if self.img2vec is not None:
            self.img2vec.name = 'img2vec'
        if self.img2vec_encoder is not None:
            self.img2vec_encoder.name = 'img2vec_encoder'
        if self.encoder is not None:
            self.encoder.name = 'encoder'
        if self.projection is not None:
            self.projection.name = 'projection'

    
    def forward(self, x1, x2, x3, train=True):
        tr_batch_size = x1.shape[0]
        x = torch.cat([x1, x2, x3], dim=0)

        batch_size, n_images, c, h, w = x.shape
        x = x.view(-1, c, h, w)  # Reshape to (batch_size * n_images, c, h, w)

        # Get embeddings for all images
        x = self.img2vec(x)  

        # Img2Vec encoder
        if self.img2vec_encoder is not None:
            x = x.view(x.size(0), -1) # Reshape to (batch_size * n_images, embedding_size)
            x = self.img2vec_encoder(x)

        # Reshape back to (batch_size, n_images, embedding_size)
        x = x.view(batch_size, n_images, -1)  

        # Pool the embeddings
        if self.pooling == 'concat':
            x = x.view(batch_size, -1)
        else:
            methods = self.pooling.split('-')
            xs = []
            for m in methods:
                if m == 'mean':
                    xs.append(x.mean(dim=1))
                elif m == 'std':
                    xs.append(x.std(dim=1))
                elif m == 'max':
                    xs.append(x.max(dim=1).values)
                elif m == 'min':
                    xs.append(x.min(dim=1).values)
                elif m == 'median':
                    xs.append(x.median(dim=1).values)
            x = torch.cat(xs, dim=1)

        if train:
            # Encoder
            if self.encoder is not None:
                x = self.encoder(x)

            # Projection
            if self.projection is not None:
                x = self.projection(x)
            
            if self.L2_norm:
                x = F.normalize(x, p=2, dim=1)
            
            x1, x2, x3 = x[:tr_batch_size], x[tr_batch_size:2*tr_batch_size], x[2*tr_batch_size:]
            return x1, x2, x3
        
        else:
            # Encoder
            if self.encoder is not None:
                x = self.encoder(x)

            # Projection
            if self.projection is not None:
                px = self.projection(x)
                if self.L2_norm:
                    px = F.normalize(px, p=2, dim=1)
            return x, px

            #if not self.training:
            #    x1, x2, x3 = x[:tr_batch_size], x[tr_batch_size:2*tr_batch_size], x[2*tr_batch_size:]
            #    proj_x1, proj_x2, proj_x3 = proj_x[:tr_batch_size], proj_x[tr_batch_size:2*tr_batch_size], proj_x[2*tr_batch_size:]

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PlaceEmbeddingBase(nn.Module):
    def __init__(self, name,
                       n_images = 5, 
                       base_model = 'resnet18',
                       base_pretrained = True,
                       pooling = 'max'):
        super(PlaceEmbeddingBase, self).__init__()

        # General parameters
        self.name = name
        self.n_images = n_images

        # Model parts
        self.img2vec = None

        # Image embedding model
        self.img2vec, self.vec_size = ImageEmbeddingModel.get_model(base_model, base_pretrained)
        
        # Pooling
        self.pooling = pooling
        self.pooled_embedding_size = self.vec_size * n_images if pooling == 'concat' else self.vec_size * (pooling.count('-') + 1)


    def forward(self, x):
        batch_size, n_images, c, h, w = x.shape
        x = x.view(-1, c, h, w)

        # Get embeddings for all images
        x = self.img2vec(x)

        # Reshape back to (batch_size, n_images, embedding_size)
        x = x.view(batch_size, n_images, -1)

        # Pool the embeddings
        if self.pooling == 'concat':
            x = x.view(batch_size, -1)
        else:
            methods = self.pooling.split('-')
            xs = []
            for m in methods:
                if m == 'mean':
                    xs.append(x.mean(dim=1))
                elif m == 'std':
                    xs.append(x.std(dim=1))
                elif m == 'max':
                    xs.append(x.max(dim=1).values)
                elif m == 'min':
                    xs.append(x.min(dim=1).values)
                elif m == 'median':
                    xs.append(x.median(dim=1).values)
            x = torch.cat(xs, dim=1)

        return x
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PlaceEmbedding(nn.Module):
    def __init__(self, name,
                       n_images = 5, 
                       base_model = 'resnet18',
                       base_pretrained = True,
                       img2vec_encoder_layers = [512], # None
                       pooling = 'max', # ['mean', 'std', 'max', 'min', 'median',  OR 'concat']
                       encoder_layers = [512], # None
                       projection_layers = [32], # None
                        L2_norm = True,
                       use_dropout = False,
                       dropout_rate = 0.3):
        super(PlaceEmbedding, self).__init__()

        # General parameters
        self.name = name
        self.n_images = n_images
        self.L2_norm = L2_norm

        # Model parts
        self.img2vec = None
        self.img2vec_encoder = None
        self.encoder = None
        self.projection = None

        # Image embedding model
        self.img2vec, self.imgemb_size = ImageEmbeddingModel.get_model(base_model, base_pretrained)

        # Image2Vec encoder
        if img2vec_encoder_layers is not None:
            self.img2vec_encoder = MLP(self.imgemb_size, img2vec_encoder_layers, use_dropout, dropout_rate)
            self.vec_size = img2vec_encoder_layers[-1]
        else:
            self.vec_size = self.imgemb_size
        
        # Pooling
        self.pooling = pooling
        self.pooled_embedding_size = self.vec_size * n_images if pooling == 'concat' else self.vec_size * (pooling.count('-') + 1)

        # Encoder head
        if encoder_layers is not None:
            self.encoder = MLP(self.pooled_embedding_size, encoder_layers, use_dropout, dropout_rate)

        # Projection head
        if projection_layers is not None:
            if self.encoder is None:
                self.projection = MLP(self.pooled_embedding_size, projection_layers, use_dropout, dropout_rate)
            else:
                self.projection = MLP(encoder_layers[-1], projection_layers, use_dropout, dropout_rate)
        
        if self.img2vec is not None:
            self.img2vec.name = 'img2vec'
        if self.img2vec_encoder is not None:
            self.img2vec_encoder.name = 'img2vec_encoder'
        if self.encoder is not None:
            self.encoder.name = 'encoder'
        if self.projection is not None:
            self.projection.name = 'projection'


    def forward(self, x):
        batch_size, n_images, c, h, w = x.shape
        x = x.view(-1, c, h, w)  # Reshape to (batch_size * n_images, c, h, w)

        # Get embeddings for all images
        x = self.img2vec(x)  

        # Img2Vec encoder
        if self.img2vec_encoder is not None:
            x = x.view(x.size(0), -1) # Reshape to (batch_size * n_images, embedding_size)
            x = self.img2vec_encoder(x)

        # Reshape back to (batch_size, n_images, embedding_size)
        x = x.view(batch_size, n_images, -1)  

        # Pool the embeddings
        if self.pooling == 'concat':
            x = x.view(batch_size, -1)
        else:
            methods = self.pooling.split('-')
            xs = []
            for m in methods:
                if m == 'mean':
                    xs.append(x.mean(dim=1))
                elif m == 'std':
                    xs.append(x.std(dim=1))
                elif m == 'max':
                    xs.append(x.max(dim=1).values)
                elif m == 'min':
                    xs.append(x.min(dim=1).values)
                elif m == 'median':
                    xs.append(x.median(dim=1).values)
            x = torch.cat(xs, dim=1)
        
        # Encoder
        if self.encoder is not None:
            x = self.encoder(x)

        # Projection
        if self.projection is not None:
            px = self.projection(x)
            if self.L2_norm:
                px = F.normalize(px, p=2, dim=1)
        return x, px
    

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


