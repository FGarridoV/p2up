from tools.trainer import PlaceEmbeddingTrainer
import pandas as pd
import ast

municipality = 'Rotterdam'

base_model = 'resnet18'
base_pretrained = True
pooling = 'mean'
encoder_layers = [512, 128]
projection_layers = [128, 64]
use_dropout = True
dropout_rate = 0.3

verbose = True              # Print the progress
gpu = True                  # Try to use GPU
use_tensorboard = False      # Use tensorboard for logging 

trainer = PlaceEmbeddingTrainer(name = None,
                                use_gpu = gpu,
                                use_tensorboard = use_tensorboard, 
                                verbose = verbose)

# Set the data
trainer.set_place_data(data = 'municipalities/Rotterdam_h3_10/spatial_units.pkl', batch_size = 64)

# Define the model
trainer.set_model(base_model = base_model, base_pretrained = base_pretrained, pooling = pooling, 
                encoder_layers = encoder_layers, projection_layers = projection_layers,
                use_dropout = use_dropout, dropout_rate = dropout_rate)

# eval on the city
modelpth = '20240328_220659_e4'
responses = trainer.eval_model(f'{modelpth}.pth')

#store the dict responses in pickle file
import pickle
with open(f'{modelpth}_{municipality}.pkl', 'wb') as f:
    pickle.dump(responses, f)
