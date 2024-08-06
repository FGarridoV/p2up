from tools.trainer import PlaceEmbeddingTrainer
import pandas as pd
import ast
import pickle
import os

municipality = 'Rotterdam'
trainer_name = 'benchmark'
batch_size = 128

base_model = 'resnet34'
base_pretrained = True
pooling = 'max'
img2vec_encoder_layers = None
encoder_layers = None
projection_layers = [2]
act_f_encoder = True
act_f_projection = True
L2_norm = False
dropout = False

trainer = PlaceEmbeddingTrainer(name = None,
                                use_gpu = True,
                                use_tensorboard = False, 
                                verbose = True)

trainer.set_place_data(data = f'municipalities/{municipality}_h3_10/spatial_units.pkl', batch_size = batch_size)

# Define the model
trainer.set_model_for_application(name = None,
                                    base_model = base_model,
                                    base_pretrained = base_pretrained,
                                    img2vec_encoder_layers = img2vec_encoder_layers,
                                    pooling = pooling,
                                    encoder_layers = encoder_layers,
                                    projection_layers = projection_layers,
                                    use_dropout = dropout,
                                    act_f_encoder = act_f_encoder,
                                    act_f_projection = act_f_projection, 
                                    L2_norm= L2_norm)

# eval on the city
modelpth = None
responses = trainer.apply_model(modelpth)

# Create application folder if it does not exist
if not os.path.exists('applications'):
    os.makedirs('applications')

with open(f'applications/benchmark_{base_model}.pkl', 'wb') as f:
    pickle.dump(responses, f)
