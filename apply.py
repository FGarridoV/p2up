from tools.trainer import PlaceEmbeddingTrainer
import pandas as pd
import ast
import pickle
import os

municipality = 'Rotterdam'
trainer = '20240731_012747'
epoch_model = 4
batch_size = 64

df = pd.read_csv(f'results/models.csv')
model_hparams = df[df['trainer_name'] == trainer].iloc[0]

base_model = model_hparams['base_model']
pooling = model_hparams['pooling']
img2vec_encoder_layers = ast.literal_eval(model_hparams['img2vec_encoder_layers']) if model_hparams['img2vec_encoder_layers'] != 'not' else None
encoder_layers = ast.literal_eval(model_hparams['encoder_layers']) if model_hparams['encoder_layers'] != 'not' else None
projection_layers = ast.literal_eval(model_hparams['projection_layers']) if model_hparams['projection_layers'] != 'not' else None

trainer = PlaceEmbeddingTrainer(name = None,
                                use_gpu = True,
                                use_tensorboard = False, 
                                verbose = True)

# Set the data
trainer.set_place_data(data = f'municipalities/{municipality}_h3_10/spatial_units.pkl', batch_size = batch_size)

# Define the model
trainer.set_model_for_application(base_model = base_model,
                                    img2vec_encoder_layers = img2vec_encoder_layers,
                                    pooling = pooling,
                                    encoder_layers = encoder_layers,
                                    projection_layers = projection_layers)

# eval on the city
modelpth = f'results/{trainer}_e{epoch_model}.pth'
responses = trainer.apply_model(modelpth, municipality)

# Create application folder if it does not exist
if not os.path.exists('applications'):
    os.makedirs('applications')

with open(f'applications/{modelpth}_{municipality}.pkl', 'wb') as f:
    pickle.dump(responses, f)
