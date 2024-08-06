from tools.trainer import PlaceEmbeddingTrainer
import pandas as pd
import ast
import pickle
import os

municipality = 'Rotterdam'
trainer_name = '20240803_161225'
epoch_model = 6
batch_size = 32

df = pd.read_csv(f'results/models.csv')
model_hparams = df[df['trainer_name'] == trainer_name].iloc[0]

base_model = model_hparams['base_model']
base_pretrained = model_hparams['base_pretrained']
pooling = model_hparams['pooling']
img2vec_encoder_layers = ast.literal_eval(model_hparams['img2vec_encoder_layers']) if model_hparams['img2vec_encoder_layers'] != 'not' else None
encoder_layers = ast.literal_eval(model_hparams['encoder_layers']) if model_hparams['encoder_layers'] != 'not' else None
projection_layers = ast.literal_eval(model_hparams['projection_layers']) if model_hparams['projection_layers'] != 'not' else None
act_f_encoder = model_hparams['act_f_encoder']
act_f_projection = model_hparams['act_f_projection']
L2_norm = model_hparams['L2_norm']

trainer = PlaceEmbeddingTrainer(name = None,
                                use_gpu = True,
                                use_tensorboard = False, 
                                verbose = True)

#trainer.set_place_data(data = f'municipalities/{municipality}_h3_10/spatial_units.pkl', batch_size = batch_size)

# Define the model
trainer.set_model_for_application(name = None,
                                    base_model = base_model,
                                    base_pretrained = base_pretrained,
                                    img2vec_encoder_layers = img2vec_encoder_layers,
                                    pooling = pooling,
                                    encoder_layers = encoder_layers,
                                    projection_layers = projection_layers,
                                    act_f_encoder = act_f_encoder,
                                    act_f_projection = act_f_projection, 
                                    L2_norm= L2_norm)

# eval on the city
modelpth = f'results/{trainer_name}/{trainer_name}_e{epoch_model}.pth'
responses = trainer.apply_model(modelpth)

# Create application folder if it does not exist
if not os.path.exists('applications'):
    os.makedirs('applications')

with open(f'applications/{modelpth}_{municipality}.pkl', 'wb') as f:
    pickle.dump(responses, f)
