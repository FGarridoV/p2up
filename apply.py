from tools.trainer import PlaceEmbeddingTrainer
import pandas as pd
import ast
import pickle
import os

municipality = 'AMS'
trainer_name = '20240807_092737'
epoch_model = 2
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
dropout = model_hparams['use_dropout']

trainer = PlaceEmbeddingTrainer(name = None,
                                use_gpu = True,
                                use_tensorboard = False, 
                                verbose = True)

trainer.set_ams_data(data = f'collector/img_index/panos_ams_dl_collection.geojson', batch_size = batch_size)

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
modelpth = f'results/{trainer_name}/{trainer_name}_e{epoch_model}.pth'
responses = trainer.apply_model(modelpth)

# Create application folder if it does not exist
if not os.path.exists('applications'):
    os.makedirs('applications')

with open(f'applications/{trainer_name}_e{epoch_model}_{municipality}.pkl', 'wb') as f:
    pickle.dump(responses, f)
