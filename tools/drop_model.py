import pandas as pd
import argparse 

def drop_model_folder(trainer_name):
    import shutil
    shutil.rmtree(f'results/{trainer_name}', ignore_errors=True)
    print(f'Folder {trainer_name} deleted.')

def drop_model_from_models_csv(trainer_name):
    df = pd.read_csv('results/models.csv')
    df = df[df['trainer_name'] != trainer_name]
    df.to_csv('results/models.csv', index=False)
    print(f'Model {trainer_name} dropped from models.csv.')

def drop_model_from_indicators_csv(trainer_name):
    df = pd.read_csv('results/indicators.csv')
    df = df[df['trainer_name'] != trainer_name]
    df.to_csv('results/indicators.csv', index=False)
    print(f'Model {trainer_name} dropped from indicators.csv.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Drop a model from the models.csv and indicators.csv files.')
    parser.add_argument('trainer_name', type=str, help='The name of the trainer to drop.')
    args = parser.parse_args()

    print(f'Dropping model {args.trainer_name}...')
    drop_model_folder(args.trainer_name)
    drop_model_from_models_csv(args.trainer_name)
    drop_model_from_indicators_csv(args.trainer_name)
    print('Done.')

## Usage example in the terminal:
# python tools/drop_model.py 'PlaceEmbeddingTrainer_2021-07-06_15-33-52'