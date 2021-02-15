from env.finance_env import FinanceEnv
from model.create_model import create_model
from model.util import calculate_index, EWMA, make_env_data
from model.train_env import Environment
import argparse    
import torch
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.yml')

args = parser.parse_args()    

with open(args.config) as f:
    config = yaml.safe_load(f.read())

df = pd.read_csv(config['data'], index_col=0)
df = df.head(200)
df = df[['Close']]

train_data, close_data, return_data = make_env_data(df)
model = create_model(config['model'])

env =Environment(FinanceEnv, train_data, close_data, return_data, model, **config['env'])
env.online_run()

