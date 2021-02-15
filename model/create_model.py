from model.net import Net, DuelNet, TimeNet

model_dict = {
    'TimeNet':TimeNet,
    'DuelNet':DuelNet,
    'Net':Net
}

def create_model(model_name):
    return model_dict[model_name]