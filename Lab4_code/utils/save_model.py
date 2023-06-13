import os
import torch

def save_model(model, save_path, model_name, epoch):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, model_name + '-{}.pth'.format(epoch))
    torch.save(model.state_dict(), filename)
