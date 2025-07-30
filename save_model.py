import lpips
import torch
import os


MODEL_DIR = './models'
MODEL_PATH = os.path.join(MODEL_DIR, 'lpips_alexnet.pth')

loss_fn_alex = lpips.LPIPS(net='alex')
torch.save(loss_fn_alex.state_dict(), MODEL_PATH)