import shutil
import numpy as np
import torch
import torch.nn as nn
import time

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from torchcfm.models.unet import UNetModel
from torchcfm.conditional_flow_matching import *

def save_model(args, exp_dir, epoch, model, optimizer):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
      
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())
    
    model = UNetModel(
        dim = (args.in_chan, args.height, args.width),
        num_channels = args.num_channels,
        num_res_blocks = args.num_res_blocks
    ).to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    start_epoch = 0
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma = args.sigma)

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            x1 = data[1].to(device)
            x0 = data[0].to(device)
            t, xt, ut = FM.sample_location_and_conditional_flow(
                x0, x1
            )
            xt = xt.unsqueeze(1)
            #print(xt.shape)
            vt = model(t, xt)
            loss = torch.mean((vt - ut)**2)
            loss.backward()
            optimizer.step()
            if i % args.report_interval == 0:
                print(
                    f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                    f'Iter = [{i:4d}/{len(train_loader):4d}] '
                    f'Loss = {loss.item():.4g} '
                )

        save_model(args, args.exp_dir, epoch + 1, model, optimizer)