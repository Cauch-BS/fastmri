import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from torchdyn.core import NeuralODE
from torchcfm.models.unet import UNetModel

def test(args, model, data_loader):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    reconstructions = defaultdict(dict)
    inputs = defaultdict(dict)
    node = NeuralODE(model, solver = 'dopri5', sensitivity= 'adjoint', atol = 1e-4, rtol = 1e-4)
    
    with torch.no_grad():
        for (input, _, _, fnames, slices) in data_loader:
            input = input.cuda(non_blocking=True)
            traj = node.trajectory(
                input,
                t_span = torch.linspace(0, 1, 2, device = device),
            )
            output = traj[-1].view(input.shape)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                inputs[fnames[i]][int(slices[i])] = input[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in inputs:
        inputs[fname] = np.stack(
            [out for _, out in sorted(inputs[fname].items())]
        )
    return reconstructions, inputs


def forward(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    model = UNetModel(
        dim = (1, args.height, args.width),
        num_channels = args.num_channels,
        num_res_blocks = 1
    ).to(device=device)
    
    checkpoint = torch.load(args.exp_dir / 'model.pt', map_location='cpu')
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)
