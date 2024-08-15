from collections import defaultdict
from time import perf_counter

import torch
import tqdm
import numpy as np
from torchdyn.core import NeuralODE
from torchcfm.conditional_flow_matching import *

from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.models.unet import UNetModelWrapper as UNetModel


def save_model(args, exp_dir, epoch, model, optimizer):
    torch.save(
        {
            "epoch": epoch,
            "args": args,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "exp_dir": exp_dir,
        },
        f=exp_dir / "model.pt",
    )


def validate(model, val_loader, FM, device):
    model.eval()
    node = NeuralODE(
        model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    inputs = defaultdict(dict)
    start = perf_counter()

    with torch.no_grad():
        for _, data in tqdm.tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc="Validating with SSIM:",
            bar_format="{l_bar}{bar}{r_bar}",
            ascii=">=",
            ncols=120,
        ):
            input, target, _, fnames, slices = data
            input = input.cuda(non_blocking=True)
            input_unsqueezed = input.unsqueeze(1)
            traj = node.trajectory(
                input_unsqueezed,
                t_span=torch.linspace(0, 1, 2, device=device),
            )
            output_raw = traj[-1].squeeze(1)
            output = output_raw.view(input.shape)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()
                inputs[fnames[i]][int(slices[i])] = input[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )

    for fname in targets:
        targets[fname] = np.stack([out for _, out in sorted(targets[fname].items())])

    for fname in inputs:
        inputs[fname] = np.stack([out for _, out in sorted(inputs[fname].items())])

    metric_loss = sum(
        [ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions]
    )

    num_subjects = len(reconstructions)

    return (
        metric_loss,
        num_subjects,
        reconstructions,
        targets,
        inputs,
        perf_counter() - start,
    )


def train(args):
    device = torch.device(
        f"cuda:{args.GPU_NUM}" if torch.cuda.is_available() else "cpu"
    )
    torch.cuda.set_device(device)
    print("Current cuda device: ", torch.cuda.current_device())

    model = UNetModel(
        dim=(args.in_chan, args.height, args.width),
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
    ).to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    start_epoch = 0
    train_loader = create_data_loaders(
        data_path=args.data_path_train, args=args, shuffle=True
    )
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=args.sigma)

    # initialize the best validation loss
    best_val_loss = 0
    val_loss_log = np.empty((0, 2))
    # initialize val_loader
    val_loader = create_data_loaders(
        data_path=args.data_path_val, args=args, shuffle=False
    )

    for epoch in range(start_epoch, args.num_epochs):
        print(f"Epoch #{epoch:2d} ............... {args.net_name} ...............")
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            x1 = data[1].to(device)
            x0 = data[0].to(device)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            xt = xt.unsqueeze(1)
            # print(xt.shape)
            vt = model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optimizer.step()
            if i % args.report_interval == 0:
                print(
                    f"Epoch = [{epoch:3d}/{args.num_epochs:3d}] "
                    f"Iter = [{i:4d}/{len(train_loader):4d}] "
                    f"Loss = {loss.item():.4g} "
                )

        # Validate the Model
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(
            model, val_loader, FM, device
        )

        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = args.val_loss_dir / "val_loss_log"
        np.save(file_path, val_loss_log)
        print(f"Loss file saved at {file_path}")

        val_loss = 1 - val_loss / num_subjects

        print(f"SSIM at Epoch {epoch + 1:3d} = {val_loss:.4g}")

        # Save the Model if the Validation Loss is the Best
        if val_loss > best_val_loss:
            print(
                "@@@@@@@@@@@@@@@@@@@@@@@@@@@@ New Record !! @@@@@@@@@@@@@@@@@@@@@@@@@@@@"
            )
            best_val_loss = val_loss
            save_model(args, args.exp_dir, epoch + 1, model, optimizer)
            save_reconstructions(
                reconstructions, args.val_dir, targets=targets, inputs=inputs
            )

            print(f"Best SSIM = {best_val_loss:.4g}")
            print(f"Forward Time = {val_time:.4f}s")
