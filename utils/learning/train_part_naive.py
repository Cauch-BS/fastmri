import torch
from utils.data.load_data import create_data_loaders
from utils.models.unet import UNetModel
from torchcfm.conditional_flow_matching import *


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
    total_loss = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            x1 = data[1].to(device)
            x0 = data[0].to(device)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            xt = xt.unsqueeze(1)
            vt = model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            total_loss += loss.item()
    return total_loss / len(val_loader)


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
    best_val_loss = float("inf")
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
        val_loss = validate(model, val_loader, FM, device)
        print(f"Validation Loss at Epoch {epoch:3d} = {val_loss:.4g}")

        # Save the Model if the Validation Loss is the Best
        if val_loss < best_val_loss:
            print(
                "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
            )
            best_val_loss = val_loss
            save_model(args, args.exp_dir, epoch + 1, model, optimizer)

            print(f"Best Validation Loss = {best_val_loss:.4g}")
