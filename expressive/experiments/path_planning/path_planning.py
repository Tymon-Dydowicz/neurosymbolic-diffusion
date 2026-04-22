import glob
import multiprocessing
import os
import time

import tqdm

from expressive.args import PathPlanningArguments
from expressive.experiments.path_planning.absorbing_path import PathAbsorbing, create_nesy_diffusion
from expressive.methods.logger import (
    TestLog,
    TrainingLog,
    TestLogger,
    TrainLogger,
)
from expressive.util import EarlyStopping, get_device
from torch.utils.data import DataLoader
import torch
import wandb
import ray

from expressive.experiments.path_planning.data.dataloader import get_datasets


def get_cost_labels(costs: torch.Tensor, args: PathPlanningArguments) -> torch.Tensor:
    return torch.argmin(
        torch.abs(costs.unsqueeze(-1) - torch.tensor(args.costs, device=costs.device)), dim=-1
    )

def eval(
    loader: DataLoader,
    logger: TestLog,
    model: PathAbsorbing,
    device: torch.device,
    args: PathPlanningArguments,
):
    print("Number of eval batches:", len(loader))
    for i, batch in enumerate(loader):
        imgs, paths, costs = batch
        try:
            # Convert costs into label indices based on possible cost values
            cost_labels = get_cost_labels(costs, args)

            model.evaluate(
                imgs.to(device),
                paths.to(device),
                cost_labels.to(device),
                logger.log,
            )
        except ValueError as e:
            print(e)
            print("Warning: Batch skipped during eval")
        # if args.DEBUG:
        #     print("Leaving eval loop")
        #     break

    return logger.push(len(loader))


if __name__ == "__main__":
    args = PathPlanningArguments(explicit_bool=True).parse_args()
    device = get_device(args)

    if args.use_ray:
        ray.init(num_cpus=multiprocessing.cpu_count())

    run = wandb.init(
        project=f"nesy-diffusion-wc",
        # name=name,
        tags=[],
        config=args.__dict__,
        mode="offline" if not args.use_wandb else "online",
        id=args.wandb_resume_id,
        resume="must" if args.wandb_resume_id is not None else "never",
    )
    print(args)

    model = create_nesy_diffusion(args).to(device)
    if args.wandb_resume_id is not None:
        # model.load_state_dict(torch.load(args.load_model))
        # List all model files and find latest epoch
        model_files = glob.glob(f"{args.model_dir}/{args.wandb_resume_id}/model_*.pth")
        if not model_files:
            raise ValueError(f"No model files found in {args.model_dir}/{args.wandb_resume_id}")
            
        # Extract epoch numbers and find max
        epochs = [int(f.split('model_')[-1].replace('.pth','')) for f in model_files]
        latest_epoch = max(epochs)
        model.load_state_dict(torch.load(f"{args.model_dir}/{args.wandb_resume_id}/model_{latest_epoch}.pth"))
        print(f"Loaded model from {args.model_dir}/{args.wandb_resume_id}/model_{latest_epoch}.pth")

    train, val, test = get_datasets(args.grid_size)

    if args.DEBUG:
        print("DEBUG MODE")

    train_loader = DataLoader(train, args.batch_size, shuffle=True)
    val_loader = DataLoader(val, args.batch_size_test, shuffle=True)

    log_iterations = len(train_loader) // args.log_per_epoch
    if log_iterations == 0:
        log_iterations = 1

    train_logger = TrainLogger(log_iterations, TrainingLog, args)
    print("len(train_loader):", len(train_loader))
    print("Expected pred size:", len(train_loader) * args.batch_size * args.grid_size * args.grid_size)
    val_logger = TestLogger(TestLog, args, "val")

    if args.optimizer == "RAdam":
        optim = torch.optim.RAdam(
            model.parameters(), lr=args.lr,
        )
    elif args.optimizer == "Adam":
        optim = torch.optim.Adam(
            model.parameters(), lr=args.lr,
        )

    metric_key = f"val/{args.early_stopping_metric}"
    early_stopper = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        mode=args.early_stopping_mode,
    )

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        start_epoch_time = time.time()
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            optim.zero_grad()
            grid, label, costs = batch

            x = grid.to(device)
            label = label.to(device)
            loss = model.loss(x, label.long(), train_logger.log, get_cost_labels(costs, args).to(device))
            loss.backward()
            optim.step()

            train_logger.step()

        epoch_time = time.time() - start_epoch_time
        print(f"Epoch time: {epoch_time:.2f}s")

        if epoch % args.val_every_epochs == 0:
            print("----- VALIDATING -----")
            start_test_time = time.time()
            stats = eval(val_loader, val_logger, model, device, args)
            print(stats)
            should_stop = False
            if early_stopper.enabled:
                if metric_key in stats:
                    stop, improved = early_stopper.step(float(stats[metric_key]))
                    status = "improved" if improved else "not improved"
                    print(
                        f"Early stopping monitor {metric_key}={float(stats[metric_key]):.6f} ({status}); "
                        f"bad_epochs={early_stopper.bad_epochs}/{early_stopper.patience}"
                    )
                    should_stop = stop
                else:
                    print(f"Early stopping metric '{metric_key}' not found in validation stats; skipping check.")
            test_time = time.time() - start_test_time
            print(f"Test time: {test_time:.2f}s")

            if args.save_model:
                print(f"Saving model to {run.id}")
                os.makedirs(f"models/{run.id}", exist_ok=True)
                path = f"models/{run.id}/model_{epoch}.pth"
                torch.save(model.state_dict(), path)
                wandb.save(path)
            if should_stop:
                print(f"Early stopping triggered at epoch {epoch}.")
                break


    test_loader = DataLoader(test, args.batch_size_test, shuffle=True)
    test_logger = TestLogger(TestLog, args, "test")
    print("----- TESTING -----")
    print(eval(test_loader, test_logger, model, device, args))

