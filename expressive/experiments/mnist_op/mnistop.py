from __future__ import annotations
import json
import math
import os
import time

from tqdm.asyncio import tqdm

from expressive.util import EarlyStopping, get_device
from torch.utils.data import DataLoader
import torch
import wandb
from PIL import Image

from expressive.experiments.mnist_op.absorbing_mnist import (
    MNISTAddProblem,
    create_mnistadd,
    vector_to_base10,
)
from expressive.args import MNISTAbsorbingArguments
import sys

for i, a in enumerate(sys.argv):
    if a.startswith("--allowed-digits"):
        if a == "--allowed-digits":
            sys.argv[i] = "--allowed_digits"
        elif a.startswith("--allowed-digits="):
            sys.argv[i] = a.replace("--allowed-digits=", "--allowed_digits=")
from expressive.experiments.mnist_op.data import (
    create_nary_multidigit_operation,
    get_mnist_op_dataloaders,
)

from expressive.methods.logger import (
    TestLog,
    TrainingLog,
    TrainLogger,
    TestLogger,
)

SWEEP = True


def _digits_to_number(digits: list[int]) -> int:
    out = 0
    for d in digits:
        out = out * 10 + int(d)
    return out


def _save_mnist_image(img_chw: torch.Tensor, out_path: str) -> None:
    img = (img_chw.detach().cpu() * 0.3081 + 0.1307).clamp(0.0, 1.0)
    arr = (img.squeeze(0).numpy() * 255.0).astype("uint8")
    Image.fromarray(arr, mode="L").save(out_path)


def export_sanity_check_samples(
    split_name: str,
    loader: DataLoader,
    args: MNISTAbsorbingArguments,
    out_root: str,
    n_operands: int,
    arity: int,
    max_samples: int,
) -> int:
    if max_samples <= 0:
        return 0

    split_dir = os.path.join(out_root, split_name)
    os.makedirs(split_dir, exist_ok=True)

    exported = 0
    for batch_idx, batch in enumerate(loader):
        mn_digits = batch[:n_operands]
        label_digits = batch[n_operands:-1]
        label_target = batch[-1]

        batch_size = label_target.shape[0]
        for bi in range(batch_size):
            if exported >= max_samples:
                return exported

            sample_dir = os.path.join(split_dir, f"sample_{exported:04d}")
            os.makedirs(sample_dir, exist_ok=True)

            operand_labels = [int(label_digits[j][bi].item()) for j in range(n_operands)]
            grouped = []
            group_len = n_operands // arity
            for g in range(arity):
                start = g * group_len
                end = (g + 1) * group_len
                grouped.append(_digits_to_number(operand_labels[start:end]))

            target_int = int(label_target[bi].item())
            target_digits = vector_to_base10(
                label_target[bi : bi + 1].to(torch.long), args.N + 1
            )[0].detach().cpu().tolist()

            for j in range(n_operands):
                img = mn_digits[j][bi]
                d = operand_labels[j]
                _save_mnist_image(
                    img,
                    os.path.join(sample_dir, f"operand_{j:02d}_digit_{d}.png"),
                )

            info = {
                "split": split_name,
                "batch_index": batch_idx,
                "batch_item_index": bi,
                "n_operands": n_operands,
                "arity": arity,
                "digits_per_number": args.N,
                "operand_digit_labels": operand_labels,
                "grouped_numbers": grouped,
                "target_integer": target_int,
                "target_digits_passed_to_model": target_digits,
                "operation": args.op,
                "allowed_digits": args.allowed_digits,
                "stratified": args.stratified,
            }
            with open(os.path.join(sample_dir, "info.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)

            exported += 1

    return exported


def test(
    val_loader: DataLoader,
    test_logger: TestLog,
    model: MNISTAddProblem,
    device: torch.device,
):
    for i, batch in enumerate(val_loader):
        mn_digits, label_digits, label = (
            batch[: 2 * args.N],
            batch[2 * args.N : -1],
            batch[-1],
        )
        x = torch.cat(mn_digits, dim=1)
        model.evaluate(
            x.to(device),
            vector_to_base10(label.to(device), args.N + 1),
            torch.stack(label_digits, dim=-1).to(device),
            test_logger.log,
        )
        if args.DEBUG:
            break
    return test_logger.push(len(val_loader))


args = MNISTAbsorbingArguments(explicit_bool=True).parse_args()


def main():
    # name = "addition_" + str(args.N)
    run = wandb.init(
        project=f"nesy-diffusion",
        # name=name,
        tags=[],
        config=args.__dict__,
        mode="disabled" if not args.use_wandb else "online",
    )

    device = get_device(args)

    model = create_mnistadd(args).to(device)
    arity = 2
    digits_per_number = args.N
    n_operands = arity * digits_per_number

    bin_op = sum if args.op == "sum" else math.prod if args.op == "product" else None
    op = create_nary_multidigit_operation(arity, bin_op)

    if args.DEBUG:
        # Enable anomaly detection in PyTorch for debugging NaNs
        torch.autograd.set_detect_anomaly(True)

        # Add hooks to check for NaNs in gradients
        def hook(grad):
            if torch.isnan(grad).any():
                print("NaN gradient detected!")
                raise RuntimeError("NaN gradient detected")

        for p in model.parameters():
            if p.requires_grad:
                p.register_hook(hook)

    train_size = args.train_size if args.train_size is not None else (60000 if args.test else 50000)
    val_size = 0 if args.test else 10000
    train_loader, val_loader, test_loader = get_mnist_op_dataloaders(
        count_train=int(train_size / n_operands),
        count_val=int(val_size / n_operands),
        count_test=int(10000 / n_operands),
        batch_size=args.batch_size,
        n_operands=n_operands,
        op=op,
        # This shuffle is very weird...
        shuffle=True,
        allowed_digits=args.allowed_digits,
        stratified=args.stratified,
    )

    sanity_root = os.path.join(args.sanity_check_dir, run.id)
    if args.sanity_check_samples > 0:
        print("----- EXPORTING SANITY CHECK SAMPLES -----")
        print(f"Saving sanity-check samples to: {sanity_root}")
        n_train = export_sanity_check_samples(
            "train",
            train_loader,
            args,
            sanity_root,
            n_operands,
            arity,
            args.sanity_check_samples,
        )
        n_val = export_sanity_check_samples(
            "val",
            val_loader,
            args,
            sanity_root,
            n_operands,
            arity,
            args.sanity_check_samples,
        )
        n_test = export_sanity_check_samples(
            "test",
            test_loader,
            args,
            sanity_root,
            n_operands,
            arity,
            args.sanity_check_samples,
        )
        print(
            f"Sanity-check export complete: train={n_train}, val={n_val}, test={n_test} samples "
            f"(per split requested={args.sanity_check_samples})"
        )

    log_iterations = len(train_loader) // args.log_per_epoch

    train_logger = TrainLogger(log_iterations, TrainingLog, args)
    val_logger = TestLogger(TestLog, args, "val")
    print("Length of val loader:", len(val_loader))

    optim = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0
    )
    metric_key = f"val/{args.early_stopping_metric}"
    early_stopper = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        mode=args.early_stopping_mode,
    )
    if early_stopper.enabled and args.test:
        print("Early stopping is enabled but '--test' disables validation; early stopping will be ignored.")

    os.makedirs(f"models/{run.id}", exist_ok=True)
    for epoch in range(args.epochs):
        print("----------------------------------------")
        print("NEW EPOCH", epoch, "/", args.epochs)

        start_epoch_time = time.time()

        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            optim.zero_grad()
            mn_digits, label, w_labels = batch[: 2 * args.N], batch[-1], batch[2 * args.N : -1]

            x = torch.cat(mn_digits, dim=1).to(device)
            w_labels = torch.stack(w_labels, dim=1).to(device)
            label = vector_to_base10(label.to(device), args.N + 1)
            loss = model.loss(x, label, train_logger.log, w_labels)

            loss.backward()
            optim.step()

            train_logger.step()

            if args.DEBUG:
                break

        end_epoch_time = time.time()

        epoch_time = end_epoch_time - start_epoch_time
        print(f"Epoch time: {epoch_time} seconds")

        # If val not available, don't test during training
        should_stop = False
        if epoch % args.test_every_epochs == 0:
            if not args.test:
                print("----- VALIDATING -----")
                stats = test(val_loader, val_logger, model, device)
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
                test_time = time.time() - end_epoch_time
                print(f"Test time: {test_time} seconds")
            
            print(f"Saving model to {run.id}")
            wandb.save(f"model_{epoch}_{run.id}.pth")
            torch.save(model.state_dict(), f"models/{run.id}/model_{epoch}.pth") 
            if should_stop:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
            

    print("----- TESTING -----")
    test_logger = TestLogger(TestLog, args, "test")
    test(test_loader, test_logger, model, device)
    print(f"Saving model to {run.id}")
    wandb.save(f"model_{epoch}_{run.id}.pth")
    torch.save(model.state_dict(), f"models/{run.id}/model_{epoch}.pth") 


if __name__ == "__main__":
    main()
