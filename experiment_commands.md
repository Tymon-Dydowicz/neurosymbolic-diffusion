### Commands
MNIST Add N=4:
```
uv run expressive/experiments/mnist_op/mnistop.py
```

MNIST Add N=4 with a smaller training set:
```
uv run expressive/experiments/mnist_op/mnistop.py --train_size 20000
uv run expressive/experiments/mnist_op/mnistop.py --allowed_digits 1 2 3 --stratifed True
```

MNIST Add N=15:
```
uv run expressive/experiments/mnist_op/mnistop.py --N 15 --epochs 1000
uv run expressive/experiments/mnist_op/mnistop.py --N 15 --epochs 1000 --allowed_digits 1 2 3 --stratifed True
```

Path Planning 12x12:
```
./expressive/experiments/path_planning/download.sh # If data is not yet downloaded
uv run expressive/experiments/path_planning/path_planning.py
```

Path Planning 30x30:
```
./expressive/experiments/path_planning/download.sh  # If data is not yet downloaded
uv run expressive/experiments/path_planning/data/merge.py # Data postprocessing step required for N=30
uv run expressive/experiments/path_planning/path_planning.py --grid_size 30 --loss_S 2 --variational_K 2 --test_K 2
```

### Commands With Early Stopping
MNIST Add N=4:
```
uv run expressive/experiments/mnist_op/mnistop.py --early_stopping_patience 10 --early_stopping_min_delta 0.001 --early_stopping_metric w_acc_avg --early_stopping_mode max
```

MNIST Add N=15:
```
uv run expressive/experiments/mnist_op/mnistop.py --N 15 --epochs 1000 --early_stopping_patience 15 --early_stopping_min_delta 0.001 --early_stopping_metric w_acc_avg --early_stopping_mode max
```

Path Planning 12x12:
```
./expressive/experiments/path_planning/download.sh # If data is not yet downloaded
uv run expressive/experiments/path_planning/path_planning.py --early_stopping_patience 5 --early_stopping_min_delta 0.001 --early_stopping_metric w_acc_avg --early_stopping_mode max
```

Path Planning 30x30:
```
./expressive/experiments/path_planning/download.sh  # If data is not yet downloaded
uv run expressive/experiments/path_planning/data/merge.py # Data postprocessing step required for N=30
uv run expressive/experiments/path_planning/path_planning.py --grid_size 30 --loss_S 2 --variational_K 2 --test_K 2 --early_stopping_patience 5 --early_stopping_min_delta 0.001 --early_stopping_metric w_acc_avg --early_stopping_mode max
```