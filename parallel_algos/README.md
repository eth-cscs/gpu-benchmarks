# Parallel algorithms benchmarks

Dependencies: standard CUDA or HIP installation is enough

## Radix sort

### Running on AMD GPUs

Building with CMake:
```bash
cmake -DCMAKE_HIP_ARCHITECUTRES=gfx90a;gfx942 <GIT-SOURCE-DIR>/parallel-algos
make
```
Running the test on `2^n` key-value pairs:
```bash
./radix-sort n
```

Example output on MI250X:
```bash
LUMI$ srun -pdev-g --gres=gpu:1 -Aproject_465001618 -N1 --ntasks-per-node=1 -c7 -t 15:00 ./radix-sort 27
radix sort time for 134217728 key-value pairs: 0.0484017 s, bandwidth: 66552 MiB/s
```

Metric to plot is bandwidth and/or keys/second (134217728 / 0.0484017 = 2.77 Gkeys/s)

### building with CUDA

```bash
cmake -DCMAKE_CUDA_ARCHITECUTRES=90 <GIT-SOURCE-DIR>/parallel-algos
```

Example output:
```bash
ALPS-GH200$ ./radix-sort 27
radix sort time for 134217728 key-value pairs: 0.0247207 s, bandwidth: 130305 MiB/s
```

## Prefix sums

## Reductions

