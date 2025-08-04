# AMD GPU benchmarks

A collection of GPU benchmarks to evaluate software stack performance

## building with HIP

```bash
cmake -DCMAKE_HIP_ARCHITECUTRES=gfx90a;gfx942 <GIT-SOURCE-DIR>
```

## building with CUDA

```bash
cmake -DCMAKE_HIP_ARCHITECUTRES=90 <GIT-SOURCE-DIR>
```
