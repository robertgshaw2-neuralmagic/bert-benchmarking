# Benchmarking BERT

Install:

```bash
pip install transformers torch sparsezoo datasets
```

Benchmark

```bash
python3 benchmark.py --sequence-length 386 --dtype fp16
```