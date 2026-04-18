# RWKVIR CSR Training Repo

This repository is a trimmed version of the project that keeps only the code and configs needed for the following commands:

```bash
python basicsr/train.py -opt options/train/rwkvir/train_RWKVIR_CSR_x4.yml
python basicsr/test.py -opt options/test/test_CSR.yml
```

## Project Structure

- `basicsr/`: training, testing, model, dataset, loss, metric, and utility code used by the commands above
- `options/train/rwkvir/train_RWKVIR_CSR_x4.yml`: training config
- `options/test/test_CSR.yml`: test config

## Install

Create an environment with Python 3.10+ and install:

```bash
pip install -r requirements.txt
```

If you prefer editable install:

```bash
pip install -e .
```

## Train

```bash
python basicsr/train.py -opt options/train/train_CSR_x4.yml
```

## Test

```bash
python basicsr/test.py -opt options/test/test_CSR.yml
```

## Notes

- Update dataset paths and pretrained model paths inside the yaml files before running on a new machine.
- `basicsr/module/base/cuda_v6/` contains the custom RWKV CUDA extension source used by `vrwkv6.py`.
- The repo was cleaned for GitHub publishing, so unrelated experiments, analysis scripts, duplicate configs, caches, and preview images were removed.

## License

This project keeps the original [Apache 2.0](LICENSE) license file.


## Acknowledgments
The basic code is partially from the below repos.
- [RWKVIR](https://github.com/YuzhenD/Resyn/tree/master)
