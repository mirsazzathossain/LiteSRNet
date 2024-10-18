 # LiteSRNet

A lightweight recurrent network for image super-resolution. This repository contains the code for the paper ["Lightweight Recurrent Neural Network for Image Super-Resolution"](https://ieeexplore.ieee.org/abstract/document/10647844/), accepted in ICIP 2024.

#### Setup configuration

```bash
$ python -m utils.setup_configs --config_name recursive_cnn
```

#### Train

```bash
$ python main.py --config recursive_cnn --test_only False
```
