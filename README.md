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

Cite this work:

```bibtex
@INPROCEEDINGS{10647844,
  author={Hossain, Mir Sazzat and Rahman, Akm Mahbubur and Amin, Md. Ashraful and Ali, Amin Ahsan},
  booktitle={2024 IEEE International Conference on Image Processing (ICIP)}, 
  title={Lightweight Recurrent Neural Network for Image Super-Resolution}, 
  year={2024},
  volume={},
  number={},
  pages={1567-1573},
  keywords={Performance evaluation;Recurrent neural networks;Costs;Convolution;Computational modeling;Superresolution;Transformers;Single Image Super-Resolution;Recurrent Neural Networks;Efficient Super-Resolution},
  doi={10.1109/ICIP51287.2024.10647844}
}
```
