<div align="center">
  
  <div>
  <h1>Dynamic Sub-graph Distillation for Robust Semi-supervised Continual Learning</h1>
  </div>

  <div>
      Yan Fan&emsp; Yu Wang*&emsp; Pengfei Zhu&emsp; Qinghua Hu
  </div>


  <br/>

</div>

Official PyTorch implementation of our AAAI 2024 paper "[Dynamic Sub-graph Distillation for Robust Semi-supervised Continual Learning](https://ojs.aaai.org/index.php/AAAI/article/view/29079)". 

## How to run DSGD?


### Dependencies

1. [torch 1.8.1](https://github.com/pytorch/pytorch)
2. [torchvision 0.6.0](https://github.com/pytorch/vision)
3. [tqdm](https://github.com/tqdm/tqdm)
4. [numpy](https://github.com/numpy/numpy)
5. [scipy](https://github.com/scipy/scipy)
6. [quadprog](https://github.com/quadprog/quadprog)
7. [POT](https://github.com/PythonOT/POT)

### Datasets

We have implemented the pre-processing of `CIFAR10`,  `CIFAR100`, and  `imagenet100`. When training on `CIFAR10` and `CIFAR100`, this framework will automatically download it.  When training on `imagenet100`, you should specify the folder of your dataset in `utils/data.py`.

```python
    def download_data(self):
        assert 0,"You should specify the folder of your dataset"
        train_dir = '[DATA-PATH]/train/'
        test_dir = '[DATA-PATH]/val/'
```

### Run experiment

1. Edit the `[MODEL NAME].json` file for global settings.
2. Edit the hyperparameters in the corresponding `[MODEL NAME].py` file (e.g., `models/icarl.py`).
3. Run:

```bash
python main.py --config=./exps/[MODEL NAME].json --label_num [NUM OF LABELED DATA]
```

where [MODEL NAME] should be chosen from `icarl`, `der`, `icarl_10`, `der_10` etc.



## Acknowledgments

We thank the following repos providing helpful components/functions in our work.

- [PYCIL](https://github.com/G-U-N/PyCIL)

## CITATION
If you find our codes or paper useful, please consider giving us a star or cite with:
```
@inproceedings{fan2024dynamic,
  title={Dynamic Sub-graph Distillation for Robust Semi-supervised Continual Learning},
  author={Fan, Yan and Wang, Yu and Zhu, Pengfei and Hu, Qinghua},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={11},
  pages={11927--11935},
  year={2024}
}
```


