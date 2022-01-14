# Distance-dependent anomaly detection

This repo enables measuring the sensitivity of anomaly detection to distance from the camera
on the LostAndFound dataset as described in [Dense anomaly detection by robust learning
on synthetic negative data](https://arxiv.org/pdf/2112.12833.pdf).

### Requirements
Available in *requirements.txt*
* Pillow
* prettytable
* torch
* torchvision

### Setup

For LostAndFound dataset download simply run:
```shell
./prepare_dataset.sh
```

### Usage
Simple demo evaluation script:
```shell
python evaluate.py
```
Example output:
```
+----------------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
|   Range (m)    |  5-10 | 10-15 | 15-20 | 20-25 | 25-30 | 30-35 | 35-40 | 40-45 | 45-50 |
+----------------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
|       AP       |  48.3 | 52.94 | 55.21 | 54.06 | 51.94 | 42.07 | 37.11 | 43.75 | 35.23 |
| FPR at TPR 95% |  7.95 | 10.23 | 11.33 | 16.42 | 20.52 | 26.14 | 28.98 | 34.32 |  43.8 |
|     AUROC      | 98.07 |  97.7 | 97.41 | 96.52 | 95.36 |  93.5 | 91.45 |  90.2 | 86.99 |
+----------------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
```

If used, please cite:
```
@article{grcic22arxiv,
  author    = {Matej Grcic and
               Petra Bevandic and
               Zoran Kalafatic and
               Sinisa Segvic},
  title     = {Dense anomaly detection by robust learning on synthetic negative data},
  journal   = {CoRR},
  volume    = {abs/2112.12833},
  year      = {2021}
  }
```
