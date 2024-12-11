# WT-BCP

Code for paper: WT-BCP: Wavelet Transform based Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation


## Usage
We provide `code`, `data_split` and `models` for LA and ACDC dataset.

Data could be got at [LA](https://github.com/yulequan/UA-MT/tree/master/data) and [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC).

To train a model,
```
python ./code/LA_BCP_train_xnet_plus.py  #for LA training
python ./code/ACDC_BCP_train_xnet_plus.py  #for ACDC training
``` 

To test a model,
```
python ./code/test_LA_xnet_plus.py  #for LA testing
python ./code/test_ACDC_xnet_plus.py  #for ACDC testing
```
