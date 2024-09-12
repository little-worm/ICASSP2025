# Template-Free Brain MRI Alignment Using Outlier-Robust Wasserstein Distance under Rigid Transformation

## Environment configuration
```
pip install fugw
python3 -m pip install ipykernel -U --user --force-reinstall
pip install matplotlib
pip install nilearn
pip install plotly
pip install nbformat
pip install wanndb
pip install gym
pip install scikit-image
pip install SimpleITK
pip install git+https://github.com/individual-brain-charting/api.git#egg=ibc_api
``` 
More details about environment configuration can be found at file `environment.yml`.


## Data preparation
- First, you can download the original dataset by  `IBC_dataDownload.py`.
- Second, you can obtain our dataset in this paper by using `gen_my_ibc_data.py`, and the corresponding dataset is stored at folder `my_ibc_data/`.


## Experimental reproduction
### for pairwise alignment 
- run `bash exp_BC.sh`
### for barycenter algorithm 
- run `bash exp_dist.sh`
- the visual version of barycenters are available at   `my_BC_exp_clean.ipynb`.





<!-- download IBC 网址：https://individual-brain-charting.github.io/docs/get_data.html -->
