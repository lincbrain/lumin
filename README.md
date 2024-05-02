# lsm-segmentation
add some stuff

# Installation
clone the repository
``` shell
git clone https://github.com/lincbrain/lsm-segmentation
cd lsm-segmentation
```
create new conda environment using the ```environment.yml```file:

``` shell
conda env create --name lsm --file=environment.yml
```
next, install the code as an editable package in the conda environment created above
``` shell
pip install -e .
```
once this is done, a specified model can be evaluated by running the following:

``` shell
python main.py --config ./configs/<config>.yaml
```

