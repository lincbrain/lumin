# LUMIN: <u>L</u>ight-sheet Microscopy Analysis <u>U</u>nified with Distributed and Do<u>m</u>a<u>in</u>-Randomized Generative Models. 
This repository provides a framework to perform large-scale, distributed image segmentation of light-sheet microscopy (LSM) volumes. Additionally, we also provide a new set of augmentation and domain-randomization techniques based on the use of spherical harmonics, to synthesize cortical sections of _ex-vivo_ human brains acquired using LSM. This is particularly helpful in enabling zero-shot segmentation, on previously unseen data.

# Installation
Clone the repository
``` shell
git clone https://github.com/lincbrain/lumin
cd lumin
```
Create new conda environment using the ```environment.yml```file:

``` shell
conda env create --name lsm --file=environment.yml
```
Next, install the code as an editable package in the conda environment created above.
``` shell
pip install -e .
```

# Usage
We provide 3 basic functions in the repository, which are described below. 

1. Distributed Segmentation
2. Light-Sheet Microscopy Synthesis
3. Analysis

The method to use the code for any of the above functions is more or less the same, i.e.: define a configuration file for each of its corresponding scripts.

## Distributed Segmentation
To run distributed segmentation, one can invoke the ```distributed_segment.py```script as demonstratated below.

``` shell
    python distributed_segment.py --config ./configs/segment/<config>.yaml
```
Note that the path specified to the config file argument is just an example, and can be changed as per convenience. 

## Light-Sheet Microscopy Synthesis
[TODO] 

## Analysis
One can perform post-hoc, heuristic analyses on the segmentation and distrbuted stitching algorithms by running the following

``` shell
python analysis.py --config ./configs/analysis/<config>.yaml
```
We provide 2 example config files in ```./configs/analysis/```, where we also briefly describe the function of each of the parameters used.
