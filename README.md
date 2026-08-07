# Table of Contents
- [Getting started](#getting-started)
- [Installation](#installation)
- [How to use popquis](#how-to-use-popquis)

# Getting started
```
# Obtain popquis
git clone https://github.com/zkstewart/popquis.git

# Create conda environment
conda create -n popquis python numpy pandas npy-append-array matplotlib scipy pillow

# Run popquis (example below)
python popquis.py \
    -q 0/1:1/1:1/1 0/0:0/1:0/1 0/1:1/2:0/2 \
    -c 1 AND 2 AND 3 \
    -l none weak \
    -t 10 \
    -o outdir \
    --centimorgans 3.0 \
    --popsize 1000 \
    --bootstraps 1000
```

# Installation
It is recommended that you set up an Anaconda or Miniconda environment to run popquis. For now, you should do this as indicated in [Getting started](#getting-started) although a conda package will be made available in the future.

# How to use popquis
popquis is a standalone Python script which performs an all-in-one process of simulating parental genomes, crossing them to produce progeny, and randomly sampling progeny groups in various configurations to calculate segregation statistics. A previously (but incompletely) run analysis can be resumed by use of the same parameters.

Call popquis and request help information like:

```
popquis.py -h
```

All parameters with a single dash e.g., `-q` are mandatory, whereas parameters with double dash e.g., `--centimorgans` have default values. Look through each to see if you should modify these to accommodate your simulation's requirements.

# How to cite
A short technical note describing popquis is in the works. For now, if popquis is useful to you, you can link to this repository.
