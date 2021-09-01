This repository contains a summary of my 10 week summer research project at Imperial College London. Hopefully there is enough information and code for someone to pick up this project some time in the future. Feel free to contact me with any questions at <eloise.lardet@gmail.com>.

## Overview
For a detailed overview of the project please read the `xxx' document. This explains all the equations and models behind the Python code, as well as discussion of relevant figures and analyses about phase separation and demixing. There is also an explanation of the efficient neighbour list used to speed up simulation times.

## Models
Key numerical models used throughout the project written in Python.
All the models have been updated to use an efficient combined neighbour list method, along with the Python package numba.

- `abp-one-pop.py': One population ABP model using Euler or SRK method.
- `abp-alpha-const.py': Two population ABP model with different (constant) persistences using Euler method.
- `abp-alpha-change.py': Two population ABP model with a decrease in persistence in particle type B after an A/B collision

## Analysis
- `anis_snapshots.py': Functions to plot snapshots and create animations for both one population and two population models
- `dp.py': Function to calculate demixing parameter (using Python package Freud)
- `local-density.py': Various functions for calculating and plotting local number densities

## Figures
Various figures contained in the report linked above (most generated from functions in Analysis codes). There are also a few animation examples :)
