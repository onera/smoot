[![Tests](https://github.com/OneraHub/smoot/workflows/Tests/badge.svg)](https://github.com/OneraHub/smoot/actions?query=workflow%3ATests)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# smoot

## Installation
<code>
  git clone git@github.com:RobinGRAPIN/smoot.git
</code>

Necessary packages : <code>pymoo</code>,<code>smt</code>

## Description

This surrogate based multi-objective Bayesian optimizer has been created to see the performance of the WB2S criterion adapted to multi-objective problems.
Given a black box function f : **x** -> **y** with **bolds** characters as vectors, <code>smoot</code> will give an accurate approximation of the optima with few calls of f.

![modeli1](ressources/f1_avant.png)
![modeli2](ressources/f2_avant.png)

![activ](ressources/wb2s_vs_ehvi.png)

![modeli12](ressources/f1_apres.png)
![modeli22](ressources/f2_apres.png)

### Utilisation

Look at the Jupyter notebook in the *tutorial* folder.

You will learn how to use implemented the functionnalities and options such as :
 - The choice of the infill criterion
 - The method to manage the constraints

For additional questions, contact: robingrapin@orange.fr
