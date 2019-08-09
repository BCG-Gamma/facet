# gamma-ml

Top level project for BCG GAMMA Python projects. 

Designed to contain solely
minimalistic artifacts for reuse, which *do not fit better* in any of the
specific projects.

Currently this project contains:

- `gamma.ml` a package for various Machine Learning tasks like model selection, 
validation, fitting & predicting as well as inspection (using SHAP).
- `gamma.ml.Sample` utility class wrapping around a Pandas dataframe, easing common
ML related operations
- `gamma.ml.viz` Drawer and styles for Dendrograms, OOP model of a scipy linkage matrix

# Installation
The pip-project `gamma-ml` can be installed using:
- `pip install git+ssh://git@git.sourceai.io/alpha/gamma-ml.git#egg=gamma.common`
 (*latest version*)
 - `pip install git+ssh://git@git.sourceai.io/alpha/gamma-ml.git@[VERSION-TAG]#egg=gamma.common`
 (*specific version -  currently not available*)

Ensure that you have set up a working SSH key on git.sourceai.io!

# Documentation
Documentation for all of alpha's Python projects is available at: 
https://git.sourceai.io/pages/alpha/alpha/

# API-Reference
See: https://git.sourceai.io/pages/alpha/alpha/gamma.ml.html

# Contribute & Develop
Check out https://git.sourceai.io/alpha/alpha for developer instructions and guidelines.