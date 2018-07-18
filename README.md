# PlasmaTomoML

Tomographic algorithms implemented based on machine learning techniques to perform tomographic reconstructions in [JET](https://www.euro-fusion.org/devices/jet/) bolometer setup and [COMPASS](http://www.ipp.cas.cz/vedecka_struktura_ufp/tokamak/tokamak_compass/) SXR system. 

The code presented relates to the work developed for my M.Sc. Thesis [1] at [Instituto de Plasma e Fusão Nuclear](https://www.ipfn.tecnico.ulisboa.pt/) - [Instituto Superior Técnico](https://tecnico.ulisboa.pt/pt/) (IPFN/IST) under the supervision of Prof. Diogo R. Ferreira and Prof. Horácio Fernandes. Includes the work that generated a contribution for the European Physical Society Conference in Plasma Physics ([EPS2018](https://eps2018.eli-beams.eu/en/)) in Prague [[2](http://ocs.ciemat.es/EPS2018ABS/pdf/P4.1005.pdf)].

Part of the code is based on the work previously developed at JET by D. R. Ferreira in [[3](https://arxiv.org/pdf/1802.02242.pdf)] and the original can be accessed [here](https://github.com/diogoff/plasma-tomography)

## Pre-requisites 

One must have installed and configured the following programs and packages to run the available code at full performance:

- [Python 2](https://www.python.org/downloads/)
- [Theano](http://deeplearning.net/software/theano/)
- [Keras](https://keras.io/) (optimized to run on top of Theano)

## Access to Tomography Databases 

The folders developed for JET/ and COMPASS/ are completely independent of each other. No data from the actual experimets is available in this repository since disclosure is not permited. Neverthless scripts to read from the databases (which will only work if one has access to them) are given as well as instructions for the outputed files structure.

## References:

[1] Plasma Tomography with Machine Learning - D. D. Carvalho (2018) - M.Sc. Thesis on the making...

[2] [Regularization extraction for real-time plasma tomography at JET](http://ocs.ciemat.es/EPS2018ABS/pdf/P4.1005.pdf) - D. R. Ferreira, D. D. Carvalho, P. J. Carvalho, H. Fernandes (2018)

[3] [Full-Pulse Tomographic Reconstruction with Deep Neural Networks](https://arxiv.org/pdf/1802.02242.pdf) - D. R. Ferreira, P. J. Carvalho, H. Fernandes (2018)
