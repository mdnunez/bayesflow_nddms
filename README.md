# bayesflow_nddms
#### (Repository version 0.7.3)
Repository for fitting Drift-Diffusion models with identifiable within-trial noise parameters in Python using BayesFlow (and JAGS and Stan)

**Authors: Michael D. Nunez from the Psychological Methods group at the University of Amsterdam**

### Citation
Nunez, M. D., Schubert, A.-L., Frischkorn, G. T., & Oberauer, K. (2024). [Cognitive models of decision-making with identifiable parameters: Diffusion Decision Models with within-trial noise](https://psyarxiv.com/h4fde) PsyArXiv. https://doi.org/10.31234/osf.io/h4fde

### Prerequisites

[Python 3 and Scientific Python libraries](https://www.anaconda.com/products/individual)

### Possible requirements

#### BayesFlow

[BayesFlow](https://github.com/stefanradev93/BayesFlow)

See [BayesFlow install instructions](https://github.com/stefanradev93/BayesFlow/blob/master/INSTALL.rst) to create a BayesFlow conda environment for the most stable method to run these scripts. It is also recommended to keep a local version of BayesFlow on your computer because the package is being actively developed. For this project, I used BayesFlow version 1.1 with Python 3.10. 

See also yaml/bayesflow.yml.

#### JAGS + pyjags

For JAGS installation steps in Ubuntu, see [jags_wiener_ubuntu.md](https://github.com/mdnunez/pyhddmjags/blob/master/jags_wiener_ubuntu.md)

[MCMC Sampling Program: JAGS](http://mcmc-jags.sourceforge.net/)

[Program: JAGS Wiener module](https://sourceforge.net/projects/jags-wiener/)

[Python Repository: pyjags](https://github.com/michaelnowotny/pyjags), can use pip:
```bash
pip install pyjags
```
See also yaml/pyjags.yml

#### Stan + PyStan 2

For this project I used PyStan 2. The newest version of PyStan was PyStan 3, but I didn't find PyStan 3 as easy to use with custom diagnostic and plotting scripts as PyStan 2.

[Here are the docs for PyStan 2](https://pystan2.readthedocs.io)

See also yaml/pystan.yml

### Downloading

The repository can be cloned with `git clone https://github.com/mdnunez/bayesflow_nddms.git`

The repository can also be downloaded via the Code -> _Download zip_ buttons above on this Github page.


### License

bayesflow_nddms is licensed under the GNU General Public License v3.0 and written by Michael D. Nunez from the Psychological Methods group at the University of Amsterdam.


### Selected References
(see also References in preprint of **Citation** above)

Ghaderi-Kangavari, A., Rad, J.A. & Nunez, M.D. (2023). [A General Integrative Neurocognitive Modeling Framework to Jointly Describe EEG and Decision-making on Single Trials.](https://link.springer.com/article/10.1007/s42113-023-00167-4) Computational Brain & Behavior https://doi.org/10.1007/s42113-023-00167-4

Nunez, M. D., Fernandez, K., Srinivasan, R., & Vandekerckhove, J. (2024). [A tutorial on fitting joint models of M/EEG and behavior to understand cognition.](https://link.springer.com/article/10.3758/s13428-023-02331-x) Behavior Research Methods. https://doi.org/10.3758/s13428-023-02331-x

Mattes, A., Porth, E., & Stahl, J. (2022). [Linking neurophysiological processes of action monitoring to post-response speed-accuracy adjustments in a neuro-cognitive diffusion model.](https://www.sciencedirect.com/science/article/pii/S1053811921010697) NeuroImage, 247, 118798. https://doi.org/10.1016/j.neuroimage.2021.118798