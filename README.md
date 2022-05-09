# Forward modelling of towed time-domain electromagnetics

This is a repository to provide a course materials explaining tTEM (towed time-domain electromagnetics). 

### Locally

To run them locally, you will need to have python installed, preferably through [anaconda](https://www.anaconda.com/download/).

You can then clone this reposiroty. From a command line, run

```
git clone https://github.com/sgkang/gp_190_ttem_forward.git
```

Then `cd` into `gp_190_ttem_forward`

```
cd gp_190_ttem_forward
```

To setup your software environment, we recommend you use the provided conda environment

```
conda env create -f environment.yml
conda activate gp_190_ttem_forward
```

You can then launch Jupyter
```
jupyter notebook
```

Jupyter will then launch in your web-browser.

## Running the notebooks

Each cell of code can be run with `shift + enter` or you can run the entire notebook by selecting `cell`, `Run All` in the toolbar.

![cell-run-all](https://em.geosci.xyz/_images/run_all_cells.png)

For more information on running Jupyter notebooks, see the [Jupyter Documentation](https://jupyter.readthedocs.io/en/latest/)
