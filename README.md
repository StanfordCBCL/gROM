## Graph Reduced Order Models ##

![run_tests](https://github.com/StanfordCBCL/gROM/actions/workflows/run_tests.yml/badge.svg)

In this repository we implement reduced order models for cardiovascular simulations using Graph Neural Networks (GNNs).

<p  align="center">
    <img src="https://github.com/lucapegolotti/gROM/blob/main/.github/aortofemoral_simulation.gif" alt="Simulation">
</p>


### Install the virtual environment ###

Let us first install `virtualenv`:

    pip install virtualenv

Then, from the root of the project:

    bash create_venv.sh

This will create a virtual environment `gromenv` with the required dependencies.

### Download the data ###

The data can be downloaded [here](https://drive.google.com/open?id=1IByz6kyouNtNgnOxKrFK4DnAVu2yh6S1&authuser=lpego%40stanford.edu&usp=drive_fs).
Next, duplicate or rename `data_location_example.txt` as `data_location.txt` and set in it the location of the downloaded `gromdata` folder.

Note: `.vtp` files can be  inspected with [Paraview](https://www.paraview.org).

The `gromdata` contains all the data necessary to train the GNN. However, it is possible to regenerate the data by launching `python graph1d/generate_graphs.py` from the root of the project.

### Train a GNN ###

From root, type

    python network1d/training.py

The parameters of the trained model and hyperparameters will be saved in `models`, in a folder named as the date and time when the training was launched.

### Test a GNN ###

Within the directory `graphs`, type

    python network1d/tester.py $NETWORKPATH

For example,

    python network1d/tester.py models/01.01.1990_00.00.00

This compute errors for all train and test geometries.
In the example, `models/01.01.1990_00.00.00` is a model generated after training (see Train a GNN).

Some already-trained models are included in `gromdata`
