# Logic Based Reward Shaping for Multi-agent Reinforcement Learning (MARL)

This repository contains the implementation of the project described in this [document]().

This repository also includes the implementation of the learning-based synthesis algorithm described in this [article](https://arxiv.org/abs/1909.07299) which was developed by Alper Kamil Bozkurt from this [repository](https://github.com/alperkamil/csrl.git).

The video rendering and recording is based on this [gridworld repository](https://github.com/sjunges/gridworld-by-storm).

## Dependencies
 - [Python](https://www.python.org/): (>=3.5)
 - [Rabinizer 4](https://www7.in.tum.de/~kretinsk/rabinizer4.html): ```ltl2ldba``` must be in ```PATH``` (```ltl2ldra``` is optional)
 - [NumPy](https://numpy.org/): (>=1.15)
 
The examples in this repository also require the following optional libraries for visualization:
 - [Matplotlib](https://matplotlib.org/): (>=3.03)
 - [JupyterLab](https://jupyter.org/): (>=1.0)
 - [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/): (>=7.5)
 - [Spot Library](https://spot.lrde.epita.fr/)

## Installation
To install the current release and install the CSRL codebase:
```
git clone https://github.com/IngyN/macsrl.git
cd macsrl
pip3 install .
```

## Basic Usage of this repo
The main class for this repo is [MultiControlSynthesis](multi.py), it takes set of ```ControlSynthesis``` classes (based on the number agents), a ```GridMDP``` object, a ```OmegaAutomaton``` object representing the shared automaton with ```sharedoa=True``` for our method. 

The Graphing is done by loading the saved episode returns then loaded in the [Graphing Notebook](graphing.ipynb). For the video rendering, we use the ```Annotation``` and the ```Plotter```classes the [annotation.py](annotation.py) and the [plotter.py](plotter.py).

## Basic Usage of CSRL
The package consists of three main classes ```GridMDP```, ```OmegaAutomaton``` and ```ControlSynthesis```. The class ```GridMDP``` constructs a grid-world MDP using the parameters ```shape```, ```structure``` and ```label```. The class ```OmegaAutomaton``` takes an LTL formula ```ltl``` and translates it into an LDBA. The class ```ControlSynthesis``` can then be used to compose a product MDP of the given ```GridMDP``` and ```OmegaAutomaton``` objects and its method ```q_learning``` can be used to learn a control policy for the given objective. For example,

## Examples
The repository contains a couple of example IPython notebooks:
 - [Motivating example: Modified Buttons Experiment](shared_oa_ex2.ipynb)
 - [Flag Collection Experiment](shared_oa_ex3.ipynb)
 - [Rendez-vous Experiment](shared_oa_benchmark2.ipynb)

Animations of the case studies: 
 - [Buttons Experiment Video](sharedoa2/shared_oa_ex2_maptest_returns.mp4) 
 - [Buttons Step by Step Images](sharedoa2/)
 - [Flag Collection Step by Step Images](sharedoa3/)
 - [Rendez-vous Step by Step Images](sharedoa_bench2/)

HTML representation of the Automatons: 
 - [Motivating example: Modified Buttons Experiment](example_1.html)
 - [Flag Collection Experiment](example_2.html)
 - [Rendez-vous Experiment](example_3.html)