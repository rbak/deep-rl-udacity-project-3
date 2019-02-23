# Deep Reinforcement Learning Project 3
This is an implementation of DDPG for the third project in Udacity's Deep Reinforcement Learning class.  The model is built and trained to solve Unity's [Tennis Environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) scenario.


## Environment Details

## Agent Details

## Installation Requirements
  1. Create a python 3.6 virtual environment.  I used Anaconda for this. (`conda create -n yourenvname python=3.6`)
  2. After activating the environment, pip install the requirements file. (`pip install -r requirements.txt`)

Note: The agent is setup to use the MacOS Tennis.app environment file.

## Running
Run main.py from the repo directory. The main.py file can be run with any of the following flags.
e.g. `python main.py --test`

* `--examine`: Prints information on the learning environment.
* `--random`: Runs an agent that takes random actions.
* `--train`: Trains a new agent including saving a checkpoint of the model weights and printing out scores.
* `--test`: Runs an agent using the included checkpoint file.

Rather than the standard matplotlib plots, I set this one up to push the results to comet.ml
If you want to see a graph of the results from training, you can setup an account there and use the following flag.
The most recent runs can be found here: (https://www.comet.ml/rbak/udacity-deeprl-project-3)

* `--log`: Logs the results to comet.ml.  Must have set the environment variable COMET_API_KEY.
