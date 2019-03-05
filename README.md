# Deep Reinforcement Learning Project 3
This is an implementation of DDPG for the third project in Udacity's Deep Reinforcement Learning class.  The model is built and trained to solve Unity's [Tennis Environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) scenario.

![Trained Agent](https://github.com/rbak/deep-reinforcement-learning-project-3/blob/master/results/tennis2-hd.gif)


## Environment Details
The Tennis learning environment has the following properties:

  * State Space - The state space has 8 dimensions, including the position and velocity of both the agent and ball.
  * Action Space - There are two continuous actions available: move toward or away from the net, and jumping.
  * Goal - The Unity Github page specifies a goal of 2.5.  However for this project my goal was to achieve a mean of .5 points over 100 episodes.

For more information see the [Unity Github repo](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis)

## Agent Details
The agent uses DDPG.  It is implemented in python3 using PyTorch, and uses a two hidden layers.  While this implementation does learn successfully, it does so unreliably.  See the report for more details.


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

This project can push the results to comet.ml
If you want to see a graph of the results from training, you can setup an account there and use the following flag.
The most recent runs can be found here: (https://www.comet.ml/rbak/udacity-deeprl-project-3)

* `--log`: Logs the results to comet.ml.  Must have set the environment variable COMET_API_KEY.
