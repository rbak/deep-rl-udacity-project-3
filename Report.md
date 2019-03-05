# Project 2 Report

## Algorithm
The learning algorithm used is DDPG.

## Hyperparameters
Most of the hyperparameter values I used are unchanged from the DDPG implementation in the training materials.  The one exception is tau, which I changed to .01.  However due to the instability of this problem, I'm unsure whether this actually helped, or if the following runs were better due to random chance.  I also added epsilon and epsilon decay to decrease the impact of noise over time.

    * memory_size: 10000,      # replay buffer size
    * batch_size: 128,         # sample batch size
    * gamma: .99,              # discount (γ)
    * tau: .01,                # soft update ()
    * lr_actor: 3e-5,          # learning rate of the actor
    * lr_critic: 3e-4,         # learning rate of the critic
    * weight_decay: 0,         # L2 weight decay
    * epsilon: 1,              # starting value for the amount of noise to apply
    * epsilon_decay: 0.999     # Decay rate for the randomness/noise

## Model Architecture
The neural network architecture I choise contained two hidden layers, both with  with 256 nodes.  I chose this after trying a few different sized hidden layers.  I also chose to use leaky_relu for the actor model because it seemed to perform better, although due to the randomness of the training this could have been down to luck.

## Rewards
Rewards for several runs.  The training runs were allowed up to 1000 episodes and cut off when they reached their goal.
The project materials warned this project might have some instability.  The appears to be very true as there was no clear average number of steps to train. Although I never saw less than 200 steps to train, some runs also failed to complete in 1000 steps.

The rewards received during each run.

![Rewards](https://github.com/rbak/deep-reinforcement-learning-project-3/blob/master/results/rewards.png)

The mean rewards received during each run.  This clearly shows the random nature of the training time.

![Mean Rewards](https://github.com/rbak/deep-reinforcement-learning-project-3/blob/master/results/mean-rewards.png)

## Future Improvements
My biggest question coming away from this project was whether this algorithm could be made to train more reliably.  Given more time I wwould like to experiment more with the hyperparameters and the model to see if this is possible.  There may also be improvements to DDPG I am unaware of, so researching that could also be interesting.  Lastly, it might be worthwhile to implement other algorithms and see if they train any more reliably.
