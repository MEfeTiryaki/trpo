# Trust Region Policy Optimization

This a trust region policy optimization implementation for continues action space system. This repo uses some methods from ![\*](https://github.com/ikostrikov/pytorch-trpo).

## Dependencies


## TODO
* Documentation : dependendies, how to save network
* Experience Replay
* Make code stupid proof : check log folders, save and load folders
* Reward plotter for only one run
* automatic log generation bash for the given plots
* various minimal examples 

## Useful Flags

* --max-iteration-number {int} : the max number of episode
* --batch-size {int} : batch size of each episode
* --episode-length {int} : length of each episode (Environmens may limit this internally for example Pendulum has length 200)

* --log : if added at the end of the training the mean reward
* --log-dir {string} : the logging directory
* --log-prefix {string} : name your log file

## Logging
If the --log flag is added to the command line instruction, at the end of the
training the average cummulative reward will be logged automatically.

Recommended way of logging:

First create a log directory
```bash
  mkdir log
  mkdir log/example
```
Then run the train.py
```bash
  python train.py --log -log-dir "log/example"
```
the log file will appear in the given folder. If you run the same command multiple time,
the log code will automatically enumurate the log file names.

## Saving the trained networks


## Useful Reference
### Books
* ![Reinforcement Learning An Introduction](http://incompleteideas.net/book/RLbook2018.pdf)
* ![A Survey on Policy Search for Robotics](https://spiral.imperial.ac.uk:8443/bitstream/10044/1/12051/7/fnt_corrected_2014-8-22.pdf)

### Papers
* ![Trust Region Policy Optimization](http://proceedings.mlr.press/v37/schulman15.pdf)
* ![High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438.pdf)
* ![Towards Generalization and Simplicity in Continuous Control](http://papers.nips.cc/paper/7233-towards-generalization-and-simplicity-in-continuous-control.pdf)

## Other TRPO repos
* ![modular_rl](https://github.com/joschu/modular_rl)
* ![pat-coady/trpo](https://github.com/pat-coady/trpo)
* ![ikostrikov/pytorch-trpo](https://github.com/ikostrikov/pytorch-trpo)
* ![mjacar/pytorch-trpo](https://github.com/mjacar/pytorch-trpo)

If there are some other good implementations please inform me to add to the list

## Results

* Bootstrapping works way better
* Increasing batch size increases the learning rate but simulations takes to long time
* Training policy and value networks with data from same time step results a poor learning performance, even if the value training perform after policy optimization. Training value function with previous data solves the problem. Using more than one previous batch does not improve the results.
* High the value training iteration number results overfitting, and low cause poor learning. Though, this experiments are performed with minibatches with size batch_size/iter, namely minibatch size is not constant.(TODO: add constant batch)


## Experiments
The experiments are performed in Pendulum-v0 environment
### Monte Carlo vs Bootstrap
In this experiment, two different way of estimating return is compared.

1- Monte Carlo : return is calculated using the discounted return of next state

2- Bootstrap : return is calculated using the discounted value approaximation of next state

![bacth_size](https://github.com/MEfeTiryaki/trpo/blob/master/fig/td_mc.png)

### Value function training batch
In this experiment, we train the system with 4 different batch sizes.

![bacth_size](https://github.com/MEfeTiryaki/trpo/blob/master/fig/bacht_size.png)


### Past data for value learning
In ![\*](http://papers.nips.cc/paper/7233-towards-generalization-and-simplicity-in-continuous-control.pdf), they used previous batches data to train the value function to avoid overfitting. ![\*](https://github.com/pat-coady/trpo) used the previous+current batch to train the value function. Here, we are testing different combinations of both to see difference.

![bacth_size](https://github.com/MEfeTiryaki/trpo/blob/master/fig/memory.png)


### Value training iteration number
We test the value training iteration number. The experiment is performed with the a batch size of 5k and the minibatch size are 5k/iter_num.

![bacth_size](https://github.com/MEfeTiryaki/trpo/blob/master/fig/value_iter_max.png)
