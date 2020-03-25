<script type="text/javascript" async
src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js? 
config=TeX-MML-AM_CHTML"
</script>
# Trust Region Policy Optimization

This a trust region policy optimizatio implementation for continues action space system.

## Useful Referece
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

## Experiments

### Monte Carlo vs Bootstrap
In this experiment, two different way of estimating return is compared.

1- Monte Carlo : return is calculated using the discounted return of next state

2- Bootstrap : return is calculated using the discounted value approaximation of next state

![bacth_size](https://github.com/MEfeTiryaki/trpo/blob/master/fig/td_mc.png)

### Value function training batch
In this experiment, we train the system with 3 different batch sizes. 

![bacth_size](https://github.com/MEfeTiryaki/trpo/blob/master/fig/bacht_size.png)


### Past data for value learning

![bacth_size](https://github.com/MEfeTiryaki/trpo/blob/master/fig/memory.png)


### Value training iteration number
![bacth_size](https://github.com/MEfeTiryaki/trpo/blob/master/fig/value_iter_max.png)
