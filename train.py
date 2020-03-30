import argparse
from itertools import count
import signal
import sys
import os
import time

import numpy as np

import gym

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import scipy.optimize

import matplotlib.pyplot as plt

from value import Value
from policy import Policy
from utils import *

from trpo import trpo_step

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')

# Algorithm Parameters
parser.add_argument('--gamma', type=float, default=0.995, metavar='G', help='discount factor (default: 0.995)')
parser.add_argument('--lambda-', type=float, default=0.97, metavar='G', help='gae (default: 0.97)')
# Value Function Learning Parameters
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G', help='(NOT USED)l2 regularization regression (default: 1e-3)')
parser.add_argument('--val-opt-iter', type=int, default=200, metavar='G', help='iteration number for value function learning(default: 200)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='G', help='learning rate for value function (default: 1e-3)')
parser.add_argument('--value-memory', type=int, default=1, metavar='G', help='ratio of past value to be used to batch size (default: 1)')
parser.add_argument('--value-memory-shuffle', action='store_true',help='if not shuffled latest memory stay') # TODO: implement

# Policy Optimization parameters
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G', help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G', help='damping (default: 1e-1)')
parser.add_argument('--fisher-ratio', type=float, default=1, metavar='G', help='ratio of data to calcualte fisher vector product (default: 1)')

# Environment parameters
parser.add_argument('--env-name', default="Pendulum-v0", metavar='G', help='name of the environment to run')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 1)')

# Training length
parser.add_argument('--batch-size', type=int, default=5000, metavar='N', help='number of steps per iteration')
parser.add_argument('--episode-length', type=int, default=1000, metavar='N', help='max step size for one episode')
parser.add_argument('--max-iteration-number', type=int, default=200, metavar='N', help='max policy iteration number')
# Rendering
parser.add_argument('--render', action='store_true', help='render the environment')
# Logging
parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument('--log', action='store_true', help='log the results at the end')
parser.add_argument('--log-dir', type=str, default=".", metavar='N', help='log directory')
parser.add_argument('--log-prefix', type=str, default="log", metavar='N', help='log file prefix')

# Load
parser.add_argument('--load', action='store_true', help='load models')
parser.add_argument('--save', action='store_true', help='load models')
parser.add_argument('--load-dir', type=str, default=".", metavar='N', help='')


args = parser.parse_args()

env = gym.make(args.env_name)
env.seed(args.seed)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

torch.set_printoptions(profile="full")

if args.load:
    policy_net = Policy(num_inputs, num_actions,30)
    value_net = Value(num_inputs,30)
    set_flat_params_to(value_net, loadParameterCsv(args.load_dir+"/ValueNet"))
    set_flat_params_to(policy_net, loadParameterCsv(args.load_dir+"/PolicyNet"))
    print("Networks are loaded from "+args.load_dir+"/")
else:
    policy_net = Policy(num_inputs, num_actions,30)
    value_net = Value(num_inputs,30)

def signal_handler(sig, frame):
    """ Signal Handler to save the networks when shutting down via ctrl+C
    Parameters:
    Returns:
    """
    if(args.save):
        valueParam = get_flat_params_from(value_net)
        policyParam = get_flat_params_from(policy_net)
        saveParameterCsv(valueParam,args.load_dir+"/ValueNet")
        saveParameterCsv(policyParam,args.load_dir+"/PolicyNet")
        print("Networks are saved in "+args.load_dir+"/")

    print('Closing!!')
    env.close()
    sys.exit(0)

def prepare_data(batch,valueBatch,previousBatch):
    """ Get the batch data and calculate value,return and generalized advantage
    Detail: TODO
    Parameters:
    batch (dict of arrays of numpy) : TODO
    valueBatch  (dict of arrays of numpy) : TODO
    previousBatch (dict of arrays of numpy) : TODO
    Returns:
    """
    # TODO : more description above

    stateList = [ torch.from_numpy(np.concatenate(x,axis=0)) for x in batch["states"]]
    actionsList = [torch.from_numpy(np.concatenate(x,axis=0)) for x in batch["actions"]]

    for states in stateList:
        value = value_net.forward(states)
        batch["values"].append(value)

    advantagesList = []
    returnsList = []
    rewardsList = []
    for rewards,values,masks in zip(batch["rewards"],batch["values"],batch["mask"]):
        returns = torch.Tensor(len(rewards),1)
        advantages = torch.Tensor(len(rewards),1)
        deltas = torch.Tensor(len(rewards),1)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(len(rewards))):
            returns[i] = rewards[i] + args.gamma * prev_value * masks[i] # TD
            # returns[i] = rewards[i] + args.gamma * prev_return * masks[i] # Monte Carlo
            deltas[i] = rewards[i] + args.gamma * prev_value * masks[i]- values.data[i]
            advantages[i] = deltas[i] + args.gamma * args.lambda_* prev_advantage* masks[i]

            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]
        returnsList.append(returns)
        advantagesList.append(advantages)
        rewardsList.append(torch.Tensor(rewards))


    batch["states"] = torch.cat(stateList,0)
    batch["actions"] = torch.cat(actionsList,0)
    batch["rewards"] = torch.cat(rewardsList,0)
    batch["returns"] = torch.cat(returnsList,0)

    advantagesList = torch.cat(advantagesList,0)
    batch["advantages"] = (advantagesList- advantagesList.mean()) / advantagesList.std()

    valueBatch["states"] = torch.cat(( previousBatch["states"],batch["states"]),0)
    valueBatch["targets"] =  torch.cat((previousBatch["returns"],batch["returns"]),0)

def update_policy(batch):
    """ Get advantage , states and action and calls trpo step
    Parameters:
    batch (dict of arrays of numpy) : TODO (batch is different than prepare_data by structure)
    Returns:
    """
    advantages = batch["advantages"]
    states = batch["states"]
    actions = batch["actions"]
    trpo_step(policy_net, states,actions,advantages , args.max_kl, args.damping)

def update_value(valueBatch):
    """ Get valueBatch and run adam optimizer to learn value function
    Parameters:
    valueBatch  (dict of arrays of numpy) : TODO
    Returns:
    """
    # shuffle the data
    dataSize = valueBatch["targets"].size()[0]
    permutation = torch.randperm(dataSize)
    input = valueBatch["states"][permutation]
    target = valueBatch["targets"][permutation]

    iter = args.val_opt_iter
    batchSize = int(dataSize/ iter)

    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(value_net.parameters(), lr=args.lr)
    for t in range(iter):
        prediction = value_net(input[t*batchSize:t*batchSize+batchSize])
        loss = loss_fn(prediction, target[t*batchSize:t*batchSize+batchSize])
        # XXX : Comment out for debug
        # if t%100==0:
        #     print("\t%f"%loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def save_to_previousBatch(previousBatch,batch):
    """ Save previous batch to use in future value optimization
    Details: TODO
    Parameters:
    Returns:
    """
    if args.value_memory<0:
        print("Value memory should be equal or greater than zero")
    elif args.value_memory>0:
        if previousBatch["returns"].size() == 0:
            previousBatch= {"states":batch["states"],
                            "returns":batch["returns"]}
        else:
            previous_size = previousBatch["returns"].size()[0]
            size =  batch["returns"].size()[0]
            if previous_size/size == args.value_memory:
                previousBatch["states"] = torch.cat([previousBatch["states"][size:],batch["states"]],0)
                previousBatch["returns"] = torch.cat([previousBatch["returns"][size:],batch["returns"]],0)
            else:
                previousBatch["states"] = torch.cat([previousBatch["states"],batch["states"]],0)
                previousBatch["returns"] = torch.cat([previousBatch["returns"],batch["returns"]],0)
    if args.value_memory_shuffle:
        permutation = torch.randperm(previousBatch["returns"].size()[0])
        previousBatch["states"] = previousBatch["states"][permutation]
        previousBatch["returns"] = previousBatch["returns"][permutation]

def calculate_loss(reward_sum_mean,reward_sum_std,test_number = 10):
    """ Calculate mean cummulative reward for test_nubmer of trials

    Parameters:
    reward_sum_mean (list): holds the history of the means.
    reward_sum_std (list): holds the history of the std.

    Returns:
    list: new value appended means
    list: new value appended stds
    """
    rewardSum = []
    for i in range(test_number):
        state = env.reset()
        rewardSum.append(0)
        for t in range(args.episode_length):
            state, reward, done, _ = env.step(policy_net.get_action(state)[0] )
            state = np.transpose(state)
            rewardSum[-1] += reward
            if done:
                break
    reward_sum_mean.append(np.array(rewardSum).mean())
    reward_sum_std.append(np.array(rewardSum).std())
    return reward_sum_mean, reward_sum_std

def log(rewards):
    """ Saves mean and std over episodes in log file
    Parameters:
    Returns:
    """
    # TODO : add duration to log
    filename = args.log_dir+"/"+ args.log_prefix \
             + "_env_" + args.env_name \
             + "_maxIter_" +   str(args.max_iteration_number) \
             + "_batchSize_" +   str(args.batch_size) \
             + "_gamma_" +   str(args.gamma) \
             + "_lambda_" +   str(args.lambda_) \
             + "_lr_" +   str(args.lr) \
             + "_valOptIter_" + str(args.val_opt_iter)

    if os.path.exists(filename + "_index_0.csv"):
        id = 0
        file = filename + "_index_" + str(id)
        while os.path.exists(file + ".csv"):
            id = id +1
            file = filename + "_index_" + str(id)
        filename = file
    else:
        filename = filename + "_index_0"

    import csv
    filename = filename+ ".csv"
    pythonVersion = sys.version_info[0]
    if pythonVersion == 3:
        with open(filename, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(rewards)
    elif pythonVersion == 2:
        with open(filename, 'w', ) as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(rewards)
def main():
    """
    Parameters:
    Returns:
    """
    signal.signal(signal.SIGINT, signal_handler)
    time_start = time.time()

    reward_sum_mean,reward_sum_std  = [], []
    previousBatch= {"states":torch.Tensor(0) ,
                    "returns":torch.Tensor(0)}

    reward_sum_mean,reward_sum_std = calculate_loss(reward_sum_mean,reward_sum_std)
    print("Initial loss \n\tloss | mean : %6.4f / std : %6.4f"%(reward_sum_mean[-1],reward_sum_std[-1])  )

    for i_episode in range(args.max_iteration_number):
        time_episode_start = time.time()
        # reset batches
        batch = {"states":[] ,
                 "actions":[],
                 "next_states":[] ,
                 "rewards":[],
                 "returns":[],
                 "values":[],
                 "advantages":[],
                 "mask":[]}
        valueBatch = {"states" :[],
                      "targets" : []}
        num_steps = 0
        while num_steps < args.batch_size:
            state = env.reset()
            reward_sum = 0
            states,actions,rewards,next_states,masks = [],[],[],[],[]
            steps = 0
            for t in range(args.episode_length):
                action = policy_net.get_action(state)[0] # agent
                next_state, reward, done, info = env.step(action)
                next_state = np.transpose(next_state)
                mask = 0 if done else 1

                masks.append(mask)
                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                rewards.append(reward)

                state = next_state
                reward_sum += reward
                steps+=1
                if args.render:
                    env.render()
                if done:
                    break

            batch["states"].append(np.expand_dims(states, axis=1) )
            batch["actions"].append(actions)
            batch["next_states"].append(np.expand_dims(next_states, axis=1))
            batch["rewards"].append(rewards)
            batch["mask"].append(masks)
            num_steps += steps

        prepare_data(batch,valueBatch,previousBatch)
        update_policy(batch) # First policy update to avoid overfitting
        update_value(valueBatch)

        save_to_previousBatch(previousBatch,batch)


        print("episode %d | total: %.4f "%( i_episode, time.time()-time_episode_start))
        reward_sum_mean,reward_sum_std = calculate_loss(reward_sum_mean,reward_sum_std)
        print("\tloss | mean : %6.4f / std : %6.4f"%(reward_sum_mean[-1],reward_sum_std[-1])  )

    if args.log:
        print("Data is logged in "+args.log_dir+"/")
        log(reward_sum_mean)

    print("Total training duration: %.4f "%(time.time()-time_start))

    env.close()




if __name__ == '__main__':
    main()
