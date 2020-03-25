import argparse

import csv
import glob

import numpy as np
import matplotlib.pyplot as plt

def main():
    # print("/home/efe/git/trpo/"+args.log_dir+"*.csv")

    for log_dir in ["log/td/memory/no_shuffle/_0",\
                    "log/td/memory/no_shuffle/_previous",\
                    "log/td",\
                    "log/td/memory/no_shuffle/_5",\
                    "log/td/memory/no_shuffle/_10"]:
        filenames = glob.glob(log_dir+"/*.csv")
        print(len(filenames))

        data = []
        for filename in filenames:
            with open(filename, newline='') as csvfile:
                 spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                 for row in spamreader:
                     row = [float(x) for x in row]
                     data.append(row)

        data = np.array(data)

        mean = np.mean(data,axis=0)
        std = np.std(data,axis=0)

        episode = np.arange(0,200,1)
        plt.plot(episode,mean)


        plt.fill_between(episode,mean-std, mean+std, alpha=0.2)
    plt.xlabel("Iteration Number")
    plt.ylabel("Average Total Reward (for 10 run)")
    plt.legend(["now","prev","now+prev","now+5prev","now+10prev"])
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
