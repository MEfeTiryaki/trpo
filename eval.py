import argparse

import csv
import glob

import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Evaluation tool')
parser.add_argument('--log-dir', type=str, default=".", metavar='N', help='log directory')
args = parser.parse_args()

def main():
    # print("/home/efe/git/trpo/"+args.log_dir+"*.csv")
    filenames = glob.glob(args.log_dir+"/*.csv")
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


    plt.fill_between(episode,mean-std, mean+std, alpha=0.5)
    plt.grid()
    plt.xlabel("Iteration Number")
    plt.ylabel("Average Total Reward (for 10 run)")
    plt.show()

if __name__ == '__main__':
    main()
