# Graph out the
import matplotlib.pyplot as plt
import pickle
import numpy as np


def plot_cumulative_loss_file(fname):

    p = pickle.load(open(fname,"rb"))

    x = [i[0] for i in p]
    val_loss = [i[1] for i in p]
    train_loss = [i[2] for i in p]
    
    plt.plot(x, val_loss,"r")
    plt.plot(x, train_loss,"b")
    plt.show()

if __name__ == "__main__":
    plot_cumulative_loss_file("/Users/vivek/work/ssd-code/final-100-run-model/cumulative_losses_till_epoch-105")
