

'''Gives multiple layers
and their activations, and the gradients'
activations, and then plots them!'''

# testing out intialization
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F 

# simple 2-linear layers (optional act)
# to examine our activation
# functions
def plot_layers(div_amounts,act=False):
    batch_size = 32
    in_dim = 200
    h_dim = 400

    # nin,nout
    input = torch.randn([32,in_dim])
    lin_1 = torch.randn([in_dim,h_dim]) / (div_amounts[0])
    lin_2 = torch.randn([h_dim,800]) / (div_amounts[1])

    out1 = input @ lin_1
    if act:
        out1 = F.relu(out1)
    out2 = out1 @ lin_2
    if act: 
        out2 = F.relu(out2)

    out2.shape


    # backward pass
    lin_2_grad = out1.T @ out2 * 1
    out_1_grad = out2 @ lin_2.T * 1
    lin_1_grad = input.T @ out_1_grad


    fig,axes = plt.subplots(1,4)

    variance = torch.var(out1[0],dim=-1).item()
    variance2 = torch.var(out2[0],dim=-1).item()


    variance_grad2 = torch.var(lin_2_grad[0],dim=-1).item()
    variance_grad1 = torch.var(lin_1_grad[0],dim=-1).item()

    fig.set_size_inches(30,10)

    axes[0].hist(out1[0],bins=50)
    axes[0].set_title(f"Variance: {variance}")
    axes[1].hist(out2[0],bins=50)
    axes[1].set_title(f"Variance: {variance2}")

    axes[2].hist(lin_2_grad[0],bins=50)
    axes[2].set_title(f"Variance: {variance_grad2}")


    axes[3].hist(lin_1_grad[0],bins=50)
    axes[3].set_title(f"Variance: {variance_grad1}")

    fig.suptitle("Out1, Out2, W2 grad, W1 grad",fontsize=20)
    plt.show()
