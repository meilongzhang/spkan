# SPKAN - Sparse Kanvolutions (Sparse Convolutions with Kolmogorov-Arnold Network)
### Introducing Convolutional KAN Networks!
This project extends the idea of the innovative architecture of Kolmogorov-Arnold Networks (KAN) to sparse convolutions.

### Authors
This repository was made by:
 - Mellon Zhang | meilongz@gatech.edu | [LinkedIn](https://www.linkedin.com/in/alexanderbodner/)

### Credits
This repository builds upon an implementation of Convolutional-KANS which is available [here](https://github.com/AntonioTepsich/Convolutional-KANs).
This repository uses an efficient implementation of KAN which is available [here](https://github.com/Blealtan/efficient-kan).
The original implementation of KAN is available [here](https://github.com/KindXiaoming/pykan). 
The original paper of the KAN is available [here](https://arxiv.org/pdf/2404.19756).

### What is a KAN?
KANs are promising alternatives of Multi-Layer Perceptrons (MLPs). KANs have strong mathematical foundations just like MLPs: MLPs are based on the universal approximation theorem, while KANs are based on Kolmogorov-Arnold representation theorem. KANs and MLPs are dual: KANs have activation functions on edges, while MLPs have activation functions on nodes. KAN seems to be more parameter efficient than MLPs, but each KAN Layer has more parameters than a MLP layer. 

<img width="1163" alt="mlp_kan_compare" src="https://github.com/KindXiaoming/pykan/assets/23551623/695adc2d-0d0b-4e4b-bcff-db2c8070f841">

For more information about this novel architecture please visit:
- The official Pytorch implementation of the architecture: https://github.com/KindXiaoming/pykan
- The research paper: https://arxiv.org/abs/2404.19756

### What is a KAN Convolution?
KAN Convolutions are very similar to convolutions, but instead of applying the dot product between the kernel and the corresponding pixels in the image, we apply a **Learnable Non Linear activation function** to each element, and then add them up. The kernel of the KAN Convolution is equivalent to a KAN Linear Layer of 4 inputs and 1 output neuron. For each input i, we apply a ϕ_i learnable function, and the resulting pixel of that convolution step is the sum of ϕ_i(x_i). This can be visualized in the following figure.

![image](./images/Convs.png)


### Parameters in a KAN Convolution
Suppose that we have a KxK kernel. In this case, for each element of this matrix we have a ϕ, which its parameter count is: gridsize + 1. For implementation issues, efficient kan defines:
![image](./images/splines.png)

This gives more expressability to the activation function b. So the parameter count for a linear layer is gridsize + 2. So in total we have K²(gridsize + 2) parameters for KAN Convolution, vs only K² for a common convolution. Consider that gridsize is typically (in our experiments) between k and k², but k tends to be a small value, between 2 and 16. 

# Installation
Currently only tested on Python 3.9

```bash
pip install spkan
```
