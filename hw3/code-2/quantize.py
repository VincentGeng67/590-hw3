import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _quantize_layer(weight, bits=8):
    """
    :param weight: A numpy array of any shape.
    :param bits: quantization bits for weight sharing.
    :return quantized weights and centriods.
    """
    # Your code: Implement the quantization (weight sharing) here. Store 
    # the quantized weights into 'new_weight' and store kmeans centers into 'centers_'
    
    return new_weight, centers_

def quantize_whole_model(net, bits=8):
    """
    Quantize the whole model.
    :param net: (object) network model.
    :return: centroids of each weight layer, used in the quantization codebook.
    """
    cluster_centers = []
    assert isinstance(net, nn.Module)
    layer_ind = 0
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().numpy()
            weight, centers = _quantize_layer(weight, bits=bits)
            centers = centers.flatten()
            cluster_centers.append(centers)
            m.conv.weight.data = torch.from_numpy(weight).to(device)
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
        elif isinstance(m, PruneLinear):
            weight = m.linear.weight.data.cpu().numpy()
            weight, centers = _quantize_layer(weight, bits=bits)
            centers = centers.flatten()
            cluster_centers.append(centers)
            m.linear.weight.data = torch.from_numpy(weight).to(device)
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
    return np.array(cluster_centers)

