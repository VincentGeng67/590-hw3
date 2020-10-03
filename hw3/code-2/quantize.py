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
    cent = pow(2, bits)
    wshape=weight.shape
    flatten_weight=weight.reshape(-1,1)
    cpy=flatten_weight
    flatten_weight=flatten_weight[abs(flatten_weight)!=0.0]
    flatten_weight=flatten_weight.reshape(-1,1)
    
    init = np.linspace(flatten_weight.min(), flatten_weight.max(), cent)
    kmeans = KMeans(n_clusters=cent, init=init.reshape(-1, 1), n_init=1).fit(flatten_weight)
    centers_ = kmeans.cluster_centers_.flatten()
    centers_=np.where(abs(centers_) !=0.0, centers_, 0.0001)
    ind = kmeans.predict(flatten_weight)
    Map  = np.vectorize(lambda x:centers_ [x])
    result=Map(ind)
    index=0
    
    for c in range(len(cpy)):
        if(abs(cpy[c])!=0.0):
            cpy[c]=result[index]
            index=index+1

    new_weight=cpy.reshape(wshape)
    
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

