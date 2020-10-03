import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn
from heapq import heappush, heappop, heapify
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _huffman_coding_per_layer(weight, centers):
    """
    Huffman coding for each layer
    :param weight: weight parameter of the current layer.
    :param centers: KMeans centroids in the quantization codebook of the current weight layer.
    :return: encoding map and frequency map for the current weight layer.
    """
    # Your code: Implement the Huffman coding here. Store the encoding map into 'encoding'
    # and frequency map into 'frequency'.
    # flatten_weight=flatten_weight[abs(flatten_weight)!=0.0]
    labels, counts = np.unique(weight, return_counts=True)
    posi0=np.where(abs(labels) == 0.0)
    labels=np.delete(labels,posi0)
    counts=np.delete(counts,posi0)
    # print('label',labels)
    # print('counts',counts)
    # my_list.sort(key=lambda x: x[0])
    heap = [ [con, [element, ""]] for (element,con) in zip(labels,counts) ]
    heapify(heap)
    # mylist.sort(key=lambda x: x[0],reverse=True)
    

    while len(heap) > 1:
        lower = heappop(heap)
        for pair in lower[1:]:
            pair[1] = '0' + pair[1]
            
        upper = heappop(heap)
        for pair in upper[1:]:
            pair[1] = '1' + pair[1]

        heappush(heap, [lower[0] + upper[0]] + lower[1:] + upper[1:])
        
    encodings = dict(heappop(heap)[1:])
    frequency = { value : cnt for (value, cnt) in zip(labels,counts) }
    print('end',encodings)
    return encodings, frequency


def compute_average_bits(encodings, frequency):
    """
    Compute the average storage bits of the current layer after Huffman Coding.
    :param encodings: encoding map of the current layer w.r.t. weight (centriod) values.
    :param frequency: frequency map of the current layer w.r.t. weight (centriod) values.
    :return (float) a floating value represents the average bits.
    """
    total = 0
    total_bits = 0
    for key in frequency.keys():
        total += frequency[key]
        total_bits += frequency[key] * len(encodings[key])
    return total_bits / total

def huffman_coding(net, centers):
    """
    Apply huffman coding on a 'quantized' model to save further computation cost.
    :param net: a 'nn.Module' network object.
    :param centers: KMeans centroids in the quantization codebook for Huffman coding.
    :return: frequency map and encoding map of the whole 'net' object.
    """
    assert isinstance(net, nn.Module)
    layer_ind = 0
    freq_map = []
    encodings_map = []
    bits=0
    total=0
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            bits=bits+(weight.size)*huffman_avg_bits
            total=total+weight.size
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)
        elif isinstance(m, PruneLinear):
            weight = m.linear.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            bits=bits+(weight.size)*huffman_avg_bits
            total=total+weight.size
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)
    avbit=bits/total
    return freq_map, encodings_map,avbit