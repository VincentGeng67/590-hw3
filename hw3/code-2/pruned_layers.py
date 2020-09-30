import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class PruneLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PruneLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        # Transpose to match the dimension
        self.mask = np.ones([self.out_features, self.in_features])
        m = self.in_features
        n = self.out_features
        self.sparsity = 1.0
        # Initailization
        self.linear.weight.data.normal_(0, math.sqrt(2. / (m+n)))

    def forward(self, x):
        out = self.linear(x)
        return out
        pass

    def prune_by_percentage(self, q=5.0):
        """
        Pruning the weight paramters by threshold.
        :param q: pruning percentile. 'q' percent of the least 
        significant weight parameters will be pruned.
        """
        np_weight = self.linear.weight.data.cpu().numpy()
        flattened_weights = np.abs(np_weight.flatten())
        # Generate the pruning threshold according to 'prune by percentage'. (Your code: 1 Line) 
        thresh = np.partition(flattened_weights, len(flattened_weights)*(q/100))[len(flattened_weights)*(q/100)]
        # Generate a mask to determine which weights should be pruned (Your code: <=3 Lines)
        position=np.argwhere(np_weight<thresh)
        self.mask[position]=0
#         self.mask = 
        # Multiply weight by mask (Your code: 1 Line) 
        np_weight = self.mask*np_weight
        # Copy back to linear.weight and assign to device (Your code: 1 Line)
        self.linear.weight.data = np_weight.data.device
        # Compute sparsity (Your code: 1 Line)
        self.sparsity = torch.sum(self.linear.weight.data != 0)/torch.numel(self.linear.weight)
        # Copy mask to device for faster computation [Your code: 1 Line]
        self.mask = self.mask.device()


    def prune_by_std(self, s=0.25):
        """
        Pruning by a factor of the standard deviation value.
        :param std: (scalar) factor of the standard deviation value. 
        Weight magnitude below std(weight)*s will be pruned.
        """
        np_weight = self.linear.weight.data.cpu().numpy()
        flattened_weights = np.abs(np_weight.flatten())
        # Generate the pruning threshold according to 'prune by std'. (Your code: 1 Line) 
        thresh = np.std(flattened_weights)*s
        # Generate a mask to determine which weights should be pruned (Your code: <=3 Lines)
        position=np.argwhere(np_weight<thresh)
        self.mask[position]=0
        # Multiply weight by mask (Your code: 1 Line)
        np_weight = self.mask*np_weight
        # Copy back to linear.weight and assign to device (Your code: 1 Line)
        self.linear.weight.data = np_weight.data.device
        # Compute sparsity (Your code: 1 Line)
        self.sparsity = torch.sum(self.linear.weight.data != 0)/torch.numel(self.linear.weight)
        # Copy mask to device for faster computation (Your code: 1 Line)
        self.mask =  self.mask.device()

        pass

class PrunedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(PrunedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        # Expand and Transpose to match the dimension
        self.mask = np.ones([out_channels, in_channels, kernel_size, kernel_size])

        # Initialization
        n = self.kernel_size * self.kernel_size * self.out_channels
        m = self.kernel_size * self.kernel_size * self.in_channels
        self.conv.weight.data.normal_(0, math.sqrt(2. / (n+m) ))
        self.sparsity = 1.0

    def forward(self, x):
        out = self.conv(x)
        return out

    def prune_by_percentage(self, q=5.0):
        """
        Pruning by a factor of the standard deviation value.
        :param std: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """
        np_weight = self.conv.weight.data.cpu().numpy()
        flattened_weights = np.abs(np_weight.flatten())
        # Generate the pruning threshold according to 'prune by percentage. (Your code: 1 Line) 
        thresh = np.partition(flattened_weights, len(flattened_weights)*(q/100))[len(flattened_weights)*(q/100)]
        # Generate a mask to determine which weights should be pruned (Your code: <=3 Lines)
        position=np.argwhere(np_weight<thresh)
        self.mask[position]=0
#         self.mask = 
        # Multiply weight by mask (Your code: 1 Line) 
        np_weight = self.mask*np_weight
        # Copy back to linear.weight and assign to device (Your code: 1 Line)
        self.linear.weight.data = np_weight.data.device
        # Compute sparsity (Your code: 1 Line)
        self.sparsity = torch.sum(self.linear.weight.data != 0)/torch.numel(self.linear.weight)
        # Copy mask to device for faster computation [Your code: 1 Line]
        self.mask = self.mask.device()


    def prune_by_std(self, s=0.25):
        """
        Pruning by a factor of the standard deviation value.
        :param std: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """
        np_weight = self.conv.weight.data.cpu().numpy()
        flattened_weights = np_weight.flatten()
        # Generate the pruning threshold according to 'prune by std'. (Your code: 1 Line) 
        thresh = np.std(flattened_weights)*s
        # Generate a mask to determine which weights should be pruned (Your code: <=3 Lines)
        position=np.argwhere(np_weight<thresh)
        self.mask[position]=0
        # Multiply weight by mask (Your code: 1 Line)
        np_weight = self.mask*np_weight
        # Copy back to linear.weight and assign to device (Your code: 1 Line)
        self.linear.weight.data = np_weight.data.device
        # Compute sparsity (Your code: 1 Line)
        self.sparsity = torch.sum(self.linear.weight.data != 0)/torch.numel(self.linear.weight)
        # Copy mask to device for faster computation (Your code: 1 Line)
        self.mask =  self.mask.device()
        pass



