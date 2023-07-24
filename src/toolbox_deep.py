"""
License CC BY-NC-SA 4.0 : https://creativecommons.org/licenses/by-nc-sa/4.0/
March 2019, Jonathan Vacher
"""


import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.autograd import Function

import copy


def im_gray_norm_and_torch(im, mean, std, ny, nx):
    n_im = im.shape[0]
    im_whitened = (im - im.mean(axis=(1,2), keepdims=True))
    im_whitened /= im_whitened.std(axis=(1,2), keepdims=True)
    
    im_color = np.zeros((n_im,3,ny,nx))
    for i in range(3):
        im_color[:,i] = im_whitened*std[i] + mean[i]

    im_torch = torch.from_numpy(im_color).float()
    return im_torch

def im_color_norm_and_torch(im, mean, std, ny, nx):
    n_im = im.shape[0]
    im_whitened = (im - im.mean(axis=(1,2), keepdims=True))
    im_whitened /= im_whitened.std(axis=(1,2), keepdims=True)
    
    im_color = np.zeros((n_im,3, ny, nx))
    for i in range(3):
        im_color[:,i] = im_whitened[...,i]*std[i]+mean[i]

    im_torch = torch.from_numpy(im_color).float()
    return im_torch

def get_conv2d_features(model, im_torch):
    deep_features = []
    i = 0 
    temp = im_torch
    for layer in model.children():
        with torch.no_grad():
            if isinstance(layer, nn.Conv2d):
                deep_features.append(layer(temp).numpy()[0])
                i+=1
            temp = layer(temp)
    return np.array(deep_features)



# rebuild a sequential cnn with only conv and relu layers
def keep_conv_relu(cnn):
    cnn = copy.deepcopy(cnn)
    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))

    i = 0  
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
            model.add_module(name, layer)
        elif isinstance(layer, nn.ReLU):
            i += 1
            name = 'relu_{}'.format(i)
            model.add_module(name, layer)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

    return model


# rebuild a sequential cnn with custom layers
def keep_custom(cnn):
    cnn = copy.deepcopy(cnn)
    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))

    i = 0  
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
            model.add_module(name, layer)
        elif isinstance(layer, nn.ReLU):
            i += 1
            name = 'relu_{}'.format(i)
            model.add_module(name, layer)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            model.add_module(name, layer)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

    return model




# matrix square root
class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(sp.linalg.sqrtm(m).real).type_as(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_variables
            sqrtm = sqrtm.data.numpy().astype(np.float_)
            gm = grad_output.data.numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = sp.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).type_as(grad_output.data)
        return Variable(grad_input)


sqrtm = MatrixSquareRoot.apply
