from lasagne.layers import (
    NonlinearityLayer, Conv2DLayer, DropoutLayer, Pool2DLayer, ConcatLayer, Deconv2DLayer,
    DimshuffleLayer, ReshapeLayer, get_output, BatchNormLayer)

from lasagne.nonlinearities import linear, softmax
from lasagne.init import HeUniform
import lasagne
import cPickle
from lasagne import layers as L
import numpy as np

def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2):

    l = NonlinearityLayer(BatchNormLayer(inputs))
    l = Conv2DLayer(l, n_filters, filter_size, pad='same', W=HeUniform(gain='relu'), nonlinearity=linear,
                    flip_filters=False)
    if dropout_p != 0.0:
        l = DropoutLayer(l, dropout_p)
    return l


def TransitionDown(inputs, n_filters, dropout_p=0.2):
    l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
    l = Pool2DLayer(l, 2, mode='max')

    return l

def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
    l = ConcatLayer(block_to_upsample)
    l = Deconv2DLayer(l, n_filters_keep, filter_size=3, stride=2,
                      crop='valid', W=HeUniform(gain='relu'), nonlinearity=linear)
    l = ConcatLayer([l, skip_connection], cropping=[None, None, 'center', 'center'])

    return l




def build(datal,n_classes=11,use_pretrained=True):
    n_filters_first_conv=48
    n_pool=5
    growth_rate=16
    n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
    dropout_p=0.2
    if type(n_layers_per_block) == list:
        assert (len(n_layers_per_block) == 2 * n_pool + 1)
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError
    stack = L.Conv2DLayer(datal, n_filters_first_conv, filter_size=3, pad='same', W=HeUniform(gain='relu'),
                        nonlinearity=linear, flip_filters=False)
    n_filters = n_filters_first_conv

    #####################
    # Downsampling path #
    #####################

    skip_connection_list = []

    for i in range(n_pool):
        # Dense Block
        for j in range(n_layers_per_block[i]):
            # Compute new feature maps
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            # And stack it : the Tiramisu is growing
            stack = L.ConcatLayer([stack, l])
            n_filters += growth_rate
        # At the end of the dense block, the current stack is stored in the skip_connections list
        skip_connection_list.append(stack)

        # Transition Down
        stack = TransitionDown(stack, n_filters, dropout_p)

    skip_connection_list = skip_connection_list[::-1]

    #####################
    #     Bottleneck    #
    #####################

    # We store now the output of the next dense block in a list. We will only upsample these new feature maps
    block_to_upsample = []

    # Dense Block
    for j in range(n_layers_per_block[n_pool]):
        l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
        block_to_upsample.append(l)
        stack = L.ConcatLayer([stack, l])

    #######################
    #   Upsampling path   #
    #######################

    for i in range(n_pool):
        # Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        stack = TransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep)

        # Dense Block
        block_to_upsample = []
        for j in range(n_layers_per_block[n_pool + i + 1]):
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            block_to_upsample.append(l)
            stack = L.ConcatLayer([stack, l])
            
    if(use_pretrained):
        print('load pretrained FC-DenseNet103_weights')
        with np.load("models/FC-DenseNet103_weights.npz") as f:
            saved_params_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(stack, saved_params_values[:-2])
    return stack