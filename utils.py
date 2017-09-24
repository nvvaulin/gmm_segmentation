import time
import numpy as np
import lasagne
from lasagne.layers import get_all_layers
from collections import deque, defaultdict
import theano
import theano.tensor as T
import lasagne
from lasagne import layers as L
from broadcast import BroadcastLayer,UnbroadcastLayer
from lasagne.nonlinearities import rectify
from lasagne.regularization import l2, regularize_network_params

def get_network_str(layer, get_network=True, incomings=False, outgoings=False):
    """ Returns a string representation of the entire network contained under this layer.
        Parameters
        ----------
        layer : Layer or list
            the :class:`Layer` instance for which to gather all layers feeding
            into it, or a list of :class:`Layer` instances.
        get_network : boolean
            if True, calls `get_all_layers` on `layer`
            if False, assumes `layer` already contains all `Layer` instances intended for representation
        incomings : boolean
            if True, representation includes a list of all incomings for each `Layer` instance
        outgoings: boolean
            if True, representation includes a list of all outgoings for each `Layer` instance
        Returns
        -------
        str
            A string representation of `layer`. Each layer is assigned an ID which is it's corresponding index
            in the list obtained from `get_all_layers`.
        """

    # `layer` can either be a single `Layer` instance or a list of `Layer` instances.
    # If list, it can already be the result from `get_all_layers` or not, indicated by the `get_network` flag
    # Get network using get_all_layers if required:
    if get_network:
        network = get_all_layers(layer)
    else:
        network = layer

    # Initialize a list of lists to (temporarily) hold the str representation of each component, insert header
    network_str = deque([])
    network_str = _insert_header(network_str, incomings=incomings, outgoings=outgoings)

    # The representation can optionally display incoming and outgoing layers for each layer, similar to adjacency lists.
    # If requested (using the incomings and outgoings flags), build the adjacency lists.
    # The numbers/ids in the adjacency lists correspond to the layer's index in `network`
    if incomings or outgoings:
        ins, outs = _get_adjacency_lists(network)

    # For each layer in the network, build a representation and append to `network_str`
    for i, current_layer in enumerate(network):

        # Initialize list to (temporarily) hold str of layer
        layer_str = deque([])

        # First column for incomings, second for the layer itself, third for outgoings, fourth for layer description
        if incomings:
            layer_str.append(ins[i])
        layer_str.append(i)
        if outgoings:
            layer_str.append(outs[i])
        if not(current_layer is None):
            layer_str.append(str(current_layer.name)+str(current_layer.output_shape)) 
        else:
            layer_str.append(str(current_layer)+str(current_layer.output_shape))
        network_str.append(layer_str)
    return _get_table_str(network_str)


def _insert_header(network_str, incomings, outgoings):
    """ Insert the header (first two lines) in the representation."""
    line_1 = deque([])
    if incomings:
        line_1.append('In -->')
    line_1.append('Layer')
    if outgoings:
        line_1.append('--> Out')
    line_1.append('Description')
    line_2 = deque([])
    if incomings:
        line_2.append('-------')
    line_2.append('-----')
    if outgoings:
        line_2.append('-------')
    line_2.append('-----------')
    network_str.appendleft(line_2)
    network_str.appendleft(line_1)
    return network_str


def _get_adjacency_lists(network):
    """ Returns adjacency lists for each layer (node) in network.
        Warning: Assumes repr is unique to a layer instance, else this entire approach WILL fail."""
    # ins  is a dict, keys are layer indices and values are lists of incoming layer indices
    # outs is a dict, keys are layer indices and values are lists of outgoing layer indices
    ins = defaultdict(list)
    outs = defaultdict(list)
    lookup = {repr(layer): index for index, layer in enumerate(network)}

    for current_layer in network:
        if hasattr(current_layer, 'input_layers'):
            layer_ins = current_layer.input_layers
        elif hasattr(current_layer, 'input_layer'):
            layer_ins = [current_layer.input_layer]
        else:
            layer_ins = []

        ins[lookup[repr(current_layer)]].extend([lookup[repr(l)] for l in layer_ins])

        for l in layer_ins:
            outs[lookup[repr(l)]].append(lookup[repr(current_layer)])
    return ins, outs


def _get_table_str(table):
    """ Pretty print a table provided as a list of lists."""
    table_str = ''
    col_size = [max(len(str(val)) for val in column) for column in zip(*table)]
    for line in table:
        table_str += '\n'
        table_str += '    '.join('{0:<{1}}'.format(val, col_size[i]) for i, val in enumerate(line))
    return table_str


def save_weights(network, name ):
    print('checkpoint '+name+'.npz')
    np.savez(name+".npz", **{"param%d" % i: param for i, param in enumerate(L.get_all_param_values(network))})
             
def load_weights(network,name ):
    f = np.load(name+".npz")
    params = [f["param%d" % i] for i in range(len(f.files))]
    f.close()
    L.set_all_param_values(network,params)

def define_updates(network, input_var,background, target_var, learning_rate=0.01, momentum=0.9, l2_lambda=1e-5,train_only=False,params=None):
    if params is None:
        params = L.get_all_params(network, trainable=True)
    l2_loss = l2_lambda * regularize_network_params(network, l2)
        
    train_out = L.get_output(network)
    train_loss, train_acc = _score_metrics(train_out, target_var, l2_loss)
    updates = lasagne.updates.nesterov_momentum(
            train_loss, params, learning_rate=learning_rate, momentum=momentum)
    train_fn = theano.function([input_var,background, target_var],[train_loss, train_acc,train_out],updates=updates)
    if not train_only:
        val_out = L.get_output(network, deterministic=True)
        val_loss, val_acc = _score_metrics(val_out, target_var)
        val_fn = theano.function([input_var,background, target_var], [val_loss, val_acc,val_out])
        return train_fn, val_fn
    else:
        return train_fn

def _score_metrics(out, target_var,  l2_loss=0):
    _EPSILON=1e-6

    target_flat = target_var.dimshuffle(1,0,2,3).flatten(ndim=2).dimshuffle(1,0)
    
    prediction = out.dimshuffle(1,0,2,3).flatten(ndim=2).dimshuffle(1,0)
    prediction = T.clip(prediction,_EPSILON,1-_EPSILON)

    loss = lasagne.objectives.categorical_crossentropy(prediction, target_flat).mean()+l2_loss
    accuracy = T.mean(T.eq(T.argmax(prediction, axis=1), T.argmax(target_flat, axis=1)),
                      dtype=theano.config.floatX)

    return loss, accuracy


def categorical_crossentropy(predictions,labels):    
    _EPSILON = 1e-6
    labels = labels.dimshuffle(1,0,2,3).flatten(ndim=2).dimshuffle(1,0)
    predictions = predictions.dimshuffle(1,0,2,3).flatten(ndim=2).dimshuffle(1,0)
    predictions = lasagne.nonlinearities.softmax(predictions)
    return lasagne.objectives.categorical_crossentropy(T.clip(predictions,_EPSILON,1-_EPSILON),labels).mean()
