import numpy as np
import lasagne
from lasagne.layers import get_all_layers
from collections import deque, defaultdict
import theano
import theano.tensor as T
import lasagne
from lasagne import layers as L
from lasagne.nonlinearities import rectify
import sys

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


def save_weights(network, name,additional_params = dict() ):
    print('checkpoint '+name+'.npz')
    param_dict = {"lasagne_param%d" % i: param for i, param in enumerate(L.get_all_param_values(network))}
    param_dict.update({k : additional_params[k].get_value() for k in additional_params.keys()})
    np.savez(name+".npz", **param_dict)
             
def load_weights(network,name):
    f = np.load(name+".npz")
    params = [f["lasagne_param%d" % i] for i in range(len([ j for j in f.files if j.find("lasagne_param") == 0]))]
    additional_params = dict([(i,theano.shared(f[i])) for i in f.files if i.find("lasagne_param") != 0])
    f.close()
    L.set_all_param_values(network,params)
    return additional_params

class L2NormLayer(L.Layer):
    def __init__(self, incoming, epsilon=1e-4, **kwargs):
        super(L2NormLayer, self).__init__(incoming, **kwargs)
        self.epsilon = epsilon
        
    def get_output_shape_for(self,input_shape):
        return input_shape[:-1]+(input_shape[-1]+1,)
        
    def get_output_for(self, input, **kwargs):
        sym = T.concatenate([input,T.ones_like(input[...,:1])],-1)
        sym = sym/(T.sqrt(T.square(sym).sum(-1)+self.epsilon)[:,:,:,None]) 
        return sym

class NormedDense(L.Layer):
    def __init__(self, incoming,num_output, W = lasagne.init.GlorotUniform(),epsilon=1e-4, **kwargs):
        super(NormedDense, self).__init__(incoming, **kwargs)
        self.epsilon = epsilon
        self.num_output = num_output
        self.W = self.add_param(W,(self.input_shape[-1],num_output), 'W', trainable=True, regularizable=True)
        
    def get_output_shape_for(self,input_shape):
        return input_shape[:-1]+(self.num_output,)
        
    def get_output_for(self, input, **kwargs):
        sym = T.dot(input,self.W/(T.sqrt((T.sum(T.square(self.W),0)+self.epsilon))[None,:]))
        return sym
    
class TrivialInit(lasagne.init.Initializer):
    def __init__(self):
        pass
    def sample(self, shape):
        assert(len(shape) == 4)
        assert(shape[0]%3 == 0 and shape[1] % 3 == 0)
        assert(shape[2] == 3 and shape[3] == 3)
        W = np.random.randn(*shape).astype(np.float32)*0.1
        if(shape[0]>shape[1]):
            for k in range(shape[0]/shape[1]):
                for i in range(shape[1]):
                    W[i+shape[1]*k,i,1,1] += 1
        else:
            for k in range(shape[1]/shape[0]):
                f = float(shape[1]/shape[0])
                for i in range(shape[0]):
                    W[i,i+shape[0]*k,1,1] += 1./f
        return theano.shared(W)
    
class Logger(object):
    def __init__(self,*args):
        self.outputs = []
        for a in args:
            if(a == 'std'):
                self.outputs.append(sys.stdout)
            else:
                self.outputs.append(open(a,'a'))
    def log(self,*args,**kwargs):
        if('end' in kwargs):
            end = kwargs['end']
        else:
            end = '\n'
        s = ' '.join([str(a) for a in args])
        for o in self.outputs:
            o.write(s+end)
            o.flush()