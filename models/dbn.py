"""
"""
import numpy

import theano
import theano.tensor as T

from layers.logistic_sgd import LogisticRegression
from layers.mlp import HiddenLayer
from layers.rbm import RBM, GBRBM

from models import nnet


class DBN(nnet):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10,
                 first_layer_gb = True,pretrainedLayers=None):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type first_layer_gb: bool
        :param first_layer_gb: wether first layer is gausian-bernolli or 
                                bernolli-bernolli
        """
        super(DBN, self).__init__()
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.n_layers = len(hidden_layers_sizes)

        if pretrainedLayers == None:
            self.nPreTrainLayers = n_layers
        else :
            self.nPreTrainLayers = pretrainedLayers

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector
                                 # of [int] labels

        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                input_size = n_ins
                layer_input = self.x
            else:
                input_size = hidden_layers_sizes[i - 1]
                layer_input = self.sigmoid_layers[-1].output


            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # the parameters of the sigmoid_layers are parameters of the DBN. 
            # The visible biases in the RBM are parameters of those RBMs, 
            # but not of the DBN.
            self.params.extend(sigmoid_layer.params)
            self.delta_params.extend(sigmoid_layer.delta_params)

            # Construct an RBM that shared weights with this layer
            # the first layer could be Gaussian-Bernoulli RBM
            # other layers are Bernoulli-Bernoulli RBMs
            if i == 0 and first_layer_gb:
                rbm_layer = GBRBM(numpy_rng=numpy_rng,
                              theano_rng=theano_rng,
                              input=layer_input,
                              n_visible=input_size,
                              n_hidden=hidden_layers_sizes[i],
                              W=sigmoid_layer.W,
                              hbias=sigmoid_layer.b)
            else:
                rbm_layer = RBM(numpy_rng=numpy_rng,
                              theano_rng=theano_rng,
                              input=layer_input,
                              n_visible=input_size,
                              n_hidden=hidden_layers_sizes[i],
                              W=sigmoid_layer.W,
                              hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)            

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.params.extend(self.logLayer.params)
        self.delta_params.extend(self.logLayer.delta_params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

        self.output = self.logLayer.prediction();
        self.features = self.sigmoid_layers[-1].output;

    def pretraining_functions(self, train_set_x, batch_size, weight_cost):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param weight_cost: weigth cost

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        momentum = T.scalar('momentum')
        learning_rate = T.scalar('lr')  # learning rate to use

        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None,k=1) for training each RBM.
            r_cost, fe_cost, updates = rbm.get_cost_updates(batch_size, learning_rate,
                                                            momentum, weight_cost)
            # compile the theano function
            fn = theano.function(inputs=[index,
                              theano.Param(learning_rate, default=0.0001),
                              theano.Param(momentum, default=0.5)],
                              outputs= [r_cost, fe_cost],
                              updates=updates,
                              givens={self.x: train_set_x[batch_begin:batch_end]})
            # append function to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns
