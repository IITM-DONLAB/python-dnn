#Activation Functions#

Activation Function is a function used to transform the activation level of a unit (neuron) into an output signal. Typically, activation functions have a "squashing" effect.

*Python-DNN* currently support following Activation Functions:

* `sigmoid`:

> Sigmoid function with equation: f(x) = 1/(1 + e^(-x)).This is an S-shaped (sigmoid) curve, with output in the range (0,1).

* `tanh`:

> The Hyperbolic tangent function is a sigmoid curve, like the logistic function, except that output lies in the range (-1,+1).

* `relu`:

> The rectifier is an activation function defined as f(x) = max(0, x)

* `cappedrelu`:

>It is same as ReLU except we cap the units at 6.ie, f(x) = min(max(x,0),6)
