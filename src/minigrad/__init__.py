from cProfile import label
import math

from numpy.random import random
import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph, view
import torch


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format="svg", graph_attr={"rankdir" : "LR"})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(
            name=uid,
            label="{%s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad),
            shape="record",
        )
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    dot.render("graph", view=True)

class Neuron:
    def __init__(self, nin):
        self.w = [Value(np.random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(np.random.uniform(-1,1))
    def __call__(self, x):
        #w * x + b inferencia ou foward pass
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) 
        out = act.tanh()
        return out 
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range (len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self.prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad    
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad 
        out._backward = _backward
        return out
    def __rmul__(self, other):
        return self * other

    def __truediv__(self,other):
        return self * other**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other -1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1
    def __sub__(self, other):
        return self + (-other)
    def exp(self):
        x = self.data
        out = Value(math.exp(x),(self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self, ), label="tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    def backward(self):

         topo = []
         visited = set() 
         def build(node: Value):
             if node not in visited:
                 visited.add(node)
                 
                 for child in node.prev:
                     build(child)
                 topo.append(node)
         self.grad = 1.0
         build(self)
         for node in reversed(topo):
             node._backward()

def neuronLayer():
    #Tanh formula  = e^(2x) - 1 / e^(2x) + 1
    #Derivada de tanh = 1 - tanh² ->  
    #inputs
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    #weights w1,w2
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    # bias of the neuron
    b = Value(6.8813735870195432, label='b')
    x1w1 = x1 * w1; x1w1.label = 'x1*w1'
    x2w2 = x2 * w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b; n.label = 'n'
    o = n.tanh(); o.label = 'o'
    #initialize the backprop
    #e^(2x)
    e = (2*n).exp()
    o = (e - 1) / (e + 1)
    o.backward()
    draw_dot(o)
def torchNeuronImpl():
    x1 = torch.Tensor([2.0]).double(); x1.requires_grad = True    
    x2 = torch.Tensor([0.0]).double(); x2.requires_grad = True
    w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
    w2 = torch.Tensor([1.0]).double(); w2.requires_grad = True
    b = torch.Tensor([6.8813735870195432]).double(); b.requires_grad = True
    n = x1*w1 + x2*w2 + b
    o = torch.tanh(n)
    o.backward()
    print('______')
    print('x2', x2.grad.item())
    print('w2', w2.grad.item())
    print('x1', x1.grad.item())
    print('w1', w1.grad.item())
def forwardingNeurons():
    x = [2.0, 3.0, -1.0]
    n = MLP(3, [4,4,1]) 
    o = n(x)
    print(o)
def main():
    #neuronLayer()
    #torchNeuronImpl()
    forwardingNeurons()
if __name__ == "__main__":
    main()
