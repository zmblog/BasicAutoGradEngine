from MicrogradEngine import Value
import random
#this impl is similar to that of pytorch API
class Neuron:
    def __input__(self, NoOfInputs):
        self.weight = [Value(random.uniform(-1, 1)) for i in range(NoOfInputs)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        #we initiate dot product (w dot x) + b
        activation = sum((w_i*x_i for w_i, x_i in zip(self.w, x)), self.b)
        out = activation.tanh()
        return out
    def param (self):
        return self.weight + [self.bias]
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer:
    def __init__(self, NoOfInputs, NoOfOutputs) -> None:
        self.neurons = [Neuron(NoOfInputs) for i in range (NoOfOutputs)]
    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        return outputs
    def param(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

#layers of layers that feed into each other sequentially    
class MLP:
    def __init__(self, NoOfInputs, NoOfOutputs):
        size = [NoOfInputs] + NoOfOutputs
        self.layers = [Layer(size[i], size[i+1])for i in range (len(NoOfOutputs))]
    def __call__(self, x):
        for layers in self.layers:
            x = Layer(x)
        return x
    def param(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    
#performing grad descent manually on example
x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
n(x)
#this is the update
for p in n.param():
    p.data+= -0.01*p.grad #to minimise the loss

#we can do a forward pass on some multidimensional array arr
#i.e let ypred = [n(x) for x in arr]
#loss = sum((yout - yget)**2 for yget, yout in zip(ydesired, ypred))
#then a forward pass by calling loss.backward()
#note this is where you can make a mistake if you dont initialize 0 grad!
#to fix this, call for p in n.param(): p.grad = 0 before calling loss.backwards
#we can repeat the above process many (arbitrary) times