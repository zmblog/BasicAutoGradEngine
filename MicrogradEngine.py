import math
class Value:
    def __init__(self, data, _children=(), operation='') -> None:
        self.data = data
        self._prev = set(_children)
        #set used for efficientcy
        self._operation = operation
        self.grad = 0.0
        #Variable grad maintains the derivative of the output wrt. self
        self.back = lambda: None #we set the basic function to be an empty function

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        if (isinstance(other, Value)!=True):
            Value(other)
        #this allows us to 
        out = Value(self.data+other.data, (self, other), '+')
        # this makes self and other the children
        def back():
            self.grad += out.grad
            other.grad += out.grad #this is all chain rule
        out.back = back
        return out

    def __mul__(self, other):
        if (isinstance(other, Value)!=True):
            Value(other)
        out = Value(self.data*other.data, (self, other), '*') 
        def back():
            self.grad += out.grad*other.data
            other.grad += out.grad*self.data #this is all chain rule
        
        out.back = back
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'**{other}')
        def back():
            self.grad+=(other*self.data**(other-1))*out.grad
        out.back=back
        return out
    
    #common activation functions

    def tanh(self):
        x = self.data
        tanhVal = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(tanhVal, (self, ), 'tanh')
        def back():
            self.grad += (1-tanhVal**2)*out.grad
        out.back = back
        return out
    def logistic(self):
        x = self.data
        sigmoid = (1)/(1+math.exp(-x))
        out = Value(sigmoid, (self, ), 'logistic')
        def back():
            self.grad += (sigmoid*(1-sigmoid))*out.grad
        out.back = back
        return out
    def ReLu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def back():
            self.grad += (out.data > 0) * out.grad
        out.back = back

        return out
    
    #Building out the graph of nodes (and their gradients wrt output)
    #We use topological sort to build a DAG of the nodes
    #this is so that we can make sure that for any node, it is only added
    #to the DAG after its children have been added
    #if we call 'back' on all the noes in a toplogical order, this then
    #carries out back propogration algorithm

    def backprop(self):
        TopologicalGraph = []
        visited = set()
        def BuildGraph(v):
            if vertex not in visited:
                visited.add(vertex)
                for child in vertex._prev:
                    BuildGraph(child)
                TopologicalGraph.append(vertex)
        BuildGraph(self)
        self.grad = 1 #output always has grad 1 trivially
        #we now carry out backprop
        for vertex in reversed(TopologicalGraph):
            vertex.back()


    #functionalities to make sure we can do reverse operations
    #eg if we cannot do 2*self, b.c 2 is not Value, then we do self*2

    def __neg__(self): # -self
        return self * -1
    def __ReverseAdd__(self, other): # other + self
        return self + other
    def __sub__(self, other): # self - other
        return self + (-other)
    def __ReverseSub__(self, other): # other - self
        return other + (-self)
    def __ReverseMul__(self, other): # other * self
        return self * other
    def __div__(self, other): # self / other
        return self * other**-1
    def __ReverseDiv__(self, other): # other / self
        return other * self**-1
    
#tests  
a = Value(2.0)
b = Value(-3)
d = a*b
print(d.data)
    