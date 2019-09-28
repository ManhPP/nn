import numpy as np
from scipy.io import loadmat

def Sigmoid(x):
    return 1/(1+np.exp(-x))

def SigmoidPrime(x):
    return Sigmoid(x)*(1-Sigmoid(x))

class NeuralNetwork(object):

    def __init__(self, shape):
        #shape is a list whose each element is the numbers of unit in layer respectively
        self.numLayers = len(shape)
        self.shape = shape
        self.biases = [np.random.randn(y,1) for y in shape[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(shape[:-1],shape[1:])]

    def FeedForward(self, x):
        for b,w in zip(self.biases,self.weights):
            x = Sigmoid(np.dot(w,x)+b)
        return x

    def SGD(self, trainingData, epochs, miniBatchSize, eta, testData = None):
        if testData:
            nTest = len(testData)
        nTrain = len(trainingData)
        for i in range(epochs):
            np.random.shuffle(trainingData)
            miniBatches = [trainingData[k:k+miniBatchSize] for k in range(0,nTrain,miniBatchSize)]
            for miniBatch in miniBatches:
                self.UpdateMiniBatch(miniBatch,eta)
            if testData:
                print("epoch{0}: {1} / {2}".format(i,self.evaluate(testData),nTest))
            else:
                print("epoch {0} complete".format(i))

    def UpdateMiniBatch(self, miniBatch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in miniBatch:
            deltaNabla_b, deltaNabla_w = self.BackProp(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b,deltaNabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w,deltaNabla_w)]
        self.weights = [w-(eta/len(miniBatch))*nw for w, nw in zip(self.weights,nabla_w)]
        self.biases = [b-(eta/len(miniBatch))*nb for b, nb in zip(self.biases,nabla_b)]

    def BackProp(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs =[]
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = Sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1],y)*SigmoidPrime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activations[-2].transpose())
        for layer in range(2,self.numLayers):
            z = zs[-layer]
            delta = np.dot(self.weights[-layer+1].transpose(),delta) * SigmoidPrime(z)
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta,activations[-layer-1].transpose())
        return (nabla_b,nabla_w)
        
    def evaluate(self,testData):
        testResults = [(np.argmax(self.FeedForward(x)),y) for (x,y) in testData]
        return sum(int(x==y) for (x,y) in testResults)

    def cost_derivative(self, outputActivation, y):
        return (outputActivation - y) 

if __name__ == "__main__":
    net = NeuralNetwork([784,30,10])
    mnist = loadmat('/home/manhpp/Documents/Machine_learning/mnist/mnist-original.mat')
    X = mnist['data'].T
    y = mnist['label'].T.reshape(70000,)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
    trainingData = list(zip(X_train,y_train))
    testData = list(zip(X_test,y_test))
    net.SGD(trainingData,30, 10, 0.001)
