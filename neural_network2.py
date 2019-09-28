import numpy as np 
from scipy.io import loadmat
from scipy.sparse import coo_matrix

def Sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def SigmoidPrime(x):
    return Sigmoid(x)*(1.0-Sigmoid(x))

def ReLU(x):
    return max(x,0)

def Softmax(x):
    if x.ndim == 1:
        probs = np.exp(x-np.max(x))
        probs /= np.sum(probs)
    else:
        probs = np.exp(x-np.max(x,axis=0,keepdims=True))
        probs /= np.sum(probs,axis =0, keepdims=True)

    return probs

def ConvertLabelsToOneHotEncoded(y, C = 10):
    Y = coo_matrix((np.ones_like(y),
        (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y

def CostFunction(Y,Yhat):
    return -np.sum(Y*np.log(Yhat))/Y.shape[1]

def Predict(X,weights,biases):
    pass

class NeuralNetwork(object):

    def __init__(self, shape):
        #shape is a list whose each element is the numbers of unit in layer respectively
        self.numLayers = len(shape)
        self.shape = shape
        self.biases = [np.random.randn(y,1) for y in shape[1:]]
        self.weights = [np.random.randn(y,x) 
                        for x,y in zip(shape[:-1],shape[1:])]
            
    def FeedForward(self,X):
        i = 0
        z = X
        Z = []
        a = X
        A = [X]
        
        for b,w in zip(self.biases, self.weights):
            if i < self.numLayers - 1:
                z = np.dot(w,z)+b
            #print(z.shape)
                Z.append(z)
                a = Sigmoid(z)
                A.append(a)
                i += 1
            else:
                z = np.dot(w,z)+b
                Z.append(z)
                a = Sigmoid(z)
            A.append(a)
        return (A,Z)

    def Backprop(self,X,Y):
        nablaBiases = [np.zeros(b.shape) for b in self.biases]
        nablaWeights = [np.zeros(w.shape) for w in self.weights]

        (A, Z) = self.FeedForward(X)
        Yhat = A[-1]
        delta = self.CostDerivative(Yhat,Y)/X.shape[1]
        nablaBiases[-1] = np.sum(delta,axis=1)
        nablaBiases[-1] = nablaBiases[-1].reshape((nablaBiases[-1].shape[0],1))
        nablaWeights[-1] = np.dot(delta,A[-2].T)
        
        for layer in range(2,self.numLayers):
            z = Z[-layer]
            delta = np.dot(self.weights[-layer+1].T,delta) * SigmoidPrime(z)
            nablaBiases[-layer] = np.sum(delta,axis=1)
            nablaBiases[-layer] = nablaBiases[-layer].reshape((nablaBiases[-layer].shape[0],1))
            nablaWeights[-layer] = np.dot(delta,A[-layer-1].T)
        return (nablaBiases,nablaWeights)

    def CostDerivative(self,outputActivation,y):
        return outputActivation-y 
    
    def UpdateMiniBatch(self, trainingData, trainingLabel, miniBatchSize, eta):
        nablaBiases = [np.zeros(b.shape) for b in self.biases]
        nablaWeights = [np.zeros(w.shape) for w in self.weights]
        nTrain = trainingData.shape[1]
        idx = np.random.permutation(nTrain)
        for k in range(0,nTrain,miniBatchSize):
            idxMiniBatches = idx[k:k+miniBatchSize]
            deltaNablaBiases, deltaNablaWeights = self.Backprop(trainingData[:,idxMiniBatches],trainingLabel[:,idxMiniBatches])
            nablaBiases = [ nb+dnb for nb,dnb in zip(nablaBiases,deltaNablaBiases)]
            nablaWeights = [ nw+dnw for nw,dnw in zip(nablaWeights,deltaNablaWeights)]

        self.weights = [w-(eta/miniBatchSize)*nw 
                        for w, nw in zip(self.weights,nablaWeights)]        
        self.biases = [b-(eta/miniBatchSize)*nb
                        for b, nb in zip(self.biases,nablaBiases)]

    def evaluate(self, testData, testLabel):
        A,Z = self.FeedForward(testData)

        result = 0
        for i in range(np.argmax(A[-1],axis = 0).shape[0]):
            result += int(np.argmax(A[-1],axis = 0)[i]==y_test[i])
        return result

    def SGD(self, trainingData, trainingLabel, epochs=2000, miniBatchSize=30, eta=1.0, testData = None, testLabel = None):
        nTest = testData.shape[1]
        for i in range(epochs):
            self.UpdateMiniBatch(trainingData, trainingLabel, miniBatchSize, eta)
            print("epoch {0}: {1}/{2}".format(i,self.evaluate(testData,testLabel),nTest))
        return (self.biases,self.weights)

    def writeResult(self, path):
        for i in range(self.numLayers-1):
            np.savetxt(path,self.biases[i].T, delimiter=', ', fmt='%12.8f')
    
    def readData(self, path, mode):
        pass

if __name__ == "__main__":
    mnist = loadmat('/home/manhpp/Documents/Machine_learning/mnist/mnist-original.mat')
    X = mnist['data'].T
    y = mnist['label'].T.reshape(70000,)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index].T, y_train[shuffle_index]
    Y_train = ConvertLabelsToOneHotEncoded(y_train)
    train = NeuralNetwork([784,30,10])
    #b,w = train.SGD(X_train,Y_train,testData=X_test.T, testLabel=y_test)
    train.writeResult("test.txt")
