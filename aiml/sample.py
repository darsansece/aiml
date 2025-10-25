# import numpy as np
# def step_function(x):
 
#     return 1 if x <= 0 else 0
# class Perceptron:
#     def _init_(self,earning_rate=0.1,epochs=1000):
#         self.lr=earning_rate
#         self.epochs=epochs
#         self.weights=None
#         self.bias=None

#     def train(self,X,y):
#         num_samples,num_features=X.shape
#         self.weights=np.zeros(num_features)
#         self.bias=0
#         for _ in range(self.epochs):
#             for idx,x_i in enumerate(X):
#                 linear_output=np.dot(x_i,self.weights)+self.bias
#                 y_predicted=step_function(linear_output)
#                 update=self.lr*(y[idx]-y_predicted)
#                 self.weights+=update*x_i
#                 self.bias+=update
#     def predict(self,X):
#         linear_output=np.dot(X,self.weights)+self.bias
#         y_predicted=[step_function(i) for i in linear_output]
#         return np.array(y_predicted)
# if _name=="main_":
#     # Example usage
#     X=np.array([[0,0],[0,1],[1,0],[1,1]])
#     y=np.array([0,0,0,1])  # AND gate
#     p=Perceptron(earning_rate=0.1,epochs=10)
#     p.train(X,y)
#     predictions=p.predict(X)
#     print("Predictions:",predictions)

#or gate opp
import numpy as np
def step_function(x):
    return 1 if x >= 0 else 0
class Perceptron:
    def _init_(self, learning_rate=0.1, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        
    def train(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = step_function(linear_output)
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = [step_function(i) for i in linear_output]
        return np.array(y_predicted)
if _name_ == "_main_":
    # Example usage
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])  # OR gate
    p = Perceptron(learning_rate=0.1, epochs=10)
    p.train(X, y)
    predictions = p.predict(X)
    print("Predictions:", predictions)