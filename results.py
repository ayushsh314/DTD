import numpy as np
import torch
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

val_x = np.load("val_x.npy")
val_y = np.load("val_y.npy")

# converting validation images into torch format
val_x = val_x.reshape(564, 1, 128, 128)
val_x  = torch.from_numpy(val_x)

# converting the target into torch format
val_y = torch.from_numpy(val_y)

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(1, 64, 2),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(0.4),

            # Conv2d(32, 64, 3),
            # BatchNorm2d(64),
            # ReLU(inplace=True),
            # MaxPool2d(kernel_size=2, stride=2),
            # Dropout(0.35)
        )

        self.linear_layers = Sequential(
            Linear(64*63*63, 47)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
    
    
# Model class must be defined somewhere
model = Net()
model.load_state_dict(torch.load('model.pth'))

def accuracy_score(arr1, arr2):
    accuracy = np.sum(np.equal(arr1,arr2))/len(arr1)
    return accuracy

# prediction for validation set
predictions = []
for i in range(564) :
    with torch.no_grad():
        output = model(val_x[i:i+1])

    softmax = torch.exp(output)
    prob = list(softmax.detach().numpy())
    predictions.append(np.argmax(prob, axis=1).item())

# accuracy on validation set
print("Accuracy of model: ",accuracy_score(val_y.tolist(), predictions))     