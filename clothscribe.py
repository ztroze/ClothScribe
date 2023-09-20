import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def get_data_loader(training = True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    # Input preprocessing
    custom_transform=transforms.Compose([
        transforms.ToTensor(), # Converts a PIL Image or numpy.ndarray to tensor
        transforms.Normalize((0.1307,), (0.3081,)) # Normalizes the tensor with a mean and standard deviation
        ])

    if training == True:
        train_set=datasets.FashionMNIST('./data',train=True, # Contains images and labels we'll be using to train our neural network
            download=True,transform=custom_transform)
        loader = torch.utils.data.DataLoader(train_set, batch_size = 64) # Retrieve images and labels from data set
        return loader

    else:
        test_set=datasets.FashionMNIST('./data',train=False, # Contains images and labels for model evaluation
                transform=custom_transform)
        loader = torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle = False) # Retrieve images and labels from data set
        return loader




def build_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    # Sequential container for storing layers
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
        )

    return model




def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    for epoch in range(T):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(train_loader):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Zero the parameter gradients
            opt.zero_grad()

            # Forward + backward + optimize
            outputs = model[0](inputs)
            outputs = model[1](outputs)
            outputs = model[2](outputs)
            outputs = model[3](outputs)
            outputs = model[4](outputs)
            outputs = model[5](outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * 64
        print(f'Train Epoch: {epoch}    Accuracy: {correct}/{len(train_loader.dataset)}({100*(correct/len(train_loader.dataset)):0.2f}%) Loss: {running_loss/len(train_loader.dataset):0.3f}')




def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    # Evaluation mode
    correct = 0
    total = 0
    model.eval()
    running_loss = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data

            outputs = model[0](inputs)
            outputs = model[1](outputs)
            outputs = model[2](outputs)
            outputs = model[3](outputs)
            outputs = model[4](outputs)
            outputs = model[5](outputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * 64

    if show_loss == True:
        print(f'Average loss: {running_loss/len(test_loader.dataset):0.4f}')

    print(f'Accuracy: {100*(correct/len(test_loader.dataset)):0.2f}%')




def predict_label(model, test_images, index):
    """
    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1

    RETURNS:
        None
    """
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
                    'Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    outputs = model[0](test_images[index])
    outputs = model[1](outputs)
    outputs = model[2](outputs)
    outputs = model[3](outputs)
    outputs = model[4](outputs)
    outputs = model[5](outputs)
    logits = F.log_softmax(outputs, 1)
    probs = F.softmax(logits, dim=1)
    probs = probs.detach()
    probs_list = []

    for val in probs[0]:
        probs_list.append(float(val))

    probs1 = max(probs_list)
    index1 = probs_list.index(probs1)
    probs_list[index1] = 0
    probs2 = max(probs_list)
    index2 = probs_list.index(probs2)
    probs_list[index2] = 0
    probs3 = max(probs_list)
    index3 = probs_list.index(probs3)
    probs_list[index3] = 0
    
    print(f'{class_names[index1]}: {probs1*100:0.2f}%')
    print(f'{class_names[index2]}: {probs2*100:0.2f}%')
    print(f'{class_names[index3]}: {probs3*100:0.2f}%')

