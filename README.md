# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
![421905091-a932ae3d-dda2-4d0f-b588-f4092f3a0155](https://github.com/user-attachments/assets/b81ec363-d084-492f-ac84-72b39e20c25a)

## DESIGN STEPS

### STEP 1:
Understand the classification task and identify input and output variables.

### STEP 2:
Gather data, clean it, handle missing values, and split it into training and test sets.

### STEP 3:
Normalize/standardize features, encode categorical labels, and reshape data if needed.

### STEP 4:
Choose the number of layers, neurons, and activation functions for your neural network.

### STEP 5:
Select a loss function (e.g., binary cross-entropy), optimizer (e.g., Adam), and metrics (e.g., accuracy).

### STEP 6:
Feed training data into the model, run multiple epochs, and monitor the loss and accuracy.

### STEP 7:
Save the trained model, export it if needed, and deploy it for real-world use.

## PROGRAM

### Name: SWATHI D
### Register Number: 212222230154

```
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x
```
```

# Initialize the Model, Loss Function, and Optimizer
model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

```
```
def train_model(model,train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()

  if(epoch+1)%10==0:
    print(f'Epoch [{epoch+1}/{epochs}],Loss:{loss.item():.4f}')
```



## Dataset Information

![421906194-98829171-025b-41e4-b3e8-6f08aed6da4c](https://github.com/user-attachments/assets/b7a68e28-f139-4091-a821-e8a260a5590a)


## OUTPUT

### Confusion Matrix

![image](https://github.com/user-attachments/assets/34bdc59f-956e-4c81-8c39-f94ca1e87c86)


### Classification Report

![image](https://github.com/user-attachments/assets/6e810645-ff55-4ed9-9f6f-003d72236186)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/86774c4a-9554-47a8-8037-8fa091c7c601)



## RESULT

Thus a neural network classification model for the given dataset is executed successfully.
