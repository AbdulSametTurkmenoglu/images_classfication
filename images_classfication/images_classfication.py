import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%


torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])


train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)



classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 
           'frog', 'horse', 'ship', 'truck')


# %%


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3,padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*8*8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        
    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1,64*8*8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x 




model = SimpleCNN().to(device)
   
# %%


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
num_epochs = 10


# %%


def train_model():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0 
        for i ,(images,labels) in enumerate(train_loader):
            images , labels = images.to(device) , labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs,labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss +=loss.item()
            if (i+1) % 100 == 0 :
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0


# %%


def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

#%%


def save_sample_predictions(num_samples=16):
    model.eval()
    images_shown = 0
    fig = plt.figure(figsize=(12, 12))

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if images_shown >= num_samples:
                    break
                ax = fig.add_subplot(4, 4, images_shown + 1, xticks=[], yticks=[])
                img = images[i].cpu() * 0.5 + 0.5  # normalize geri çevir
                np_img = img.numpy().transpose((1, 2, 0))
                ax.imshow(np_img)
                ax.set_title(f"True: {classes[labels[i]]}\nPred: {classes[preds[i]]}", 
                             color=("green" if preds[i] == labels[i] else "red"))
                images_shown += 1
            
            if images_shown >= num_samples:
                break

    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()






# %%

if __name__ == '__main__':
    print("Starting training...")
    train_model()
    print("\nEvaluating model...")
    evaluate_model()
    print("saving sample predictions...")   
    save_sample_predictions()

    
    
    
            



















