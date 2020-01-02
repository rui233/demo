import torch.utils.data
import torch.nn
import torch.optim
import torchvision.datasets
import torchvision.transforms
from cnn_net import Net 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 数据读取
train_dataset = torchvision.datasets.MNIST(root='./data/mnist',
        train=True, transform=torchvision.transforms.ToTensor(),
        download=False)
test_dataset = torchvision.datasets.MNIST(root='./data/mnist',
        train=False, transform=torchvision.transforms.ToTensor(),
        download=False)


batch_size = 32
train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size)


net = Net().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr = 0.001)   

# 训练
num_epochs = 5
for epoch in range(num_epochs):
    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = net(images)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print('epoch {}, batch {}, loss = {:g}'.format(
                    epoch, idx, loss.item()))

# 测试
correct = 0
total = 0
for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    preds = net(images)
    predicted = torch.argmax(preds, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    
accuracy = correct / total
print('test accuracy: {:.1%}'.format(accuracy))

torch.save(net.state_dict(),'mnist2.pt')