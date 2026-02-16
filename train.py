import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import argparse
# import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import TestModel

DEVICE = None

def parse_args():
    parser = argparse.ArgumentParser(description='train MNIST data for study')
    parser.add_argument('--BatchSize','-bs',dest='bs',type=int,default=512)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--learningRate','-lr',dest='lr',default=1e-3)
    parser.add_argument('--savePath',type=str,default=None)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')

    print(f'Device : {DEVICE.type}')
    print(f'BatchSize : {args.bs}') 
    print(f'epochs : {args.epochs}')
    print(f'LeraningRate : {args.lr}')

    mnist_train = MNIST('./datasets/MNIST',download=True,train=True,transform=transforms.ToTensor())
    mnist_test = MNIST('./datasets/MNIST',download=True,transform=transforms.ToTensor())

    train_dataloader = DataLoader(mnist_train,batch_size=args.bs,shuffle=True)
    test_dataloader = DataLoader(mnist_test,batch_size=args.bs,shuffle=True)

    model = TestModel(prob=0.2).to(DEVICE)

    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        for idx, (input, label) in enumerate(train_dataloader):
            input = input.to(DEVICE)
            label = label.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print(f'Epoch:{epoch}[{idx}/{len(train_dataloader)}],Loss:{loss.item():.2f}')

        if (epoch + 1) % 2 == 0:
            model.eval()

            correct = 0
            test_loss = 0
            with torch.no_grad():
                for idx, (input, label) in tqdm(enumerate(test_dataloader),total=len(test_dataloader)):
                    input = input.to(DEVICE)
                    label = label.to(DEVICE)
                    
                    output = model(input)
                    loss = criterion(output,label)

                    prediction = output.max(1,keepdim = True)[1]

                    test_loss += loss.item()
                    correct += prediction.eq(label.view_as(prediction)).sum().item()

            test_loss /= len(test_dataloader)
            test_accuracy = 100. * correct / len(test_dataloader.dataset)
            
            print(f'[EPOCH: {epoch}], Test Loss:{test_loss:.4f},Test Accuracy: {test_accuracy:.2f}')
    
    if args.savePath is not None:
        torch.save(model.state_dict(),args.savePath)