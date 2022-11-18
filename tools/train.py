import datetime
import argparse
import logging
logging.basicConfig(level=logging.INFO)

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Resize, Normalize, RandomRotation, RandomHorizontalFlip, GaussianBlur, ToTensor
from torchinfo import summary

from datasets.imagenet import ImageNet
from datasets.caltech256 import Caltech256
from models.plain_18 import Plain18
from models.plain_34 import Plain34
from models.resnet_18 import ResNet18
from models.resnet_34 import ResNet34
from models.resnet_50 import ResNet50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='select network (plain-18/34, resnet-18/34/50)')
    parser.add_argument('dataset', type=str, help='select dataset (imagenet, caltech256)')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs.')
    parser.add_argument('--batch_size', default=64, type=int, help='how many samples per batch to load.')
    parser.add_argument('--init_lr', default=1e-2, type=float, help='initial learning rate.')
    parser.add_argument('--step_size', default=10, type=int, help='period of learning rate decay.')
    parser.add_argument('--gamma', default=0.1, type=float, help='multiplicative factor of learning rate decay.')
    return parser.parse_args()

def train_1epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):
    epoch_acc = 0.
    epoch_loss = 0.
    model.train()
    for data, target in dataloader:
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        probs = torch.softmax(output, dim=1)
        preds = probs.argmax(dim=1)
        epoch_acc += torch.sum(preds == target).item()
        epoch_loss += loss.item()
    epoch_acc = 100 * epoch_acc / len(dataloader.dataset)
    epoch_loss = epoch_loss / len(dataloader)

    return epoch_acc, epoch_loss

def validate_1epoch(
    model,
    dataloader,
    criterion,
    device
):
    epoch_acc = 0.
    epoch_loss = 0.
    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            preds = probs.argmax(dim=1)
            epoch_acc += torch.sum(preds == target).item()
            epoch_loss += criterion(output, target).item()
    epoch_acc = 100 * epoch_acc / len(dataloader.dataset)
    epoch_loss = epoch_loss / len(dataloader)

    return epoch_acc, epoch_loss  


if __name__ == '__main__':
    # fix seed
    torch.manual_seed(0)

    # parse arguments.
    args = parse_args()

    # check gpu availability.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # resize to 224x224.
    trans = Compose([
        Resize((224, 224)),
        # RandomRotation((-45, 45)),
        # RandomHorizontalFlip(0.5),
        # GaussianBlur(3),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # create dataLoader.
    if args.dataset == 'imagenet':
        train_loader = DataLoader(ImageNet(phase='train', trans=trans), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(ImageNet(phase='val', trans=trans), batch_size=args.batch_size, shuffle=False)
        num_classes = 1000
    elif args.dataset == 'caltech256':
        dataset = Caltech256(trans)
        n = len(dataset)
        n_train = int(n * 0.8)
        n_val = n - n_train
        train_dataset, test_dataset = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        num_classes = 256
    else:
        raise Exception    

    # create model.
    if args.model == 'plain-18':
        model = Plain18(num_classes)
    elif args.model == 'plain-34':
        model = Plain34(num_classes)
    elif args.model == 'resnet-18':
        model = ResNet18(num_classes)
    elif args.model == 'resnet-34':
        model = ResNet34(num_classes)
    elif args.model == 'resnet-50':
        model = ResNet50(num_classes)
    else:
        raise ValueError(f'Invalid model {args.model}')
    summary(model, (1, 3, 224, 224))
    model.to(device)

    # create loss function
    criterion = torch.nn.CrossEntropyLoss()

    # create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr)

    # create scheduler
    scheduler = StepLR(optimizer, args.step_size, args.gamma)

    # loop run epoch
    best_loss = float('inf')
    best_acc = -float('inf')
    writer = SummaryWriter('logs')
    for i in range(1, args.epoch+1):
        logging.info(f'[{datetime.datetime.now()}] start epoch : {i}')
        train_acc, train_loss = train_1epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_loss = validate_1epoch(model, val_loader, criterion, device)
        scheduler.step()
        logging.info(
            f'''
            epoch:{i},
            train loss:{train_loss:.2f},
            train acc:{train_acc:.2f},
            val loss:{val_loss:.2f},
            val acc:{val_acc:.2f}
            '''
        )
        writer.add_scalar('Acc/train', train_acc, i)
        writer.add_scalar('Acc/val', val_acc, i)
        writer.add_scalar('Loss/train', train_loss, i)
        writer.add_scalar('Loss/val', val_loss, i)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], i)
        if best_acc < val_acc:
            torch.save(model.state_dict(), 'weights/best_acc.pth')
            best_acc = val_acc
        if best_loss > val_loss:
            torch.save(model.state_dict(), 'weights/best_loss.pth')
            best_loss = val_loss

    writer.close()