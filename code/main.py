import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
from networks import Net

import argparse
import os
from torch.utils.data import random_split
from sklearn.model_selection import StratifiedKFold
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=777, help='seed')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')
parser.add_argument('--dataset', type=str, default='DD', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--epochs', type=int, default=100000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50, help='patience for earlystopping')
parser.add_argument('--combine_ratio', type=int, default=0.5, help='combine_ratio=self contribution/neighber ')
parser.add_argument('--fold_validation', type=bool, default=False, help='10 fold_validation/one fold')
args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'
print("using " + args.device)

dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset)
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features


def test(model, loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out, data.y, reduction='sum').item()
    return correct / len(loader.dataset), loss / len(loader.dataset)


def run(model):
    min_loss = 1e10
    patience = 0
    for epoch in range(args.epochs):
        print("Epoch{}:".format(epoch))
        model.train()
        for i, data in enumerate(train_loader):
            # print("batch"+str(i))
            data = data.to(args.device)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            # print("Training loss:{}".format(loss.item()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        val_acc, val_loss = test(model, val_loader)
        print("Validation loss:{}\taccuracy:{}".format(val_loss, val_acc))
        if val_loss < min_loss:
            torch.save(model.state_dict(), 'latest.pth')
            print("Model saved at epoch{}".format(epoch))
            min_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > args.patience:
            print("Early stop at epoch{}".format(epoch))
            break

    model = Net(args).to(args.device)
    model.load_state_dict(torch.load('latest.pth'))
    test_acc, test_loss = test(model, test_loader)
    print("Test accuarcy:{}".format(test_acc))


def run_only_train_test(model):
    max_acc = 0
    patience = 0
    for epoch in range(args.epochs):
        print("Epoch{}:".format(epoch))
        model.train()
        for i, data in enumerate(train_loader):
            # print("batch"+str(i))
            data = data.to(args.device)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            # print("Training loss:{}".format(loss.item()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        test_acc, test_loss = test(model, test_loader)
        print("test loss:{}\taccuracy:{}".format(test_loss, test_acc))
        print("max acc:" + str(max_acc))
        if test_acc > max_acc:
            torch.save(model.state_dict(), 'latest.pth')
            print("Model saved at epoch{}".format(epoch))
            max_acc = test_acc
            patience = 0
        else:
            patience += 1
        #if patience > args.patience:
            #print("Early stop at epoch{}".format(epoch))
            #break

    model = Net(args).to(args.device)
    model.load_state_dict(torch.load('latest.pth'))
    test_acc, test_loss = test(model, test_loader)
    print("Test accuarcy:{}".format(test_acc))


if __name__ == '__main__':
    if args.fold_validation:  # 10 fold_validation
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
        index_fold = 0
        for index_train_validation, index_test in skf.split(np.zeros(len(dataset)), dataset.data.y):
            index_fold += 1
            print("Fold " + str(index_fold))
            index_train_validation = torch.from_numpy(index_train_validation).to(torch.long)
            index_test = torch.from_numpy(index_test).to(torch.long)
            train_validation = dataset[index_train_validation]
            test_set = dataset[index_test]
            num_training = int(len(train_validation) * 0.9)
            num_val = len(train_validation) - num_training

            training_set, validation_set = random_split(train_validation, [num_training, num_val])
            train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
            model = Net(args).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            run(model)
    else:  # one random fold validation
        num_training = int(len(dataset) * 0.9)
        num_test = len(dataset) - (num_training)
        training_set, test_set = random_split(dataset, [num_training, num_test])

        train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        model = Net(args).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        run_only_train_test(model)

    """else:  # one random fold validation
        num_training = int(len(dataset) * 0.8)
        num_val = int(len(dataset) * 0.1)
        num_test = len(dataset) - (num_training + num_val)
        training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

        train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        model = Net(args).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        run(model)
    """
