import os
import torch

import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

from torch.utils.data import DataLoader
from model import Siamese, Simple

from data import MuseumDataset, MuseumDatasetTest

from opt import parse_args
from tqdm import tqdm
from utils import mkdir


def create_dataset(args, train):

    if train:
        transforms = T.Compose(
            [
                T.Resize((args.size, args.size)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    else:
        transforms = T.Compose(
            [
                T.Resize(args.size),
                T.CenterCrop(args.size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    return MuseumDataset(args.root, transforms, train)


def main():

    torch.backends.cudnn.benchmark = True

    args = parse_args()
    mkdir(args.output)
    print(args.__dict__)
    print(args.__dict__, file=open(os.path.join(args.output, "log.txt"), "a"))

    train_set = create_dataset(args, True)
    val_set = create_dataset(args, False)
    labels = torch.tensor(train_set.pairs[2])
    p_class = 1.0 / len(labels[labels == 1])
    n_class = 1.0 / len(labels[labels != 1])
    sample_probabilities = torch.where(
        labels == 1, torch.full_like(labels, p_class), torch.full_like(labels, n_class)
    )

    epoch_length = labels.shape[0]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_probabilities, epoch_length
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=sampler,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = Siamese()
    model = model.cuda()
    if "best_model.pth" in os.listdir(args.output):
        model.load_state_dict(torch.load(os.path.join(args.output, "best_model.pth")))

    if args.ngpu > 1:
        model = nn.DataParallel(model, range(args.ngpu))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )
    criterion = torch.nn.CosineEmbeddingLoss(margin=args.margin)
    if not args.test_only:
        train(
            model, optimizer, scheduler, criterion, train_loader, val_loader, args.epochs, args.output
        )
    else:
        transforms = T.Compose(
            [
                T.Resize(args.size),
                T.CenterCrop(args.size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        test_set = MuseumDatasetTest(args.root, transforms, args.val_set)

        test_loader = DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        embed(model, test_loader, args.output)


def embed(model, test_loader, output):

    train_embed = torch.zeros((len(test_loader.dataset), 1000))

    for i, img in tqdm(enumerate(test_loader)):

        model.eval()
        x = img.cuda()

        with torch.no_grad():
            idx = i * test_loader.batch_size
            train_embed[idx:idx + test_loader.batch_size], _ = model(x, x)

    torch.save(train_embed, os.path.join(output, "val2_embed.pth"))


def eval(model, val_loader, criterion, output):

    running_loss = 0
    running_corrects = 0
    total = 0

    for pair, labels, idxs in tqdm(val_loader):

        model.eval()
        x1, x2, labels = pair[0].cuda(), pair[1].cuda(), labels.cuda()

        with torch.no_grad():
            y1, y2 = model(x1, x2)
            loss = criterion(y1, y2, labels)

        preds = F.cosine_similarity(y1, y2, dim=1)
        preds = torch.where(
            preds > 0, torch.full_like(preds, 1), torch.full_like(preds, -1)
        )

        # statistics
        running_loss += loss.item() * x1.size(0)
        # .item() converts type from torch to python float or int
        running_corrects += torch.sum(preds == labels).item()
        total += float(y1.size(0))

        mean_loss = running_loss / total  # mean epoch loss
        mean_acc = running_corrects / total  # mean epoch accuracy

    print("EVAL {}/{}".format(mean_loss, mean_acc))
    print(
        "EVAL {}/{}".format(mean_loss, mean_acc),
        file=open(os.path.join(output, "log.txt"), "a"),
    )

    return mean_acc


def train(model, optimizer, scheduler, criterion, train_loader, val_loader, epochs, output):

    i = 0
    best_acc = 0
    for epoch in tqdm(range(epochs)):

        running_loss = 0
        running_corrects = 0
        total = 0

        for pair, labels, idxs in train_loader:
            i += 1

            model.train()
            optimizer.zero_grad()

            x1, x2, labels = pair[0].cuda(), pair[1].cuda(), labels.cuda()
            y1, y2 = model(x1, x2)

            preds = F.cosine_similarity(y1, y2, dim=1)
            preds = torch.where(
                preds > 0, torch.full_like(preds, 1), torch.full_like(preds, -1)
            )

            loss = criterion(y1, y2, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # statistics
            running_loss += loss.item() * x1.size(0)
            # .item() converts type from torch to python float or int
            running_corrects += torch.sum(preds == labels).item()
            total += float(y1.size(0))

            mean_loss = running_loss / total  # mean epoch loss
            mean_acc = running_corrects / total  # mean epoch accuracy

            if not i % 20:
                print("TRAIN {}/{}".format(mean_loss, mean_acc))
                print(
                    "TRAIN {}/{}".format(mean_loss, mean_acc),
                    file=open(os.path.join(output, "log.txt"), "a"),
                ),

        current_acc = eval(model, val_loader, criterion, output)

        if current_acc > best_acc:
            best_acc = current_acc
            torch.save(
                model.module.state_dict(), os.path.join(output, "best_model.pth")
            )


if __name__ == "__main__":
    main()
