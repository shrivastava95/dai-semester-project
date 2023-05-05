import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as tf
import torchattacks
from tqdm import tqdm
import argparse


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack-method",
        type=str,
        default="pgd",
        choices=["pgd", "fgsm", "jitter"]
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default=20
    )
    args = parser.parse_args()
    return args


def train(epoch, model, train_loader, criterion, optimizer):
    model.train()

    c, t, l = 0, 0, 0
    for (images, labels) in tqdm(train_loader):
        images = images.float().cuda()
        labels = labels.long().cuda()

        optimizer.zero_grad()
        outputs = model(images)
        preds = torch.argmax(outputs, 1)
        c += (preds == labels).sum().item()
        t += labels.shape[0]

        loss = criterion(outputs, labels)
        l += loss.item()

        loss.backward()
        optimizer.step()

    train_stats = {"acc": round(c/t, 4), "loss": l}
    tqdm.write(f"After {epoch+1} epoch(s), ResNet-18 gets {round(c/t, 4)*100} % train accuracy and {l} loss")
    return train_stats

def evaluate(epoch, model, test_loader, criterion):
    model.eval()

    c, t, l = 0, 0, 0
    for (images, labels) in test_loader:
        images = images.float().cuda()
        labels = labels.long().cuda()

        with torch.no_grad():
            outputs = model(images)
            preds = torch.argmax(outputs, 1)
            c += (preds == labels).sum().item()
            t += labels.shape[0]

            loss = criterion(outputs, labels)
            l += loss.item()

    test_stats = {"acc": round(c/t, 4), "loss": l}
    tqdm.write(f"After {epoch+1} epoch(s), ResNet-18 gets {round(c/t, 4)*100} % test accuracy and {l} loss")
    return test_stats

def main(args):
    transforms = tf.Compose([
        tf.Resize((32, 32)),
        tf.Grayscale(3),
        tf.ToTensor(),
        #tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = dsets.MNIST(root="./datasets/MNIST/data", train=True, download=True, transform=transforms)
    test_dataset = dsets.MNIST(root="./datasets/MNIST/data", train=False, download=True, transform=transforms)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    model = torchvision.models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(512, 10, bias=True)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # train the model
    stats = {"train":[], "test":[]}
    for epoch in range(args.epochs):
        train_stats = train(epoch, model, train_loader, criterion, optimizer)
        test_stats = evaluate(epoch, model, test_loader, criterion)

        stats["train"].append(train_stats)
        stats["test"].append(test_stats)

        tqdm.write(" ")
        tqdm.write(" ")

    # start attacking the trained model
    pgd = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
    fgsm = torchattacks.FGSM(model, eps=8/255)
    jitter = torchattacks.Jitter(model, eps=8/255, alpha=2/255, steps=10, scale=10, std=0.1, random_start=True)

    attacks = {"pgd": pgd, "fgsm": fgsm, "jitter": jitter}

    attacked_train = []
    attacked_test = []

    cc, ct = 0, 0
    ac, at = 0, 0

    for (images, labels) in train_loader:
        images = images.float().cuda()
        labels = labels.long().cuda()

        adv_images = attacks[args.attack_method](images, labels)
        attacked_train.append(adv_images.cpu())

        with torch.no_grad():
            clean_outs = model(images)
            adv_outs = model(adv_images)

            clean_preds = torch.argmax(clean_outs, 1)
            adv_preds = torch.argmax(adv_outs, 1)

            cc += (clean_preds == labels).sum().item()
            ct += labels.shape[0]

            ac += (adv_preds == labels).sum().item()
            at += labels.shape[0]

    clean_acc_train = round(cc/ct, 4)
    adv_acc_train = round(ac/at, 4)
    attacked_train = torch.cat(attacked_train, dim=0)

    cc, ct = 0, 0
    ac, at = 0, 0

    for (images, labels) in test_loader:
        images = images.float().cuda()
        labels = labels.long().cuda()

        adv_images = attacks[args.attack_method](images, labels)
        attacked_test.append(adv_images.cpu())

        with torch.no_grad():
            clean_outs = model(images)
            adv_outs = model(adv_images)

            clean_preds = torch.argmax(clean_outs, 1)
            adv_preds = torch.argmax(adv_outs, 1)

            cc += (clean_preds == labels).sum().item()
            ct += labels.shape[0]

            ac += (adv_preds == labels).sum().item()
            at += labels.shape[0]

    clean_acc_test = round(cc/ct, 4)
    adv_acc_test = round(ac/at, 4)
    attacked_test = torch.cat(attacked_test, dim=0)

    dump = {
            "train": attacked_train, "test": attacked_test, "model": model.state_dict(), 
            "clean_acc": {"train": clean_acc_train, "test": clean_acc_test}, "adv_acc":{"train": adv_acc_train, "test": adv_acc_test}
    }
    torch.save(dump, f"{args.attack_method}_samples_CIFAR10.pt")

    print(f"For ResNet-18")
    print(f"For attack method: {args.attack_method}")
    print(f"Clean Accuracy: {clean_acc_test}")
    print(f"Adversarial Accuracy: {adv_acc_test}")

if __name__ == "__main__":
    args = make_args()
    args.attack_method = "pgd"
    main(args)
    args.attack_method = "fgsm"
    main(args)
    args.attack_method = "jitter"
    main(args)
