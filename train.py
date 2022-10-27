import torch
from tqdm import tqdm, tqdm_notebook
from torchmetrics.functional import precision_recall
from torch.nn import functional as F


def train_model(model, optimizer, loss_fn, train_loader, val_loader, epochs=10, device="cpu"):
    for epoch in range(epochs):
        training_loss = 0.0
        model.train()
        for batch in tqdm(train_loader):
            train_iterator = iter(train_loader)
            optimizer.zero_grad()
            inputs, target = batch
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item()
        training_loss /= len(train_iterator)
        print("training_loss_for_epoch:", training_loss)

        model.eval()
        valid_loss = 0.0
        num_correct = 0
        num_examples = 0
        preds = []
        list_label = []
        for batch in tqdm(val_loader):
            valid_iterator = iter(val_loader)
            inputs, target = batch
            inputs = inputs.to(device)
            output = model(inputs)
            target = target.to(device)
            loss = loss_fn(output, target)
            valid_loss += loss.data.item()
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], target)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
            preds.append(torch.max(F.softmax(output, dim=1), dim=1)[1])
            list_label.append(target)
        preds = torch.cat(preds)
        list_label = torch.cat(list_label)
        print(
            f"pr_rec:, {precision_recall(preds, list_label, average='macro', num_classes=6, mdmc_average='global')}")
        valid_loss /= len(valid_iterator)
        print(
            'Epoch: {},Training Loss:{:.2f},Validation Loss:{:.2f},accuracy = {:.2f}'.format(epoch, training_loss,
                                                                                             valid_loss,
                                                                                             num_correct / num_examples)
        )
