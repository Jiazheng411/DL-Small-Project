import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Generator, List, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch import nn, optim


class ProgressBar():
    """
    Keras style progress bar
    with time estimation
    and customizable content
    """

    def __init__(self) -> None:
        super().__init__()
        self.content = ''
        self.last_item = False

    def _render_bar(self, i: int, total: int, width: int = 30) -> str:
        total_str = str(total)
        head = f'{str(i).rjust(len(total_str))}/{total_str} ['
        finished = int(i / total * width)
        if finished >= width:
            bar = '=' * width
        else:
            bar = '=' * finished + '>' + '.' * (width - finished - 1)
        return head + bar + '] '

    def set_content(self, content: str) -> None:
        """ Set custom content displayed after progress bar """
        if content:
            self.content = ' - ' + content
            return
        self.content = ''

    def __call__(self, lst: List[Any]) -> Generator[Any, None, None]:
        """ Iterate through the given list with real time progress bar """
        times_cost = []
        t_before = time.time()
        self.last_item = False
        for i, item in enumerate(lst):
            # estimate remaining time
            eta = '?'
            if len(times_cost) > 0:
                time_per_step = sum(times_cost)/len(times_cost)
                eta = round(time_per_step * (len(lst) - i))
            # update progress bar
            sys.stdout.write(
                f'\r{self._render_bar(i, len(lst))}- ETA: {eta}s{self.content}')
            sys.stdout.flush()
            if i == len(lst) - 1:
                self.last_item = True
            yield item
            # record time
            t_after = time.time()
            times_cost.append(t_after - t_before)
            t_before = t_after
        # show total time and time per step
        total_time = round(sum(times_cost))
        time_per_step = round(sum(times_cost)/len(times_cost) * 1000)
        sys.stdout.write(
            f'\r{self._render_bar(len(lst), len(lst))}'
            f'- {total_time}s {time_per_step}ms/step{self.content}\n')
        sys.stdout.flush()


def _is_binary(outputs):
    return len(outputs.shape) == 1 or outputs.shape[1] == 1


def _do_criterion(criterion, outputs, labels):
    if _is_binary(outputs):
        outputs = outputs.view(-1)
        return criterion(outputs, labels * 1.0)
    return criterion(outputs, labels)


def _do_prediction(outputs):
    if _is_binary(outputs):
        outputs = outputs.view(-1)
        return (outputs > 0.5) * 1
    return outputs.max(dim=1)[1]


def validate_model(model, validloader, criterion, device='cuda'):
    model.to(device)
    model.eval()

    test_loss = 0
    accuracy = 0
    total = 0

    with torch.no_grad():
        for images, labels in validloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model.forward(images)
            test_loss += _do_criterion(criterion, outputs, labels).item()

            equality = (labels.data == _do_prediction(outputs))
            accuracy += equality.type(torch.FloatTensor).sum()
            total += len(images)

    return float(test_loss/len(validloader)), float(accuracy/total)


def save_model(model: nn.Module, checkpoint_path: str):
    torch.save(model.state_dict(), checkpoint_path)


def load_model(model: nn.Module, checkpoint_path: str) -> nn.Module:
    model.load_state_dict(torch.load(checkpoint_path))
    return model


def _running_average(data, last_n):
    window = data[-last_n:]
    return sum(window) / len(window)


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    trainloader: torch.utils.data.DataLoader,
    validloader: torch.utils.data.DataLoader,
    epochs: int,
    device: str = 'cuda',
    valid_every: Union[int, str] = 'epoch'
):
    session = datetime.now().strftime('%Y-%m-%d_%H-%M')
    checkpoints_dir = os.path.join('checkpoints', session)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    model.to(device)

    if valid_every == 'epoch':
        valid_every = len(trainloader)
    else:
        valid_every = int(valid_every)

    steps = 0
    train_loss_history = []
    train_accuracy_history = []
    valid_loss_history = []
    valid_accuracy_history = []

    for e in range(epochs):
        print(f'Epoch: {e+1}/{epochs} @ {datetime.now()}')
        model.train()
        validated = False

        progressbar = ProgressBar()
        for images, labels in progressbar(trainloader):
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = _do_criterion(criterion, outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_history.append(loss.item())
            equality = (labels.data == _do_prediction(outputs))
            accuracy = equality.type(torch.FloatTensor).mean()
            train_accuracy_history.append(float(accuracy))

            if steps % valid_every == 0:
                # Eval mode for predictions
                model.eval()

                # Turn off gradients for validation
                with torch.no_grad():
                    valid_loss, valid_accuracy = validate_model(
                        model, validloader, criterion, device)

                valid_loss_history.append(valid_loss)
                valid_accuracy_history.append(valid_accuracy)

                # Make sure training is back on
                model.train()
                validated = True

            # Update progress bar
            avg_loss = _running_average(train_loss_history, valid_every)
            avg_accuracy = _running_average(train_accuracy_history, valid_every)
            content = [
                'loss: {:.3f}'.format(avg_loss),
                'accuracy: {:.3f}'.format(avg_accuracy)
            ]
            if validated:
                content.extend([
                    'val_loss: {:.3f}'.format(valid_loss_history[-1]),
                    'val_accuracy: {:.3f}'.format(valid_accuracy_history[-1])
                ])
            progressbar.set_content(' - '.join(content))

            # Save checkpoint
            if progressbar.last_item:
                checkpoint_path = os.path.join(checkpoints_dir, f'ckpt_{e:02d}.pth')
                model_states = model.state_dict()
                torch.save(model_states, checkpoint_path)
                progressbar.content += f' - saved to {checkpoint_path}'

    history = {
        'valid_every': valid_every,
        'total_steps': steps,
        'train_loss_history': train_loss_history,
        'train_accuracy_history': train_accuracy_history,
        'valid_loss_history': valid_loss_history,
        'valid_accuracy_history': valid_accuracy_history,
    }
    return history


def _plot_history(history, ax_loss, ax_accuracy, label_suffix='', index=0):
    valid_every = history['valid_every']
    train_loss_history = history['train_loss_history']
    valid_loss_history = history['valid_loss_history']
    train_accuracy_history = history['train_accuracy_history']
    valid_accuracy_history = history['valid_accuracy_history']

    train_steps = np.array(range(valid_every, len(train_loss_history) + 1)) / valid_every
    valid_steps = (np.array(range(len(valid_loss_history))) + 1)

    train_loss_history = np.convolve(
        train_loss_history, np.ones(valid_every)/valid_every, mode='valid'
    )
    train_accuracy_history = np.convolve(
        train_accuracy_history, np.ones(valid_every)/valid_every, mode='valid'
    )

    color = list(mcolors.TABLEAU_COLORS)[index]
    ax_loss.plot(train_steps, train_loss_history, label='training loss'+label_suffix, c=color)
    ax_loss.plot(valid_steps, valid_loss_history, '--',
                 label='validation loss'+label_suffix, c=color)
    ax_accuracy.plot(train_steps, train_accuracy_history,
                     label='training accuracy'+label_suffix, c=color)
    ax_accuracy.plot(valid_steps, valid_accuracy_history, '--',
                     label='validation accuracy'+label_suffix, c=color)


def plot_history(history: Dict[str, Any]):
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    _plot_history(history, ax1, ax2)
    ax1.legend()
    ax2.legend()
    plt.show()


def plot_histories(histories: Dict[str, Dict[str, Any]], loss_ylim=(0.0, 0.8)):
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(211)
    if loss_ylim:
        ax1.set_ylim(loss_ylim)
    ax2 = fig.add_subplot(212)
    for i, (annotation, history) in enumerate(histories.items()):
        _plot_history(history, ax1, ax2, f' ({annotation})', i)
    ax1.legend()
    ax2.legend()
    plt.show()


def plot_confusion_matrix(model, dataloader, classes, device='cuda'):
    # Predict with model
    model.to(device)
    model.eval()

    y_pred = torch.zeros(0, dtype=torch.long, device='cpu')
    y_true = torch.zeros(0, dtype=torch.long, device='cpu')

    with torch.no_grad():
        for images, true_labels in dataloader:
            outputs = model(images.to(device))
            pred_labels = _do_prediction(outputs)

            # Append batch prediction results
            y_pred = torch.cat([y_pred, pred_labels.view(-1).cpu()])
            y_true = torch.cat([y_true, true_labels.view(-1).cpu()])

    # Confusion matrix
    conf_mat = confusion_matrix(y_true.numpy(), y_pred.numpy())

    # Overall metrices
    print(f'accuracy: {conf_mat.diagonal().sum()/conf_mat.sum()*100:.2f}%')
    print()

    # Per-class metrices
    recalls = conf_mat.diagonal() / conf_mat.sum(1)
    precisions = conf_mat.diagonal() / conf_mat.sum(0)
    f1s = 2 * (precisions * recalls) / (precisions + recalls)
    print(pd.DataFrame({
        'recall': recalls,
        'precisions': precisions,
        'f1': f1s
    }, index=classes))
    print()

    # Plot matrix
    conf_df = pd.DataFrame(conf_mat, classes, classes)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    sns.heatmap(
        conf_df, ax=ax,
        cmap=plt.cm.Blues,
        annot=True,
        fmt='d',
        annot_kws={'size': 14})
    ax.set(ylabel='True label',
           xlabel='Predicted label')
    plt.show()


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(image.squeeze(0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_title(title)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    return ax


def plot_test_results(model, dataloader, classes, cols=4, max_rows=10, device='cuda'):
    # Predict with model
    model.to(device)
    model.eval()

    true_labels = torch.zeros(0, dtype=torch.long, device='cpu')
    pred_labels = torch.zeros(0, dtype=torch.long, device='cpu')
    images = []

    with torch.no_grad():
        for batch_images, batch_true_labels in dataloader:
            outputs = model(batch_images.to(device))
            batch_pred_labels = _do_prediction(outputs)
            pred_labels = torch.cat([pred_labels, batch_pred_labels.view(-1).cpu()])
            true_labels = torch.cat([true_labels, batch_true_labels.view(-1).cpu()])
            for image in batch_images:
                if len(images) >= cols * max_rows:
                    break
                images.append(image)

    # Sort result
    results = list(zip(true_labels, pred_labels, images))
    results.sort(key=lambda r: r[0])

    # Plot images and results
    rows = math.ceil(len(images) / cols)
    fig = plt.figure(figsize=(cols*2, rows*2.5))

    for i, (true_label, pred_label, image) in enumerate(results):
        ax = fig.add_subplot(rows, cols, i + 1)
        imshow(image, ax,
               f'true:{classes[true_label]}\n' +
               f'pred:{classes[pred_label]}')

    total = len(true_labels)
    correct = int(((true_labels == pred_labels) * 1).sum())
    accuracy = correct / total * 100
    fig.suptitle(
        f'Average performance {correct}/{total} = {accuracy:.1f}%',
        y=0.93)
    plt.show()


def visualize_model_feature_maps(model, dataloader, classes, cols=8, device='cuda'):
    # select a single image as the batch
    images, labels = next(iter(dataloader))
    images = images[[0]]
    labels = labels[[0]]

    # run model
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        preds = _do_prediction(outputs).cpu()

    # visualize original image
    print(f'Original image (true: {classes[labels[0]]} pred: {classes[preds[0]]})')
    imshow(images[0])
    plt.show()

    # visualize feature maps
    for i, layer_feature_maps in enumerate(model.feature_maps):
        print(f'Layer {i + 1}')

        layer_feature_maps = layer_feature_maps[0]

        rows = math.ceil(len(layer_feature_maps) / cols)
        fig = plt.figure(figsize=(cols*2, rows*2))

        for i, feature_map in enumerate(layer_feature_maps):
            ax = fig.add_subplot(rows, cols, i + 1)
            imshow(feature_map.cpu(), ax)

        plt.show()
