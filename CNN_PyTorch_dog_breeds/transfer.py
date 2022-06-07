# EECS 545 Fall 2020
import torch
import torchvision.models as models
from dataset import DogCatDataset
from train import train


def load_pretrained(num_classes=2):
    # TODO (part d): load a pre-trained ResNet-18 model
    res18 = models.resnet18(pretrained=True)
    for par in res18.parameters():
        par.requires_grad = False

    res18.fc = torch.nn.Linear(512, num_classes)
    return res18


if __name__ == '__main__':
    config = {
        'dataset_path': 'data/images/dogs_vs_cats',
        'batch_size': 4,
        'ckpt_path': 'checkpoints/transfer',
        'plot_name': 'Transfer',
        'num_epoch': 10,
        'learning_rate': 1e-4,
    }
    dataset = DogCatDataset(config['batch_size'], config['dataset_path'])
    model = load_pretrained()
    train(config, dataset, model)
