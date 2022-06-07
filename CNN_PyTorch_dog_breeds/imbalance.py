# EECS 545 Fall 2020
from dataset import DogCatDataset
from train import evaluate_loop, train
from transfer import load_pretrained
import numpy as np



def per_class_accuracy(y_true, y_pred, num_classes=2):
    # TODO (part e): compute the per-class accuracy
    dog = 0.0
    cat = 0.0
    y_t = y_true.numpy()
    y_p = y_pred.numpy()
    n_d = np.sum(y_t)
    n_c = y_t.size - n_d
    for i in range(y_t.size):
        if(y_t[i] == 1 and y_p[i] == 1):
            dog += 1.0
        elif(y_t[i] == 0 and y_p[i] == 0):
            cat += 1.0
    dog = dog/n_d
    cat = cat/n_c
    return np.array([cat, dog])


def precision(y_true, y_pred):
    y_t = y_true.numpy()
    y_p = y_pred.numpy()
    dog = 0.0
    cat = 0.0
    for i in range(y_t.size):
        if (y_p[i] == 1 and y_t[i] == 1):
            dog += 1.0 #TP
        elif (y_p[i] == 1 and y_t[i] == 0):
            cat += 1.0 #FP
    return dog/(dog+cat)


def recall(y_true, y_pred):
    # TODO (part e): compute the recall
    y_t = y_true.numpy()
    y_p = y_pred.numpy()
    dog = 0.0
    cat = 0.0
    for i in range(y_t.size):
        if (y_p[i] == 1 and y_t[i] == 1):
            dog += 1.0
        elif (y_p[i] == 0 and y_t[i] == 1):
            cat += 1.0
    out = dog / (dog + cat)
    return out


def f1_score(y_true, y_pred):
    # TODO (part e): compute the f1-score
    recall_var = recall(y_true, y_pred)
    precision_var = precision(y_true, y_pred)
    f1 = 2*precision_var*recall_var/(precision_var+recall_var)
    return f1

def compute_metrics(dataset, model):
    y_true, y_pred, _ = evaluate_loop(dataset.val_loader, model)
    print('Per-class accuracy (cat,dog): ', per_class_accuracy(y_true, y_pred))
    print('Precision: ', precision(y_true, y_pred))
    print('Recall: ', recall(y_true, y_pred))
    print('F1-score: ', f1_score(y_true, y_pred))


if __name__ == '__main__':
    # model with normal cross-entropy loss
    config = {
        'dataset_path': 'data/images/dogs_vs_cats_imbalance',
        'batch_size': 4,
        'ckpt_path': 'checkpoints/imbalance',
        'ckpt_force': True,
        'plot_name': 'Imbalance',
        'num_epoch': 20,
        'learning_rate': 1e-4,
    }
    dataset = DogCatDataset(config['batch_size'], config['dataset_path'])
    model = load_pretrained()
    train(config, dataset, model)
    compute_metrics(dataset, model)

    # model with weighted cross-entropy loss
    config = {
        'ckpt_path': 'checkpoints/imbalance_weighted',
        'plot_name': 'Imbalance-Weighted',
        'num_epoch': 20,
        'learning_rate': 1e-4,
        'use_weighted': True,
    }
    model_weighted = load_pretrained()
    train(config, dataset, model_weighted)
    compute_metrics(dataset, model_weighted)
