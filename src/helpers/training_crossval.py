"""
Main training method performing cross validation.
k models are trained and validated simultaneously on the same batches.
"""

import torch
import numpy as np
from torch.autograd import Variable

def training(num_epochs, models, datasets, dataloaders, patch_size, k_fold, cuda=False):
    """
    Train the provided models for num_epochs, with the given criterion and optimizer, for the given data.
    k-fold cross validation is performed.
    """
    best_acc = 0.0
    best_model_wts = models[0]['model'].state_dict()
    
    acc = []
    
    print('Starting training and validation of the model...')
    for epoch in range(num_epochs):
        losses, corrects = [], []
        
        for k in range(k_fold):
            # Load k-th model, criterion, optimizer and dataloader
            fold_losses, fold_corrects = [], 0.0
            model = models[k]['model']
            criterion = models[k]['criterion']
            optimizer = models[k]['optimizer']
            dataloader = dataloaders[k]
            
            # Train and validate k_th model
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                for data in dataloader[phase]:
                    inputs, labels = data

                    if cuda: 
                        inputs, labels = Variable(inputs.cuda(1)), Variable(labels.cuda(1)) 
                    else: 
                        inputs, labels = Variable(inputs), Variable(labels)

                    # Forward
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        fold_losses.append(loss.data[0])
                    else:
                        # Convert float tensor to label prediction
                        preds_np = np.rint(outputs.squeeze().data.cpu().numpy())
                        # Convert float tensor labels to numpy labels
                        labels_np = labels.squeeze().data.cpu().numpy()

                        # Compare and compute accuracy
                        fold_corrects += np.sum(preds_np == labels_np)
                        
            losses.append(fold_losses)
            corrects.append(fold_corrects)
                
        print('Epoch {}/{} results: '.format(epoch + 1, num_epochs))
        sum_acc = 0
        for k in range(k_fold):
            acc = corrects[k] / (len(datasets[k]['val']) * patch_size**2)
            sum_acc += acc
            print('Fold {}  -  Mean loss: {:.4f}  -  Acc: {:f}%'.format(k, np.mean(losses[k]), acc*100))
        
        mean_acc = sum_acc / k_fold
        print('Mean accuracy: {:f}%'.format(mean_acc * 100))
        
        # Save one of the k models
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_model_wts = models[0]['model'].state_dict()

    return acc, best_model_wts