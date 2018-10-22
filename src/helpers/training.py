"""
Main training function.
"""
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import f1_score

def training(num_epochs, model, criterion, optimizer, lr_scheduler, datasets, dataloaders, patch_size, validate=True, cuda=False, gpu_idx=0):
    """
    Train the provided models for num_epochs, with the given criterion and optimizer, for the given data.
    Validating training for each epoch is optional. If validation is enabled, best model based on validation accuracy is loaded in the model.
    """
    best_score = 0.0
    best_model_wts = model.state_dict()
    
    scores, train_loss, val_loss = [], [], []
    phases = ['train', 'val'] if validate else ['train']
    
    print('Starting training and validation of the model...')
    for epoch in range(num_epochs):
        epoch_loss_train, epoch_loss_val, f1score_sum = [], [], 0.0  
        
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            for data in dataloaders[phase]:
                # Load batch
                inputs, labels = data
                
                if cuda: 
                    inputs, labels = Variable(inputs.cuda(gpu_idx)), Variable(labels.cuda(gpu_idx)) 
                else: 
                    inputs, labels = Variable(inputs), Variable(labels)

                # Forward and compute loss
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_loss_train.append(loss.data[0])
                else:
                    # Convert float tensor to label prediction
                    preds_np = np.rint(outputs.squeeze().data.cpu().numpy())
                    # Convert float tensor labels to numpy labels
                    labels_np = labels.squeeze().data.cpu().numpy()
                    
                    # Compare and compute f1-score
                    f1score_sum += f1_score(labels_np.ravel(), preds_np.ravel(), average='micro')
                    epoch_loss_val.append(loss.data[0])
                
        if validate:
            # Validate epoch results
            epoch_score = f1score_sum / len(datasets['val'])
            scores.append(epoch_score)
            train_loss.append(np.mean(epoch_loss_train))
            val_loss.append(np.mean(epoch_loss_val))
            progress_str = '[epoch {}/{}] - Valid Loss: {:.4f} Valid score: {:f}%'.format(epoch + 1, num_epochs, np.mean(epoch_loss_val), epoch_score*100)
            
            if epoch_score > best_score:
                best_score = epoch_score
                best_model_wts = model.state_dict()
                
        else:
            progress_str = '[epoch {}/{}]'.format(epoch + 1, num_epochs)
            
        print(progress_str)
        
        # Adjust learning rate
        lr_scheduler.step(int(np.mean(epoch_loss_train) * 1000))
    
    # Load best model if validate mode enabled
    if validate:
        model.load_state_dict(best_model_wts)
        
    return scores, train_loss, val_loss, best_model_wts