import torch
from torch.optim import Adam, RMSprop
from tqdm import tqdm

# def multi_step_halted_training(model, train_dataloader, num_epochs=50, initial_lr=1e-4, 
#                                device='cpu', optimizer_class=Adam, loss_threshold=0.09,
#                                milestones=[100], gamma=1):
#     """
#     Train a LodeSTAR model with multi-step learning rate and early stopping.
    
#     Args:
#     model (LodeSTAR): The LodeSTAR model to train.
#     train_dataloader (DataLoader): DataLoader for the training data.
#     num_epochs (int): Maximum number of epochs to train for.
#     initial_lr (float): Initial learning rate for the optimizer.
#     device (str): Device to train on ('cpu' or 'cuda').
#     optimizer_class (torch.optim.Optimizer): The optimizer class to use.
#     loss_threshold (float): Threshold for early stopping.
#     milestones (list): Epochs at which to reduce the learning rate.
#     gamma (float): Factor by which to reduce the learning rate.
    
#     Returns:
#     LodeSTAR: The trained model.
#     """
    
#     model = model.build().to(device)
    
#     # Set up the PyTorch optimizer
#     optimizer = optimizer_class(model.parameters(), lr=initial_lr)
    
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
        
#         # Adjust learning rate
#         lr = initial_lr
#         for milestone in milestones:
#             if epoch >= milestone:
#                 lr *= gamma
#             else:
#                 break
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
        
#         print(f"Epoch {epoch+1}/{num_epochs}, Learning Rate: {lr:.6f}")
        
#         progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
#         for batch in progress_bar:
#             # Preprocess the batch
#             (x, class_label), inverse = model.train_preprocess(batch)
#             x, class_label = x.to(device), class_label.to(device)
            
#             # Forward pass
#             y_hat = model((x, class_label))
            
#             # Compute loss
#             loss_dict = model.compute_loss(y_hat, inverse)
#             loss = sum(loss_dict.values())
            
#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
            
#             # Update progress bar
#             progress_bar.set_postfix({'loss': loss.item()})
        
#         # Print epoch summary
#         avg_loss = total_loss / len(train_dataloader)
#         print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
#         # Print individual loss components
#         for key, value in model.get_losses().items():
#             if value:
#                 print(f"  {key}: {value[-1]:.4f}")
        
#         # Get the current losses
#         current_losses = model.get_losses()
#         latest_between_loss = current_losses['between_image_disagreement'][-1]
#         latest_within_loss = current_losses['within_image_disagreement'][-1]
#         latest_mask_loss = current_losses['mask_loss'][-1]
        
#         # Check if the last values of all three losses are below the threshold
#         if latest_between_loss <= loss_threshold and latest_within_loss <= loss_threshold and latest_mask_loss <= loss_threshold:
#             print(f"Stopping early at epoch {epoch + 1} as all losses are below the threshold {loss_threshold}.")
#             break
    
#     return model

def multi_step_halted_training(model, train_dataloader, num_epochs=50, initial_lr=1e-4, 
                               device='cpu', optimizer_class=Adam, loss_threshold=0.09,
                               milestones=[100], gamma=1):
    """
    Train a LodeSTAR model with multi-step learning rate and early stopping.
    
    Args:
    model (LodeSTAR): The LodeSTAR model to train.
    train_dataloader (DataLoader): DataLoader for the training data.
    num_epochs (int): Maximum number of epochs to train for.
    initial_lr (float): Initial learning rate for the optimizer.
    device (str): Device to train on ('cpu' or 'cuda').
    optimizer_class (torch.optim.Optimizer): The optimizer class to use.
    loss_threshold (float): Threshold for early stopping.
    milestones (list): Epochs at which to reduce the learning rate.
    gamma (float): Factor by which to reduce the learning rate.
    
    Returns:
    LodeSTAR: The trained model.
    """
    
    model = model.build().to(device)
    
    # Set up the PyTorch optimizer
    optimizer = optimizer_class(model.parameters(), lr=initial_lr)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Adjust learning rate
        lr = initial_lr
        for milestone in milestones:
            if epoch >= milestone:
                lr *= gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        print(f"Epoch {epoch+1}/{num_epochs}, Learning Rate: {lr:.6f}")
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Preprocess the batch
            (x, class_label), inverse = model.train_preprocess(batch)
            x, class_label = x.to(device), class_label.to(device)
            
            # Forward pass
            y_hat = model((x, class_label))
            
            # Compute loss
            loss_dict = model.compute_loss(y_hat, inverse)
            loss = sum(loss_dict.values())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Get the current losses
            current_losses = model.get_losses()
            latest_between_loss = current_losses['between_image_disagreement'][-1]
            latest_within_loss = current_losses['within_image_disagreement'][-1]
            latest_mask_loss = current_losses['mask_loss'][-1]
            
            # Check if the last values of all three losses are below the threshold
            if (latest_between_loss <= loss_threshold and
                latest_within_loss <= loss_threshold and
                latest_mask_loss <= loss_threshold):
                print(f"Stopping early at epoch {epoch + 1}, iteration {len(progress_bar)} as all losses are below the threshold {loss_threshold}.")
                return model
        
        # Print epoch summary
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Print individual loss components
        for key, value in model.get_losses().items():
            if value:
                print(f"  {key}: {value[-1]:.4f}")
    
    return model

# Example usage:
# model = LodeSTAR(num_classes=3, n_transforms=4)
# trained_model = multi_step_halted_training(model, train_dataloader, 
#                                            num_epochs=50, initial_lr=1e-4, 
#                                            device='cpu', optimizer_class=Adam, 
#                                            loss_threshold=0.09, milestones=[20, 40], gamma=0.5)