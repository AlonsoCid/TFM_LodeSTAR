import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

def clip_grad_training(model, train_dataloader, num_epochs=50, initial_lr=1e-3, 
                        device='cpu', patience=5, gamma=0.9, max_grad_norm=1.0, 
                        best_model=True, clip_grad=True):
    """
    Train a LodeSTAR model with early stopping based on loss improvement and exponential LR scheduler.
    
    Args:
    model (LodeSTAR): The LodeSTAR model to train.
    train_dataloader (DataLoader): DataLoader for the training data.
    num_epochs (int): Maximum number of epochs to train for.
    initial_lr (float): Initial learning rate for the optimizer.
    device (str): Device to train on ('cpu' or 'cuda').
    patience (int): Number of epochs to wait for improvement before stopping.
    gamma (float): Multiplicative factor of learning rate decay.
    max_grad_norm (float): Maximum norm for gradient clipping.
    best_model (bool): If True, save and load the best model state.
    clip_grad (bool): If True, apply gradient clipping.
    
    Returns:
    LodeSTAR: The trained model.
    """
    
    model = model.build().to(device)
    
    # Set up the PyTorch optimizer (Adam)
    optimizer = Adam(model.parameters(), lr=initial_lr)
    
    # Set up the exponential learning rate scheduler
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    
    best_loss = float('inf') if best_model else None
    epochs_without_improvement = 0
    
    # Save the initial state of the model and optimizer
    if best_model:
        best_model_state = model.state_dict()
        best_optimizer_state = optimizer.state_dict()
    best_epoch = 0
    best_batch = 0
    best_losses = {}
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
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
            
            # Gradient clipping
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Check for improvement
            if best_model and loss.item() < best_loss:
                best_loss = loss.item()
                best_model_state = model.state_dict()
                best_optimizer_state = optimizer.state_dict()
                best_epoch = epoch + 1
                best_batch = batch_idx + 1
                best_losses = {key: value[-1] for key, value in model.get_losses().items()}
        
        # Step the scheduler to update the learning rate
        scheduler.step()
        
        # Print epoch summary
        current_lr = optimizer.param_groups[0]['lr']
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Learning Rate: {current_lr:.6f}, Average Loss: {avg_loss:.4f}")
        
        # Print individual loss components
        for key, value in model.get_losses().items():
            if value:
                print(f"  {key}: {value[-1]:.4f}")
        
        # Check if early stopping criterion is met
        if best_model and best_epoch != epoch + 1:
            epochs_without_improvement += 1
        else:
            epochs_without_improvement = 0
        
        if epochs_without_improvement >= patience:
            print(f"Stopping early at epoch {epoch + 1} as no improvement in loss for {patience} consecutive epochs.")
            break
    
    if best_model:
        # Load the best model state
        model.load_state_dict(best_model_state)
        
        # Print the best epoch, batch, and losses
        print(f"Best model was from epoch {best_epoch}, batch {best_batch}")
        print(f"Best batch loss: {best_loss:.4f}")
        for key, value in best_losses.items():
            print(f"  {key}: {value:.4f}")
    
    return model

# Example usage:
# lodestar2_CEM = LodeSTAR(num_classes=3)
# trained_model = clip_grad_training(lodestar2_CEM, train_dataloader_CEM)