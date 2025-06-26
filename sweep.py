# sweep.py
import wandb
import torch
from tqdm import tqdm
from dataset import Classify
from model import Vit

def train_sweep():
    # Initialize wandb with sweep config
    wandb.init()
    
    # Get hyperparameters from sweep
    config = wandb.config
    lr = config.learning_rate
    batch_size = config.batch_size
    epochs = config.epochs
    
    # Setup device
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup data
    ds = Classify()
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    # Setup model
    vit = Vit().to(dev)
    opt = torch.optim.Adam(vit.parameters(), lr=lr)
    crt = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        prgs = tqdm(dl, desc=f"Epoch {epoch + 1}", leave=False)
        for idx, (ptch, lbls) in enumerate(prgs):
            ptch = torch.flatten(ptch, start_dim=2).to(dev)
            lbls = lbls.to(dev)
            
            outs = vit(ptch)
            opt.zero_grad()
            loss = crt(outs, lbls)
            loss.backward()
            opt.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outs.data, 1)
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()
            epoch_loss += loss.item()
            
            # Update progress bar
            prgs.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
        
        # Calculate metrics
        avg_loss = epoch_loss / len(dl)
        accuracy = 100 * correct / total
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'loss': avg_loss,
            'accuracy': accuracy,
            'learning_rate': lr,
            'batch_size': batch_size
        })
        
        # Save best model
        if epoch == epochs - 1:
            wandb.run.summary['best_accuracy'] = accuracy
            wandb.run.summary['best_loss'] = avg_loss

def main():
    # Sweep configuration
    sweep_configuration = {
        'method': 'grid',
        'name': 'lr-batch-size-sweep',
        'metric': {
            'name': 'accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'values': [0.0001, 0.001, 0.01]
            },
            'batch_size': {
                'values': [32, 64, 128, 256]
            },
            'epochs': {
                'value': 5
            }
        }
    }
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_configuration, project="visual-transformer-sweep")
    
    # Start the sweep
    print(f"Starting sweep with ID: {sweep_id}")
    wandb.agent(sweep_id, function=train_sweep, count=12)

if __name__ == "__main__":
    main()