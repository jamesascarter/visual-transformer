#
#
#
import torch
import dataset
import model
import glob
import os


#
#
#
torch.manual_seed(47)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#
#
#
vit = model.Vit().to(dev)

# Find all checkpoint files
checkpoint_files = glob.glob('./checkpoints/*.vit.pth')

# Get the latest one by modification time
latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)

# Load the latest checkpoint
vit.load_state_dict(torch.load(latest_checkpoint))
print(f"Loaded checkpoint: {latest_checkpoint}")
vit.eval()


#
#
#
ds = dataset.Classify(train=False)
dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)


#
#
#
correct = 0
total = 0

with torch.no_grad():
    for ptch, lbls in dl:
        ptch = torch.flatten(ptch, start_dim=2).to(dev)
        lbls = lbls.to(dev)
        
        outs = vit(ptch)  # [B, 4, 10]
        
        # Check if ALL 4 digits are correct
        all_correct = torch.ones(lbls.size(0), dtype=torch.bool, device=dev)
        for i in range(4):
            _, predicted = torch.max(outs[:, i, :], 1)
            print(predicted, lbls[:, i])
            all_correct &= (predicted == lbls[:, i])  # All digits must be correct
        
        correct += all_correct.sum().item()
        total += lbls.size(0)

accuracy = 100 * correct / total
print(f"All-digits-correct accuracy: {accuracy:.2f}%")