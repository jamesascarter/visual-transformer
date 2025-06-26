import torch
import torchvision as tv
import matplotlib.pyplot as plt
import random

torch.manual_seed(47)
random.seed(47)

NUMBER_DIGITS = 4
PATCH_SIZE = 28

class Classify(torch.utils.data.Dataset):
    def __init__(self, train=True):
        super().__init__()
        self.tf = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.ds = tv.datasets.MNIST(root='.', train=train, download=True)
        self.ln = len(self.ds)

    def __len__(self):
        return self.ln

    def __getitem__(self, idx):
        # Create a 112x112 composite image with 4 digits (each 56x56)
        composite_img = torch.zeros(112, 112)
        digit_labels = []
        digit_images = []

        for i in range(NUMBER_DIGITS):
            rand_idx = random.randint(0, self.ln - 1)
            img, label = self.ds[rand_idx]
            tnsrs = self.tf(img.resize((56, 56))).squeeze()
            
            digit_images.append(tnsrs)
            
            # Place each digit in a different quadrant
            row_start = (i // 2) * 56
            col_start = (i % 2) * 56
            composite_img[row_start:row_start+56, col_start:col_start+56] = tnsrs
            
            digit_labels.append(label)

        # Break into 16 patches of 28x28
        patches = []
        patch_positions = []
        
        for i in range(4):  # rows
            for j in range(4):  # columns
                row_start = i * 28
                col_start = j * 28
                patch = composite_img[row_start:row_start+28, col_start:col_start+28]
                patches.append(patch.flatten())
                patch_positions.append((i, j, row_start, col_start))
        
        flattened = torch.stack(patches)
        target = torch.tensor(digit_labels, dtype=torch.long)
        return flattened, target, patch_positions, digit_images


if __name__ == "__main__":
    dc = Classify()
    flattened, digit_labels, patch_positions, digit_images = dc[0]
    print('flattened', flattened.shape)
    print('digit_labels', digit_labels)
    
    # Create the test image using the SAME digits from __getitem__
    composite_img = torch.zeros(112, 112)
    for i in range(NUMBER_DIGITS):
        # Use the same digit images that were generated in __getitem__
        row_start = (i // 2) * 56
        col_start = (i % 2) * 56
        composite_img[row_start:row_start+56, col_start:col_start+56] = digit_images[i]
    
    # Denormalize for visualization
    composite_img = composite_img * 0.3081 + 0.1307
    composite_img = torch.clamp(composite_img, 0, 1)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(composite_img.numpy(), cmap='gray')
    ax.set_title(f'Test Image with 4 Digits: {digit_labels.tolist()}\nShowing how 112×112 image is chopped into 16 patches of 28×28 each', fontsize=14)
    ax.axis('off')
    
    # Add digit boundaries (red lines)
    ax.axhline(y=56, color='red', linewidth=3, alpha=0.8)
    ax.axvline(x=56, color='red', linewidth=3, alpha=0.8)
    
    # Add digit labels
    ax.text(28, 28, f'Digit 0\n({digit_labels[0]})', ha='center', va='center', 
            color='red', fontsize=14, weight='bold', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    ax.text(84, 28, f'Digit 1\n({digit_labels[1]})', ha='center', va='center', 
            color='red', fontsize=14, weight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    ax.text(28, 84, f'Digit 2\n({digit_labels[2]})', ha='center', va='center', 
            color='red', fontsize=14, weight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    ax.text(84, 84, f'Digit 3\n({digit_labels[3]})', ha='center', va='center', 
            color='red', fontsize=14, weight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    # Add patch boundaries (blue lines)
    for i in range(1, 4):
        ax.axhline(y=i*28, color='blue', linewidth=2, alpha=0.6)
        ax.axvline(x=i*28, color='blue', linewidth=2, alpha=0.6)
    
    # Add patch numbers
    for i in range(4):
        for j in range(4):
            patch_idx = i * 4 + j
            row_start = i * 28 + 14  # Center of patch
            col_start = j * 28 + 14
            ax.text(col_start, row_start, str(patch_idx), 
                    ha='center', va='center', color='yellow', 
                    fontsize=12, weight='bold', 
                    bbox=dict(boxstyle="circle,pad=0.3", facecolor='black', alpha=0.7))
    
    # Add legend
    ax.text(0.02, 0.98, 'Red lines: Digit boundaries (56×56 each)\nBlue lines: Patch boundaries (28×28 each)\nYellow numbers: Patch order (0-15)', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.show()
    
    print(f"✓ Digit labels: {digit_labels.tolist()}")
    print(f"✓ Flattened tensor shape: {flattened.shape}")
    print(f"✓ Each patch is 28×28 = 784 pixels")
    print(f"✓ Model expects input of shape [B, 16, 784]")
    