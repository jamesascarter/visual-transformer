#
#
#
import torch
import torchvision as tv
import matplotlib.pyplot as plt
import random
import einops

#
#
#
torch.manual_seed(47)
random.seed(47)

NUMBER_DIGITS = 4
#
#
#
class Classify(torch.utils.data.Dataset):
  def __init__(self, train=True):
    super().__init__()
    self.tf = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))])
    self.ds = tv.datasets.MNIST(root='.', train=train, download=True)
    self.ti = tv.transforms.ToPILImage()
    self.ln = len(self.ds)

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, idx):
    all_patches = []
    digit_labels = []

    for _ in range(NUMBER_DIGITS):
      rand_idx = random.randint(0, self.ln - 1)
      img, label = self.ds[rand_idx]
      tnsrs = self.tf(img.resize((56, 56))).squeeze()
      patch = einops.rearrange(tnsrs, '(h ph) (w pw) -> (h w) ph pw', ph=14, pw=14)
      all_patches.append(patch)
      digit_labels.append(label)

    composite_patch = torch.cat(all_patches, dim=2) # [16, 14, 56] concat all patches
    flattened = einops.rearrange(composite_patch, 'p h w -> p (h w)') # [16, 196] flatten the composite patch
    target = torch.tensor(digit_labels, dtype=torch.long)
    return flattened, target


#
#
#
if __name__ == "__main__":

  dc = Classify()
  flattened, digit_labels = dc[0]
  print('flattened', flattened.shape)
  print('digit_labels', digit_labels)
