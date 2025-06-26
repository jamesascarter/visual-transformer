#
#
#
#
import tqdm
import torch
import datetime
import dataset
import model
import wandb


#
#
#
#
torch.manual_seed(42)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
torch.os.makedirs('./checkpoints', exist_ok=True)


#
#
#
#
vit = model.Vit().to(dev)
torch.save(vit.state_dict(), f'./checkpoints/{ts}.0.vit.pth')
print('vit:', sum(p.numel() for p in vit.parameters())) # 626,570
opt = torch.optim.Adam(vit.parameters(), lr=0.001)
crt = torch.nn.CrossEntropyLoss()
wandb.init(
  project='mlx8-week-03-vit',
  name='real'
)


#
#
#
#
ds = dataset.Classify()
dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)


#
#
#
#
for epoch in range(10):
  prgs = tqdm.tqdm(dl, desc=f"Epoch {epoch + 1}", leave=False)
  for idx, (ptch, lbls) in enumerate(prgs):
    ptch = torch.flatten(ptch, start_dim=2).to(dev)
    lbls = lbls.to(dev)
    outs = vit(ptch)
    opt.zero_grad()
    loss = 0
    for i in range(4):
        loss += crt(outs[:, i, :], lbls[:, i])  # Loss for each digit
    loss = loss / 4  # Average loss
    loss.backward()
    opt.step()
    wandb.log({'loss': loss.item()})
  torch.save(vit.state_dict(), f'./checkpoints/{ts}.{epoch + 1}.vit.pth')


#
#
#
#
wandb.finish()