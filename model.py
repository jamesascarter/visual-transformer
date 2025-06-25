import torch

class Vit(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.cls = torch.nn.Parameter(torch.randn(1, 1, 128)) # (1, 1, 128) this is the cls token. B is the batch size.
    self.emb = torch.nn.Linear(196, 128) # this changes the dimension of the 16 patches from 196 to 128.
    self.pos = torch.nn.Embedding(17,128) # this adds the positional encoding to the 16 patches. each token gets a positional embedding added.
    self.register_buffer('rng', (torch.arange(17))) # this registers the positional encoding as a buffer.
    self.enc = torch.nn.ModuleList([EncoderLayer(128) for _ in range(6)]) # this creates a list of 6 encoder layers.
    self.fin = torch.nn.Sequential(
      torch.nn.LayerNorm(128),
      torch.nn.Linear(128, 10)
    ) # this is the final layer. it takes the output of the encoder and projects it to 10 classes.

  def forward(self, x):
    B = x.shape[0] # [B, 16, 196] this is 16 patches of 196 pixels each.
    pch = self.emb(x) # [B, 16, 128] this is the embedding of the 16 patches.
    cls = self.cls.expand(B, -1, -1) # [B, 1, 128] this is the embedding of the cls token.  
    hdn = torch.cat([cls, pch], dim=1) # [B, 17, 128] this is the embedding of the 16 patches and the cls token.
    hdn = hdn + self.pos(self.rng) # [B, 17, 128] this is the embedding of the 16 patches and the cls token with the positional encoding.
    for enc in self.enc: hdn = enc(hdn) # [B, 17, 128] this is the embedding of the 16 patches and the cls token with the positional encoding after the encoder layers.
    out = hdn[:, 0, :] # [B, 128] this is the embedding of the cls token after the encoder layers.
    return self.fin(out) # [B, 10] this is the output of the final layer.
  
class EncoderLayer(torch.nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.att = MultiHeadAttention(dim) # this is the attention layer.
    self.ffn = FFN(dim) # this is the feedforward layer.
    self.ini = torch.nn.LayerNorm(dim) # this is the initial layer normalization.
    self.fin = torch.nn.LayerNorm(dim) # this i s the final layer normalization.

  def forward(self, src):
    out = self.att(src) # [B, 17, 128] this is the output of the attention layer.
    src = src + out # [B, 17, 128] this is the output of the attention layer added to the input.
    src = self.ini(src) # [B, 17, 128] this is the output of the initial layer normalization.
    out = self.ffn(src) # [B, 17, 128] this is the output of the feedforward layer.
    src = src + out # [B, 17, 128] this is the output of the feedforward layer added to the input.
    src = self.fin(src) # [B, 17, 128] this is the output of the final layer normalization.
    return src
    
class FFN(torch.nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.one = torch.nn.Linear(dim, dim)
    self.drp = torch.nn.Dropout(0.1)
    self.rlu = torch.nn.ReLU(inplace=True)
    self.two = torch.nn.Linear(dim, dim)

  def forward(self, x):
    x = self.one(x) # [B, 17, 128] this is the output of the first linear layer.
    x = self.rlu(x) # [B, 17, 128] this is the output of the ReLU activation function.
    x = self.drp(x) # [B, 17, 128] this is the output of the dropout layer.
    x = self.two(x) # [B, 17, 128] this is the output of the second linear layer.
    return x

class MultiHeadAttention(torch.nn.Module):
  def __init__(self, dim, num_heads=8):
    super().__init__()
    self.dim = dim # this is the dimension of the input and output.
    self.num_heads = num_heads # this is the number of heads.
    self.head_dim = dim // self.num_heads # this is the dimension of the input and output.

    self.q_proj = torch.nn.Linear(dim, dim) # this is the projection layer for the query.
    self.k_proj = torch.nn.Linear(dim, dim) # this is the projection layer for the key.
    self.v_proj = torch.nn.Linear(dim, dim) # this is the projection layer for the value.
    self.o_proj = torch.nn.Linear(dim, dim) # this is the projection layer for the output.
    self.dropout = torch.nn.Dropout(0.1)

  def forward(self, x):
    B, N, C = x.shape # [B, 17, 128] this is the shape of the input. B is the batch size, N is the number of tokens, C is the dimension of the input.

      # Project to Q, K, V
    qry = self.q_proj(x)  # [B, 17, 128]
    key = self.k_proj(x)  # [B, 17, 128]
    val = self.v_proj(x)  # [B, 17, 128]
    
    # Reshape for multi-head: [B, seq_len, num_heads, head_dim]
    qry = qry.view(B, N, self.num_heads, self.head_dim)
    key = key.view(B, N, self.num_heads, self.head_dim)
    val = val.view(B, N, self.num_heads, self.head_dim)
    
    # Transpose for attention: [B, num_heads, seq_len, head_dim]
    qry = qry.transpose(1, 2)  # [B, 8, 17, 16]
    key = key.transpose(1, 2)  # [B, 8, 17, 16]
    val = val.transpose(1, 2)  # [B, 8, 17, 16]


      # Compute attention scores
    attn = qry @ key.transpose(-2, -1) * (self.head_dim ** -0.5)  # [B, 8, 17, 17]
    attn = attn.softmax(dim=-1)
    attn = self.dropout(attn)
    
    # Apply attention to values
    out = attn @ val  # [B, 8, 17, 16]
    
    # Concatenate heads: [B, seq_len, dim]
    out = out.transpose(1, 2).contiguous().view(B, N, C)
    
    return self.o_proj(out)
  
