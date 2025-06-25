class Vit(torch.nn.module)
  def __init__(self):
    super().__init__()
    self.cls = torch.nn.Parameter(torch.randn(1, 1, 1)) # (1, 1, 1) random gen cls token added. This is an addition to the 16 patches. now have 16 patches + 1 cls token.
    self.emb = torch.nn.Linear(196, 128) # this changes the dimension of the 16 patches from 196 to 128.
    self.pos = torch.nn.embedding(17,128) # this adds the positional encoding to the 16 patches. each token gets a positional embedding added.
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
  
class EncoderLayer(torch.nn.module):
  def __init__(self, dim):
    super().__init__()
    self.attn = Attention(dim) # this is the attention layer.
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
    
class FFN(torch.nn.module):
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

class Attention(torch.nn.module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim # this is the dimension of the input and output.
    self.q_proj = torch.nn.Linear(dim, dim) # this is the projection layer for the query.
    self.k_proj = torch.nn.Linear(dim, dim) # this is the projection layer for the key.
    self.v_proj = torch.nn.Linear(dim, dim) # this is the projection layer for the value.
    self.o_proj = torch.nn.Linear(dim, dim) # this is the projection layer for the output.
    self.drpout = torch.nn.Dropout(0.1) # this is the dropout layer.

    def forward(self, x):
      qry = self.q_proj(x) # [B, 17, 128] this is the output of the query projection layer.
      key = self.k_proj(x) # [B, 17, 128] this is the output of the key projection layer.
      val = self.v_proj(x) # [B, 17, 128] this is the output of the value projection layer.
      attn = qry @ key.transpose(-2, -1) * (self.dim ** -0.5) # [B, 17, 17] this is the output of the attention layer.
      attn = attn.softmax(dim=-1) # [B, 17, 17] this is the output of the softmax function. this is the attention weights.  
      attn = self.drpout(attn) # [B, 17, 17] this is the output of the dropout layer.
      out = torch.matmul(attn, val) # [B, 17, 128] this is the output of the output projection layer.
      return self.o_proj(out) # [B, 17, 128] this is the output of the output projection layer.

