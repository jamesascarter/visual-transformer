import torch

class Vit(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.cls = torch.nn.Parameter(torch.randn(1, 1, 128)) # (1, 1, 128) this is the cls token. B is the batch size.
    self.emb = torch.nn.Linear(784, 128) # this changes the 784 pixels to 128 dimensions.
    self.pos = torch.nn.Embedding(17,128) # this adds the positional encoding to the 16 patches. each token gets a positional embedding added.
    self.register_buffer('rng', (torch.arange(17))) # this registers the positional encoding as a buffer.
    self.enc = torch.nn.ModuleList([EncoderLayer(128) for _ in range(6)]) # this creates a list of 6 encoder layers.
    
    self.decoder = Decoder(128, num_layers=6)
    self.start_token = torch.nn.Parameter(torch.randn(128))
    self.digit_tokens = torch.nn.Parameter(torch.randn(4, 128))
    self.decoder_pos = torch.nn.Embedding(5, 128)  # START + 4 digits
    
    self.register_buffer('decoder_rng', torch.arange(5))
    self.fin = torch.nn.Linear(128, 10)

  def forward(self, x):
    B = x.shape[0] # [B, 16, 784] this is 16 patches of 784 pixels each.
    pch = self.emb(x) # [B, 16, 128] this is the embedding of the 16 patches.
    cls = self.cls.expand(B, -1, -1) # [B, 1, 128] this is the embedding of the cls token.  
    hdn = torch.cat([cls, pch], dim=1) # [B, 17, 128] this is the embedding of the 16 patches and the cls token.
    hdn = hdn + self.pos(self.rng) # [B, 17, 128] this is the embedding of the 16 patches and the cls token with the positional encoding.
    for enc in self.enc: hdn = enc(hdn) # [B, 17, 128] this is the embedding of the 16 patches and the cls token with the positional encoding after the encoder layers.
    encoder_output = hdn


    decoder_input = torch.zeros(B, 5, 128).to(x.device)
    decoder_input[:, 0, :] = self.start_token.expand(B, -1)
    decoder_input[:, 1:, :] = self.digit_tokens.expand(B, -1, -1) # 

    decoder_input = decoder_input + self.decoder_pos(self.decoder_rng)

    decoder_output = self.decoder(decoder_input, encoder_output)

    predictions = []

    for i in range(1,5):
      digital_pred = self.fin(decoder_output[:, i, :])
      predictions.append(digital_pred)


    predictions = torch.stack(predictions, dim=1)
    return predictions  
  
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

  def forward(self, x, context=None):
    if context is None:
            context = x

    B, N, C = x.shape # [B, 17, 128] this is the shape of the input. B is the batch size, N is the number of tokens, C is the dimension of the input.
    B_c, N_c, C_c = context.shape
      # Project to Q, K, V
    qry = self.q_proj(x)  # [B, 17, 128]
    key = self.k_proj(context)  # [B, 17, 128]
    val = self.v_proj(context)  # [B, 17, 128]
    
    # Reshape for multi-head: [B, seq_len, num_heads, head_dim]
    qry = qry.view(B, N, self.num_heads, self.head_dim)
    key = key.view(B_c, N_c, self.num_heads, self.head_dim)
    val = val.view(B_c, N_c, self.num_heads, self.head_dim)
    
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

class DecoderLayer(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim)  # Self-attention
        self.cross_attn = MultiHeadAttention(dim)  # Cross-attention to encoder
        self.ffn = FFN(dim)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.norm3 = torch.nn.LayerNorm(dim)
    
    def forward(self, x, encoder_output):
        # Self-attention (decoder tokens attend to each other)
        attn_out = self.self_attn(x)
        x = self.norm1(x + attn_out)
        
        # Cross-attention (decoder tokens attend to encoder output)
        cross_out = self.cross_attn(x, encoder_output)
        x = self.norm2(x + cross_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        return x

class Decoder(torch.nn.Module):
  def __init__(self, dim, num_layers):
    super().__init__()
    self.dec = torch.nn.ModuleList([DecoderLayer(dim) for _ in range(num_layers)])
    self.norm = torch.nn.LayerNorm(dim)

  def forward(self, x, encoder_output):
    for layer in self.dec:
        x = layer(x, encoder_output)
    return self.norm(x)

class Attention(torch.nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim
    self.q_proj = torch.nn.Linear(dim, dim)
    self.k_proj = torch.nn.Linear(dim, dim)
    self.v_proj = torch.nn.Linear(dim, dim)
    self.o_proj = torch.nn.Linear(dim, dim)
    self.drpout = torch.nn.Dropout(0.1)

  def forward(self, x):
    qry = self.q_proj(x)
    key = self.k_proj(x)
    val = self.v_proj(x)
    att = qry @ key.transpose(-2, -1) * self.dim ** -0.5
    att = torch.softmax(att, dim=-1)
    att = self.drpout(att)
    out = torch.matmul(att, val)
    return self.o_proj(out)

  
  