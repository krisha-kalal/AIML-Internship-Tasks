import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import lightning as L

# 1. Vocabulary & Mappings
token_to_id = {
    'what': 0, 'is': 1, 'python': 2, 'it': 3,
    'a': 4, 'programming': 5, 'language': 6, '<EOS>': 7
}
id_to_token = {v: k for k, v in token_to_id.items()}

# 2. Input/Label Sequences (Batch Size = 2, Sequence Length = 9)
inputs = torch.tensor([
    [token_to_id['what'], token_to_id['is'], token_to_id['python'], token_to_id['<EOS>'],
     token_to_id['it'], token_to_id['is'], token_to_id['a'], token_to_id['programming'], token_to_id['language']],
    [token_to_id['python'], token_to_id['is'], token_to_id['what'], token_to_id['<EOS>'],
     token_to_id['it'], token_to_id['is'], token_to_id['a'], token_to_id['programming'], token_to_id['language']]
], dtype=torch.long)

labels = torch.tensor([
    [token_to_id['is'], token_to_id['python'], token_to_id['<EOS>'], token_to_id['it'],
     token_to_id['is'], token_to_id['a'], token_to_id['programming'], token_to_id['language'], token_to_id['<EOS>']],
    [token_to_id['is'], token_to_id['what'], token_to_id['<EOS>'], token_to_id['it'],
     token_to_id['is'], token_to_id['a'], token_to_id['programming'], token_to_id['language'], token_to_id['<EOS>']]
], dtype=torch.long)

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset, batch_size=2)


# 3. Modules
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=2, max_len=15):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, word_embeddings):
        # word_embeddings shape: (batch_size, seq_len, d_model)
        seq_len = word_embeddings.size(1)
        return word_embeddings + self.pe[:seq_len, :].unsqueeze(0)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=2):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.d_model = d_model

    def forward(self, x, mask=None):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Scaled dot-product attention across 3D batches: (B, T, D) x (B, D, T) -> (B, T, T)
        sims = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5)

        if mask is not None:
            sims = sims.masked_fill(mask == 0, float('-inf'))

        attention_percents = F.softmax(sims, dim=-1)
        return torch.matmul(attention_percents, v)


class DecoderOnlyTransformer(L.LightningModule):
    def __init__(self, num_tokens=len(token_to_id), d_model=2, max_len=15):
        super().__init__()
        self.we = nn.Embedding(num_embeddings=num_tokens, embedding_dim=d_model)
        self.pe = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.self_attention = CausalSelfAttention(d_model=d_model)
        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, token_ids):
        # token_ids shape: (batch_size, seq_len)
        seq_len = token_ids.size(1)
        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)

        # Construct causal mask (1 for allowed, 0 for blocked)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=token_ids.device))
        
        self_attention_values = self.self_attention(position_encoded, mask=mask)
        residual_connection_values = position_encoded + self_attention_values
        logits = self.fc_layer(residual_connection_values)
        return logits

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):
        input_tokens, targets = batch
        logits = self.forward(input_tokens)
        
        # Reshape for CrossEntropyLoss: (batch_size * seq_len, num_tokens) vs (batch_size * seq_len)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss


# 4. Model Instantiation & Generation
model = DecoderOnlyTransformer()

# 2. Train the model so weights update
trainer = L.Trainer(max_epochs=200, accelerator="auto", logger=False, enable_checkpointing=False)
trainer.fit(model, dataloader)

# 3. Corrected Generation Loop
model.eval()
model_input = torch.tensor([[token_to_id['what'],
                             token_to_id['is'],
                             token_to_id['python'],
                             token_to_id['<EOS>']]], dtype=torch.long)

max_length = 15

with torch.no_grad():
    for _ in range(model_input.size(1), max_length):
        logits = model(model_input)
        
        # Get highest probability prediction for the last position
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        
        # Append predicted token
        model_input = torch.cat((model_input, next_token_id), dim=1)
        
        # STOP as soon as <EOS> is generated
        if next_token_id.item() == token_to_id['<EOS>']:
            break

print("Predicted token sequence:\n")
for token_id in model_input.squeeze().tolist():
    print("\t", id_to_token[token_id])