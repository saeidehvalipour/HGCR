import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionModel(nn.Module):
    def __init__(
        self,
        pmid_input_dim,
        node_input_dim,
        embed_size,
        num_heads,
        att_dropout,
    ):
        super(CrossAttentionModel, self).__init__()

        # Embedding layers for pmid_emb_seq and node_emb_seq
        self.pmid_fc = nn.Linear(pmid_input_dim, embed_size)
        self.node_fc = nn.Linear(node_input_dim, embed_size)

        # Self-Attention and Cross-Attention layers
        self.self_attention  = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=att_dropout)
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=att_dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.ReLU(),
            nn.Linear(embed_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, pmid_emb_seq, node_emb_seq):
        # Transforming input dimensions to embedding size
        pmid_emb_seq = self.pmid_fc(pmid_emb_seq)
        node_emb_seq = self.node_fc(node_emb_seq)

        # Transpose for nn.MultiheadAttention: [seq_len, batch_size, embed_size]
        pmid_emb_seq = pmid_emb_seq.transpose(0, 1)
        node_emb_seq = node_emb_seq.transpose(0, 1)

        # Self-Attention for pmid_emb_seq
        self_attn_output, _ = self.self_attention(pmid_emb_seq, pmid_emb_seq, pmid_emb_seq)
        self_attn_output = self.norm1(pmid_emb_seq + self_attn_output)

        # Cross-Attention: pmid_emb_seq attends to node_emb_seq
        cross_attn_output, _ = self.cross_attention(self_attn_output, node_emb_seq, node_emb_seq)
        cross_attn_output = self.norm2(self_attn_output + cross_attn_output)

        # Transpose back to [batch_size, seq_len, embed_size]
        cross_attn_output = cross_attn_output.transpose(0, 1)

        # Pooling and classification
        pooled_output = cross_attn_output.mean(dim=1)
        prediction = self.classifier(pooled_output)

        return prediction.squeeze(-1)
