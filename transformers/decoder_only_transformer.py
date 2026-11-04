import numpy as np

class Transformer:
    def __init__(self, vocab_size, d_model=768, d_ff=3072, seq_len=512):
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.seq_len = seq_len
        
        self.embedding = np.random.randn(vocab_size, d_model) # Token embedding matrix
        
        # Attention weight matrices
        self.Q_weight = np.random.randn(d_model, d_model)
        self.K_weight = np.random.randn(d_model, d_model)
        self.V_weight = np.random.randn(d_model, d_model)
        self.O_weight = np.random.randn(d_model, d_model)
        
        # Feed-forward weights
        self.L1 = np.random.randn(d_model, d_ff)
        self.L2 = np.random.randn(d_ff, d_model)
        
        self.W_out = np.random.randn(d_model, vocab_size) # Output projection
        
        self.positional_encoding = self._generate_positional_encoding(seq_len, d_model)
    

    def _generate_positional_encoding(self, seq_len, d_model):
        positional_encoding = np.zeros((seq_len, d_model)) # Positional encoding matrix

        position = np.arange(seq_len)[:, np.newaxis] # Position indices
        dimension = np.arange(d_model)[np.newaxis, :] # Dimension indices
        
        # Compute angles using sine for even indices and cosine for odd indices
        angle = position / np.power(10000, (2 * (dimension//2)) / d_model)
        positional_encoding = np.where(dimension % 2 == 0, np.sin(angle), np.cos(angle))
        return positional_encoding
    
    
    def _generate_causal_mask(self, seq_len):
        mask = np.tril(np.ones((seq_len, seq_len))) # Create lower triangular matrix to prevent attending to future tokens
        mask = np.where(mask == 0, -1e9, 0) # Convert zeros to negative values for masking
        
        return mask
    
    
    def _masked_self_attention(self, input):
        Q = np.dot(input, self.Q_weight) # Query matrix
        K = np.dot(input, self.K_weight) # Key matrix
        V = np.dot(input, self.V_weight) # Value matrix
        
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_model)
        
        seq_len = input.shape[1]
        mask = self._generate_causal_mask(seq_len)
        scores = scores + mask # Apply mask to attention scores
        
        # Apply softmax to get attention weights
        attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
        
        # Compute the weighted sum of values
        output = np.matmul(attention_weights, V)
        
        # Apply output projection
        return np.dot(output, self.O_weight)
    
    
    def _feed_forward(self, input):
        first_layer = np.dot(input, self.L1)
        activation = np.maximum(0, first_layer)  # ReLU activation
        second_layer = np.dot(activation, self.L2)
        return second_layer
    
    
    def _layer_norm(self, input):
        mean = np.mean(input, axis=-1, keepdims=True)
        std = np.std(input, axis=-1, keepdims=True)
        
        # Normalize the input
        return (input - mean) / (std + 1e-6)
    
    
def forward(self, x):
        x = self.embedding[x] # Convert token indices to embeddings
        
        x = x + self.positional_encoding[:x.shape[1], :] # Add positional encoding
        attn_output = self._masked_self_attention(x) # Compute masked self-attention
        x = self._layer_norm(x + attn_output)  # Apply layer normalization
        ff_output = self._feed_forward(x) # Apply feed-forward network
        x = self._layer_norm(x + ff_output) # Apply layer normalization
        
        vocab_scores = np.dot(x, self.W_out) # Project final hidden states to next token scores
        
        return vocab_scores
