import numpy as np

class Transformer:
    def __init__(self, d_model=768, num_heads=12, d_ff=3072, seq_len=512): #Values based on BERT model
        self.d_model = d_model  # Dimension of input embeddings
        self.num_heads = num_heads  # Number of attention heads
        self.d_ff = d_ff  # Dimension of feed-forward network
        self.seq_len = seq_len  # Maximum sequence length
        
        # Initialize weight matrices for multi-head self-attention
        self.Q_weight = np.random.randn(d_model, d_model)  # Query weight matrix
        self.K_weight = np.random.randn(d_model, d_model)  # Key weight matrix
        self.V_weight = np.random.randn(d_model, d_model)  # Value weight matrix
        self.O_weight = np.random.randn(d_model, d_model)  # Output projection weight matrix
        
        # Initialize weight matrices for feed-forward network
        self.L1 = np.random.randn(d_model, d_ff)  # First feed-forward layer
        self.L2 = np.random.randn(d_ff, d_model)  # Second feed-forward layer
        
        # Generate positional encoding matrix
        self.positional_encoding = self._generate_positional_encoding(seq_len, d_model)
        
    def _generate_positional_encoding(self, seq_len, d_model):
         # Create a matrix to store positional encodings
        pos_encoding = np.zeros((seq_len, d_model))
        # Create 2D sequence and embedding indices
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        
        # Compute angles using sine for even indices and cosine for odd indices
        angle = pos / np.power(10000, (2 * (i//2)) / d_model)
        pos_encoding = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))
        return pos_encoding
    
    def _multi_head_attention(self, input):
        
        # Compute query, key, and value matrices
        Q = np.dot(input, self.Q_weight)
        K = np.dot(input, self.K_weight)
        V = np.dot(input, self.V_weight)
        
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_model)
        
        # Apply softmax to get attention weights
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        
        # Compute the weighted sum of values
        output = np.matmul(attention_weights, V)
        
        # Apply output projection
        return np.dot(output, self.O_weight)
    
    def _feed_forward(self, input):
        # Perform foward pass through a 2 layer neural network using Rectified Linear Unit (ReLU)
        first_layer = np.dot(input, self.L1)
        activation = np.maximum(0, first_layer)  # Apply ReLU activation for non-linearity
        second_layer = np.dot(activation, self.L2)
        return second_layer
    
    def _layer_norm(self, input):
        # Compute mean and standard deviation for normalization
        mean = np.mean(input, axis=-1, keepdims=True)
        std = np.std(input, axis=-1, keepdims=True)
        
        # Normalize the input
        return (input - mean) / (std + 1e-6)
    
    def forward(self, x):
        # Add positional encoding to input embeddings
        x = x + self.positional_encoding[:x.shape[1], :]
        
        # Apply multi-head self-attention and residual connection
        attn_output = self._multi_head_attention(x)
        x = self._layer_norm(x + attn_output)
        
        # Apply feed-forward network and residual connection
        ff_output = self._feed_forward(x)
        x = self._layer_norm(x + ff_output)
        
        # Return processed output
        return x
