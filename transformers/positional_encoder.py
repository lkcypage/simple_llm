import numpy as np

class PositionalEncoder:
    def __init__(self, seq_len, d_model):
        self.seq_len = seq_len
        self.d_model = d_model
        self.positional_encoding = self.generate_positional_encoding(seq_len, d_model)
    
    def generate_positional_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))   
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe
    # 3. Basta somar a matriz posicional aos seus embeddings!
    #resultado_final = embedding_da_frase + generated_positional_encoding()