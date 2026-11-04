
from bpe import BPE

# Corpus de treinamento
textos = [
    "O Transformer utiliza mecanismos de atenção.",
    "O modelo aprende a prever o próximo token.",
    "Redes neurais são utilizadas em inteligência artificial.",
    "O mecanismo Multi Head Attention permite atenção paralela.",
    "A arquitetura GPT utiliza modelos autoregressivos.",
    "Tokenização transforma textos em sequências numéricas.",
    "O treinamento utiliza gradiente descendente.",
    "Transformers podem aprender representações de linguagem."
] * 100

# Criação do tokenizador
tokenizer = BPE(vocab_size=2000)

# Treinamento
tokenizer.train(
    textos,
    min_frequency=2
)

# Salvamento
tokenizer.save("bpe.json")

# Teste
texto = "O Transformer aprende linguagem."

tokens = tokenizer.encode(texto)

print("Texto:", texto)
print("Tokens:", tokens)
print("Decodificado:", tokenizer.decode(tokens))
