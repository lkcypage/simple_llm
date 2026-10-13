import numpy as np
class normalizacao:
    def __init__(self, X, Y, B, E):
        self.resultado = self.entrada(X, Y, B, E)
    def entrada(self, X, Y, B, E):
        def media_micro(X):
            i = len(X)
            Xs = np.sum(X)
            return (1 / i) * Xs
        def variancia(X, media_micro):
            u = media_micro(X)
            i = len(X)
            a = (X - u)**2
            Xf = np.sum(a)
            final = (1 / i) * Xf
            resultado = np.sqrt(final)
            return resultado
        def norm(X, media_micro, variancia, E):
            u = media_micro(X)
            o = variancia(X, media_micro)
            r = o**2 + E
            return (X - u) / np.sqrt(r)
        def final_y(norm, B, Y):
            x = norm(X, media_micro, variancia, E)
            return Y*x + B
        media_micro(X)
        variancia(X, media_micro)
        norm(X, media_micro, variancia, E)
        resultado = final_y(norm, B, Y)
        return resultado
    def __str__(self):
        return str(self.resultado)