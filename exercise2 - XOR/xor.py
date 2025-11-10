import numpy as np
import matplotlib.pyplot as plt

# Cel zadania: nauczyć modelu bramki logicznej XOR

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(0)
w1 = np.random.rand(2, 2)  # Tworzymy macierz wag 2x2, bo mamy dwie cechy wejściowe i dwa neurony w pierwszej warstwie
w2 = np.random.rand(2, 1) # Druga macierz out jest na wyjściu obu neuronów do jednej odpowiedzi (jednego neurona wyjściowego) dlatego 2x1
b1 = np.random.rand(1, 2) # 2 biasy po jednym dla każdego neuronu
b2 = np.random.rand(1)  # Bias neuronu wyjściowego


# FUNKCJA AKTYWACJI
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# POCHODNA OD FUNKCJI AKTYWACJI ^
def sigmoid_derivative(x):
    return x * (1 - x)


learning_rate = 0.1
epochs = 10000

# PROCES UCZENIA
for epoch in range(epochs):
    # forward propagation
    hidden_layer = np.dot(X, w1) + b1
    out_hidden = sigmoid(hidden_layer)
    out_layer = np.dot(out_hidden, w2) + b2
    out = sigmoid(out_layer)

    # backpropagation
    error_out = y - out
    adjustments_out = error_out * sigmoid_derivative(out)

    error_hidden = np.dot(adjustments_out, w2.T)
    adjustments_hidden = error_hidden * sigmoid_derivative(out_hidden)

    # aktualizacja wag
    w2 += np.dot(out_hidden.T, adjustments_out) * learning_rate
    b2 += np.sum(adjustments_out) * learning_rate

    w1 += np.dot(X.T, adjustments_hidden) * learning_rate
    b1 += np.sum(adjustments_hidden) * learning_rate

    if epoch % 1000 == 0:
        print(f"Epoka {epoch}, Błąd: {np.mean(np.abs(error_out)):.4f}")

print("\nWynik po nauce (dla wejść [0,0], [0,1], [1,0], [1,1]):")
print(out.round(3))


# wyrysowanie przewidywań
xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

Z = sigmoid(np.dot(sigmoid(np.dot(grid, w1) + b1), w2) + b2).reshape(xx.shape)

plt.imshow(Z, extent=[0, 1, 0, 1], origin='lower', cmap='inferno', alpha=0.6)
plt.title("Wizualizacja sieci neuronowej")
plt.xlabel("Wejście 1")
plt.ylabel("Wejście 2")
plt.colorbar(label="Pewność sieci (0=Fałsz, 1=Prawda)")
plt.grid(linestyle='--', alpha=0.5)
plt.show()
