import numpy as np
 
# Cel zadania: nauczyć modelu bramki logicznej AND
 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # Parametry wejściowe (nasze opcje)
y = np.array([[0], [0], [0], [1]]) # Poprawna odpowiedź (będzie potrzebne przy obliczaniu błędu naszego modelu)
 
# Początkowo nasze wagi i bias będą losowe
np.random.seed(0)
weights = np.random.rand(2, 1) # Tworzymy macierz wag 2x1, bo mamy dwie cechy wejściowe i jeden neuron wyjściowy
bias = np.random.rand(1) # Bias ma jeden parametr, bo mamy tylko jeden neuron
 
# FUNKCJA AKTYWACJI
# tutaj akurat używamy "sigmoida" - funkcji nieliniowej, która zawęzi nam możliwe wyniki na przedział od 0 do 1
# inne funkcje aktywacji: https://www.v7labs.com/blog/neural-networks-activation-functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 
# POCHODNA OD FUNKCJI AKTYWACJI ^
def sigmoid_derivative(x):
    return x * (1 - x)
 
learning_rate = 0.1  # learning rate to współczynnik, który określi nam jak duże korekty będzie wykonywał model
epochs = 10000  # ilość powtórzeń pętli do nauki
 
# PROCES UCZENIA
for epoch in range(epochs):
    # .dot służy do przemnożenia macierzy, basicly mnożymy macierz naszych wejść przez wagi i dodajemy bias
    # bias jest niezbędny - pozwala na wychylenie naszej funkcji góra dół
    # to tak jak funkcja y = mx + b gdzie 'm' to wagi a 'b' to bias
    z = np.dot(X, weights) + bias
    output = sigmoid(z)
 
    # obliczenie błędu/pomyłki modelu
    error = y - output
 
    # nasz błąd mnożymy przez pochodną sigmoida
    # pochodna mówi nam, jak "pewny siebie" jest neuron
    # jeśli wynik będzie blisko 0.5 to pochodna będzie wysoka, a jeśli wynik będzie blisko wartości granicznej 0 lub 1 - będzie niska
    # dzięki temu największe poprawki robimy tam gdzie neuron jest niezdecydowany, a myli się mocno
    adjustments = error * sigmoid_derivative(output)
 
    # zmieniamy nasze wagi i bias mnożąc przy tym przez learning rate żeby model nie wachał się zbyt mocno
    weights += np.dot(X.T, adjustments) * learning_rate
    bias += np.sum(adjustments) * learning_rate
 
    # co 1000 epok wypisujemy stan modelu
    if epoch % 1000 == 0:
        print(f"Epoka {epoch}, Błąd: {np.mean(np.abs(error)):.4f}")
 
print("\nWynik po nauce (dla wejść [0,0], [0,1], [1,0], [1,1]):")
print(output.round(3))
