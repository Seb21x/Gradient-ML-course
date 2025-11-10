import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

# Zadanie: klasyfikacja cyfry na obrazie

# odczyt danych
train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

# y ma odpowiedzi, x ma wartości pikseli
y = train_data['label']
x = train_data.drop('label', axis=1)
y_test = test_data['label']
x_test = test_data.drop('label', axis=1)

# normalizacja danych, każdy piksel otrzyma wartość od 0 do 1
x = x/255
x_test = x_test/255

# podział danych na treningowe i walidacyjne
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=777)

# ilość neuronów w naszej warstwie ukrytej
hidden_layer = 128
np.random.seed(777)
w1 = np.random.randn(784, hidden_layer) # Macierz wag ma 784, bo obrazek jest 28x28 pikseli
w2 = np.random.randn(hidden_layer, 10) # 10 możliwości na wyjściu, bo mamy 10 cyfr
b1 = np.zeros((1, hidden_layer)) # biasy dla hidden_layer
b2 = np.zeros((1, 10))  # Bias neuronów wyjściowych

# softmax do klasyfikacji, gdy mamy więcej niż 2 klasy (u nas jest ich aż 10)
# do output layer
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# relu do hidden layer, pomaga uniknąć zanikania gradientu i przyspiesza uczenie
# zanikanie gradientu uniemożliwia uczenie się sieci, ponieważ pochodna jest zbyt bliska zeru
def relu(x):
    return np.maximum(0, x)

# pochodna od relu ^
def relu_derivative(x):
    return (x > 0).astype(float)

# musimy mieć 10 elementową tablicę, która wskaże nam, która cyfra jest poprawną odpowiedzią
# przy poprawnej odpowiedzi postawimy 1, a przy reszcie 0
def one_hot_encoding(y, num_classes = 10):
    # Tworzymy macierz samych zer o wymiarach (liczba_próbek, 10)
    one_hot_y = np.zeros((y.size, num_classes))
    # Wstawiamy 1 w odpowiednie miejsce
    rows = np.arange(y.size)
    one_hot_y[rows, y] = 1.0
    return one_hot_y

y_train = one_hot_encoding(y_train.values.astype(int))
y_val = one_hot_encoding(y_val.values.astype(int))

learning_rate = 0.1
epochs = 50
# musimy podzielić sobie nasze dane na mniejsze porcje
batch_size = 64
num_samples = x_train.shape[0]  # Całkowita liczba obrazków treningowych

# PROCES UCZENIA
for epoch in range(epochs):
    # mieszanie danych przed każdą epoką
    shuffled_indices = np.random.permutation(num_samples)
    x_train_shuffled = x_train.iloc[shuffled_indices]
    y_train_shuffled = y_train[shuffled_indices]

    for i in range(0, num_samples, batch_size):
        # wyciągamy kawałek danych (batch)
        x_batch = x_train_shuffled[i: i + batch_size]
        y_batch = y_train_shuffled[i: i + batch_size]

        # sprawdzamy rzeczywisty rozmiar tego batcha (ostatni może być mniejszy niż 64)
        current_batch_size = len(x_batch)

        # FORWARD PROPAGATION
        # warstwa ukryta
        hidden_inputs = np.dot(x_batch, w1) + b1
        hidden_outputs = relu(hidden_inputs)

        # warstwa wyjściowa
        final_inputs = np.dot(hidden_outputs, w2) + b2
        final_probabilities = softmax(final_inputs)

        # BACKPROPAGATION
        # błąd na wyjściu
        output_error = final_probabilities - y_batch

        # błąd warstwy ukrytej
        hidden_error = np.dot(output_error, w2.T) * relu_derivative(hidden_outputs)

        # OBLICZENIE GRADIENTÓW
        # dzielimy przez current_batch_size, żeby uśrednić gradient
        grad_w2 = np.dot(hidden_outputs.T, output_error) / current_batch_size
        grad_b2 = np.sum(output_error, axis=0, keepdims=True) / current_batch_size

        grad_w1 = np.dot(x_batch.T, hidden_error) / current_batch_size
        grad_b1 = np.sum(hidden_error, axis=0, keepdims=True) / current_batch_size

        # AKTUALIZACJA WAG
        # wagi
        w1 -= grad_w1 * learning_rate
        w2 -= grad_w2 * learning_rate
        # biasy
        b1 -= grad_b1 * learning_rate
        b2 -= grad_b2 * learning_rate

    # sprawdzanie postępów co 10 epok
    # sprawdzimy sobie zbiór treningowy i walidacyjny, aby wiedzieć, czy model się nie przeucza,
    # jeśli strata na zbiorze treningowym jest dużo niższa od straty na zbiorze walidacyjnym
    # oznacza to, że model zaczyna się przeuczać i gorzej radzi sobie na danych, których nie widział
    if epoch % 10 == 0 or epoch == epochs - 1:
        # sprawdzamy najpierw treningowy
        hidden_inputs_train = np.dot(x_train, w1) + b1
        hidden_outputs_train = relu(hidden_inputs_train)
        final_inputs_train = np.dot(hidden_outputs_train, w2) + b2
        probs_train = softmax(final_inputs_train)

        prediction = np.argmax(probs_train, axis=1)
        true_answer = np.argmax(y_train, axis=1)
        accuracy_train = np.mean(prediction == true_answer)

        # potem walidacyjny
        hidden_inputs_val = np.dot(x_val, w1) + b1
        hidden_outputs_val = relu(hidden_inputs_val)
        final_inputs_val = np.dot(hidden_outputs_val, w2) + b2
        probs_val = softmax(final_inputs_val)

        prediction = np.argmax(probs_val, axis=1)
        true_answer = np.argmax(y_val, axis=1)
        accuracy_val = np.mean(prediction == true_answer)

        # wypisujemy wyniki
        print(f"Epoka {epoch:3d} | "
              f"Train acc: {accuracy_train:.4f} | "
              f"Val acc: {accuracy_val:.4f}")

# TESTOWANIE NA LOSOWYM OBRAZKU
# losowy obrazek ze zbioru testowego
idx = random.randint(0, len(x_test) - 1)
test_image = x_test.iloc[idx].values # dane obrazka (wektor 784 liczb)
true_label = y_test.iloc[idx] # prawdziwa cyfra

# przygotowanie obrazku do sieci (musi mieć wymiar 1, 784)
input_data = test_image.reshape(1, 784)

# przepuszczamy przez sieć
hidden = np.dot(input_data, w1) + b1
out_hidden = relu(hidden)
logits = np.dot(out_hidden, w2) + b2
probabilities = softmax(logits)

# wynik-cyfra z największym prawdopodobieństwem
predicted_label = np.argmax(probabilities)
confidence = probabilities[0][predicted_label] * 100 # % pewności sieci

# wyświetlenie wyników
plt.figure(figsize=(4, 4))
plt.imshow(test_image.reshape(28, 28), cmap='gray') # zamieniamy wektor 784 z powrotem na kwadrat 28x28
plt.title(f"Prawda: {true_label} | Sieć: {predicted_label}\n({confidence:.1f}% pewności)")
plt.axis('off')
plt.show()

print(f"Dla wybranego obrazka sieć przewidziała: {predicted_label} z pewnością {confidence:.2f}%")
