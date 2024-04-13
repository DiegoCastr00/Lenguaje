
#pip install tensorflow
#pip install keras


from pandas import read_csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt

INPUT_FILE = '...//wonderland.txt'

# extraer la entrada como un flujo de caracteres
print("Extraer texto de los datos de entrada...")
fin = open(INPUT_FILE, 'rb')
lines = []
for line in fin:
    line = line.strip().lower()
    line = line.decode("ascii", "ignore")
    if len(line) == 0:
        continue
    lines.append(line)
fin.close()
text = " ".join(lines)

# crear tablas de consulta
#Aquí chars es el número de características de nuestro "vocabulario" de caracteres
chars = set([c for c in text])
nb_chars = len(chars)
char2index = dict((c, i) for i, c in enumerate(chars))
index2char = dict((i, c) for i, c in enumerate(chars))

#   e sky was  -> f
#    sky was f -> a
#   sky was fa -> l
print("Crear entradas y etiquetas de texto...")
SEQLEN = 10
STEP = 1

input_chars = []
label_chars = []
for i in range(0, len(text) - SEQLEN, STEP):
    input_chars.append(text[i:i + SEQLEN])
    label_chars.append(text[i + SEQLEN])

# vectorizar la entrada y etiquetar los caracteres
# Cada fila de la entrada está representada por caracteres seqlen, cada uno
# representado como una codificación 1-hot de tamaño len(char). Hay
# len(input_chars) such rows, so shape(X) is (len(input_chars),
# seqlen, nb_chars).
# Cada fila de salida es un único carácter, también representado como una
# codificación densa de tamaño len(char). Hence shape(y) is (len(input_chars),
# nb_chars).
print("Vectorización del texto de entrada y de las etiquetas...")
X = np.zeros((len(input_chars), SEQLEN, nb_chars), dtype=np.bool)
y = np.zeros((len(input_chars), nb_chars), dtype=np.bool)
for i, input_char in enumerate(input_chars):
    for j, ch in enumerate(input_char):
        X[i, j, char2index[ch]] = 1
    y[i, char2index[label_chars[i]]] = 1

# Construir el modelo. Usamos una única RNN con una capa totalmente conectada
# para calcular la salida predicha más probable char
HIDDEN_SIZE = 128
BATCH_SIZE = 128
NUM_ITERATIONS = 25
NUM_EPOCHS_PER_ITERATION = 1
NUM_PREDS_PER_EPOCH = 100

model = Sequential()
model.add(SimpleRNN(HIDDEN_SIZE, return_sequences=False,
                    input_shape=(SEQLEN, nb_chars),
                    unroll=True))
model.add(Dense(nb_chars))
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

# Entrenamos el modelo por lotes y probamos la salida generada en cada paso
for iteration in range(NUM_ITERATIONS):
    print("=" * 50)
    print("Iteracion #: %d" % (iteration))
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS_PER_ITERATION)

    # modelo de pruebas
    # elegir aleatoriamente una fila de input_chars, luego usarla para
    # generar texto a partir del modelo para los siguientes 100 caracteres
    test_idx = np.random.randint(len(input_chars))
    test_chars = input_chars[test_idx]
    print("Generación a partir de la semilla: %s" % (test_chars))
    print(test_chars, end="")
    for i in range(NUM_PREDS_PER_EPOCH):
        Xtest = np.zeros((1, SEQLEN, nb_chars))
        for i, ch in enumerate(test_chars):
            Xtest[0, i, char2index[ch]] = 1
        pred = model.predict(Xtest, verbose=0)[0]
        ypred = index2char[np.argmax(pred)]
        print(ypred, end="")
       # avanzar con test_chars + ypred
        test_chars = test_chars[1:] + ypred
    print()