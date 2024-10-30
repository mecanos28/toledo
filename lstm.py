""" Este módulo prepara datos de archivos MIDI y los alimenta a la red neuronal para entrenamiento con longitudes de notas variables """

# Importamos las librerías necesarias para el procesamiento y entrenamiento
import glob  # Para encontrar todos los archivos que coinciden con un patrón específico
import pickle  # Para serializar y deserializar objetos de Python
import numpy  # Biblioteca para cálculos numéricos y matrices multidimensionales
from music21 import converter, instrument, note, chord  # Librería para trabajar con datos musicales
from keras.models import Sequential  # Para crear modelos secuenciales en Keras
from keras.layers import Dense, Dropout, LSTM, Activation  # Capas de la red neuronal
from keras.layers import BatchNormalization as BatchNorm  # Para normalización de lotes
from keras.utils import to_categorical  # Para convertir etiquetas en formato categórico
from keras.callbacks import ModelCheckpoint  # Para guardar el modelo durante el entrenamiento


def entrenar_red():
    """ Entrena una Red Neuronal para generar música """
    notas = obtener_notas()  # Llama a la función para obtener todas las notas y acordes de los archivos MIDI

    # Calcula el número de combinaciones únicas de tono y duración
    n_vocab = len(set(notas))  # El tamaño del vocabulario

    # Prepara las secuencias de entrada y salida para la red neuronal
    entrada_red, salida_red = preparar_secuencias(notas, n_vocab)

    # Crea la estructura de la red neuronal
    modelo = crear_red(entrada_red, n_vocab)

    # Entrena el modelo con las secuencias preparadas
    entrenar_modelo(modelo, entrada_red, salida_red)


def obtener_notas():
    """ Obtiene todas las notas y acordes con sus duraciones de los archivos MIDI en el directorio ./midi_songs """
    notas = []  # Lista para almacenar las notas y acordes

    # Recorre todos los archivos MIDI en el directorio especificado
    for archivo in glob.glob("input/midi_songs/*.mid"):
        midi = converter.parse(archivo)  # Convierte el archivo MIDI en un objeto de música

        print(f"Analizando {archivo}")  # Imprime el nombre del archivo que se está analizando

        notas_a_analizar = None  # Inicializa la variable para almacenar las notas a analizar

        try:
            # Intenta separar las partes por instrumento
            s2 = instrument.partitionByInstrument(midi)
            notas_a_analizar = s2.parts[0].recurse()  # Toma las notas de la primera parte
        except:
            # Si no hay partes, toma todas las notas de manera plana
            notas_a_analizar = midi.flat.notes

        # Recorre cada elemento en las notas a analizar
        for elemento in notas_a_analizar:
            duracion = elemento.duration.quarterLength  # Obtiene la duración del elemento en cuartos de nota

            # Clasifica la duración en 'corta', 'media' o 'larga'
            if duracion < 0.75:
                clase_duracion = 'corta'
            elif duracion < 1.5:
                clase_duracion = 'media'
            else:
                clase_duracion = 'larga'

            if isinstance(elemento, note.Note):
                # Si el elemento es una nota, combina el tono y la clase de duración
                nota_str = f"{str(elemento.pitch)}_{clase_duracion}"
                notas.append(nota_str)  # Agrega la nota a la lista
            elif isinstance(elemento, chord.Chord):
                # Si el elemento es un acorde, combina los tonos y la clase de duración
                acorde_str = f"{'.'.join(str(n) for n in elemento.normalOrder)}_{clase_duracion}"
                notas.append(acorde_str)  # Agrega el acorde a la lista

    # Guarda la lista de notas en un archivo utilizando pickle
    with open('notas', 'wb') as filepath:
        pickle.dump(notas, filepath)

    return notas  # Devuelve la lista de notas


def preparar_secuencias(notas, n_vocab):
    """ Prepara las secuencias utilizadas por la Red Neuronal """
    longitud_secuencia = 10  # Define la longitud de las secuencias de entrada

    # Obtiene todas las combinaciones únicas de tono y duración, ordenadas
    nombres_notas = sorted(set(notas))

    # Crea un diccionario que mapea cada nota a un número entero
    nota_a_int = dict((nota, numero) for numero, nota in enumerate(nombres_notas))

    entrada_red = []  # Lista para almacenar las secuencias de entrada
    salida_red = []  # Lista para almacenar las notas de salida correspondientes

    # Recorre las notas para crear pares de secuencias de entrada y salida
    for i in range(len(notas) - longitud_secuencia):
        secuencia_entrada = notas[i:i + longitud_secuencia]  # Extrae una secuencia de notas
        secuencia_salida = notas[i + longitud_secuencia]  # La nota siguiente a predecir
        # Convierte las notas de la secuencia de entrada a números utilizando el diccionario
        entrada_red.append([nota_a_int[char] for char in secuencia_entrada])
        # Convierte la nota de salida a su número correspondiente
        salida_red.append(nota_a_int[secuencia_salida])

    n_patrones = len(entrada_red)  # Número total de patrones

    # Redimensiona la entrada para que sea compatible con las capas LSTM (n_patrones, longitud_secuencia, 1)
    entrada_red = numpy.reshape(entrada_red, (n_patrones, longitud_secuencia, 1))
    # Normaliza los valores de entrada dividiendo por el tamaño del vocabulario
    entrada_red = entrada_red / float(n_vocab)

    # Convierte las salidas a formato categórico (one-hot encoding)
    salida_red = to_categorical(salida_red)

    return entrada_red, salida_red  # Devuelve las secuencias de entrada y salida preparadas


def crear_red(entrada_red, n_vocab):
    """ Crea la estructura de la red neuronal """
    modelo = Sequential()  # Inicializa el modelo secuencial

    # Agrega una capa LSTM con 512 unidades, con dropout recurrente y devuelve secuencias
    modelo.add(LSTM(
        512,
        input_shape=(entrada_red.shape[1], entrada_red.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    # Agrega otra capa LSTM similar
    modelo.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    # Agrega una tercera capa LSTM sin devolver secuencias
    modelo.add(LSTM(512))
    # Agrega una capa de normalización por lotes para acelerar el entrenamiento
    modelo.add(BatchNorm())
    # Agrega una capa de dropout para reducir el sobreajuste
    modelo.add(Dropout(0.3))
    # Agrega una capa densa con 256 unidades y activación ReLU
    modelo.add(Dense(256, activation='relu'))
    # Otra capa de normalización por lotes
    modelo.add(BatchNorm())
    # Otra capa de dropout
    modelo.add(Dropout(0.3))
    # Capa de salida con activación softmax para predecir la probabilidad de cada nota en el vocabulario
    modelo.add(Dense(n_vocab, activation='softmax'))
    # Compila el modelo con pérdida de entropía cruzada categórica y optimizador RMSprop
    modelo.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return modelo  # Devuelve el modelo creado


def entrenar_modelo(modelo, entrada_red, salida_red):
    """ Entrena la red neuronal """
    # Define la ruta del archivo donde se guardarán los pesos del modelo
    filepath = "pesos-mejora-midis-gino-{epoch:02d}-{loss:.4f}-mejor.keras"

    # Crea un checkpoint para guardar el modelo cada vez que se alcance una nueva mejor pérdida
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',  # Monitorea la pérdida del entrenamiento
        verbose=0,  # No muestra mensajes adicionales
        save_best_only=True,  # Solo guarda el modelo si es el mejor hasta ahora
        mode='min'  # Modo de minimizar la pérdida
    )
    callbacks_list = [checkpoint]  # Lista de callbacks

    # Entrena el modelo con los datos de entrada y salida, durante 200 épocas y tamaño de lote 128
    modelo.fit(entrada_red, salida_red, epochs=200, batch_size=128, callbacks=callbacks_list)


if __name__ == '__main__':
    entrenar_red()  # Llama a la función principal para iniciar el entrenamiento
