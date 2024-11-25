""" Este módulo genera notas para un archivo MIDI utilizando la red neuronal entrenada con longitudes de notas variables """

# Importamos las librerías necesarias
import pickle  # Para cargar y guardar datos en formato binario
import numpy  # Biblioteca para cálculos numéricos y matrices multidimensionales
from music21 import instrument, note, stream, chord  # Librería para trabajar con datos musicales
from keras.models import Sequential  # Para crear modelos secuenciales en Keras
from keras.layers import Dense, Dropout, LSTM, BatchNormalization as BatchNorm, Activation  # Capas de la red neuronal

def generar():
    """ Genera un archivo MIDI de piano """
    # Cargamos las notas utilizadas para entrenar el modelo
    with open('notas', 'rb') as filepath:
        notas = pickle.load(filepath)

    # Obtenemos todos los nombres de tonos y duraciones
    nombres_notas = sorted(set(notas))
    n_vocab = len(nombres_notas)  # Tamaño del vocabulario

    # Preparamos las secuencias de entrada para la red neuronal
    entrada_red, entrada_normalizada = preparar_secuencias(notas, nombres_notas, n_vocab)

    # Creamos la estructura de la red neuronal y cargamos los pesos entrenados
    modelo = crear_red(entrada_normalizada, n_vocab)

    # Generamos notas utilizando el modelo entrenado
    salida_prediccion = generar_notas(modelo, entrada_red, nombres_notas, n_vocab)

    # Creamos un archivo MIDI a partir de las notas generadas
    crear_midi(salida_prediccion)

def preparar_secuencias(notas, nombres_notas, n_vocab):
    """ Prepara las secuencias utilizadas por la Red Neuronal """
    # Mapeamos las notas a números enteros
    nota_a_int = dict((nota, numero) for numero, nota in enumerate(nombres_notas))

    longitud_secuencia = 10  # Definimos la longitud de las secuencias de entrada
    entrada_red = []  # Lista para almacenar las secuencias de entrada

    # Creamos las secuencias de entrada a partir de las notas
    for i in range(len(notas) - longitud_secuencia):
        secuencia_entrada = notas[i:i + longitud_secuencia]  # Obtenemos una secuencia de notas
        entrada_red.append([nota_a_int[char] for char in secuencia_entrada])  # Convertimos las notas a enteros

    n_patrones = len(entrada_red)  # Número total de patrones

    # Redimensionamos la entrada para que sea compatible con las capas LSTM
    entrada_normalizada = numpy.reshape(entrada_red, (n_patrones, longitud_secuencia, 1))
    # Normalizamos los valores de entrada dividiendo por el tamaño del vocabulario
    entrada_normalizada = entrada_normalizada / float(n_vocab)

    return entrada_red, entrada_normalizada  # Devolvemos las secuencias de entrada y entrada normalizada

def crear_red(entrada_red, n_vocab):
    """ Crea la estructura de la red neuronal """
    modelo = Sequential()  # Inicializamos el modelo secuencial

    # Agregamos una capa LSTM con 512 unidades, con dropout recurrente y devuelve secuencias
    modelo.add(LSTM(
        512,
        input_shape=(entrada_red.shape[1], entrada_red.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    # Agregamos otra capa LSTM similar
    modelo.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    # Agregamos una tercera capa LSTM sin devolver secuencias
    modelo.add(LSTM(512))
    # Agregamos una capa de normalización por lotes
    modelo.add(BatchNorm())
    # Agregamos una capa de dropout para reducir el sobreajuste
    modelo.add(Dropout(0.3))
    # Agregamos una capa densa con 256 unidades y activación ReLU
    modelo.add(Dense(256, activation='relu'))
    # Otra capa de normalización por lotes
    modelo.add(BatchNorm())
    # Otra capa de dropout
    modelo.add(Dropout(0.3))
    # Capa de salida con activación softmax
    modelo.add(Dense(n_vocab, activation='softmax'))
    # Compilamos el modelo con pérdida de entropía cruzada categórica y optimizador RMSprop
    modelo.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Cargamos los pesos entrenados en el modelo
    modelo.load_weights('pesos-mejora-LSTM-12-nov-198-0.5207-mejor.keras')

    return modelo  # Devolvemos el modelo creado

def generar_notas(modelo, entrada_red, nombres_notas, n_vocab):
    """ Genera notas a partir de la red neuronal basada en una secuencia de notas """
    # Seleccionamos una secuencia aleatoria de la entrada como punto de inicio
    inicio = numpy.random.randint(0, len(entrada_red) - 1)

    # Mapeamos los números enteros a las notas
    int_a_nota = dict((numero, nota) for numero, nota in enumerate(nombres_notas))

    patron = entrada_red[inicio]  # Obtenemos el patrón inicial
    salida_prediccion = []  # Lista para almacenar las notas generadas

    # Generamos 500 notas
    for indice_nota in range(500):
        # Preparamos la entrada para el modelo
        entrada_prediccion = numpy.reshape(patron, (1, len(patron), 1))
        entrada_prediccion = entrada_prediccion / float(n_vocab)

        # Realizamos la predicción de la siguiente nota
        prediccion = modelo.predict(entrada_prediccion, verbose=0)

        indice = numpy.argmax(prediccion)  # Obtenemos el índice con la mayor probabilidad
        resultado = int_a_nota[indice]  # Convertimos el índice a nota
        salida_prediccion.append(resultado)  # Agregamos la nota a la salida

        # Actualizamos el patrón para la siguiente predicción
        patron.append(indice)
        patron = patron[1:]

    return salida_prediccion  # Devolvemos las notas generadas

def crear_midi(salida_prediccion):
    """ Convierte la salida de la predicción a notas y crea un archivo MIDI """
    offset = 0  # Desplazamiento para cada nota
    notas_salida = []  # Lista para almacenar las notas y acordes

    # Creamos objetos de notas y acordes basados en los valores generados
    for patron in salida_prediccion:
        if '_' in patron:
            # Dividimos el patrón en tono/acorde y clase de duración
            tono_duracion, clase_duracion = patron.split('_')

            # Determinamos el valor real de duración basado en la clase de duración
            if clase_duracion == 'corta':
                duracion = 0.5  # Valor representativo para 'corta'
            elif clase_duracion == 'media':
                duracion = 1.0  # Valor representativo para 'media'
            else:  # 'long'
                duracion = 1.5  # Valor representativo para 'larga'

            if '.' in tono_duracion or tono_duracion.isdigit():
                # El patrón es un acorde
                notas_en_acorde = tono_duracion.split('.')
                notas = []
                for nota_actual in notas_en_acorde:
                    nueva_nota = note.Note(int(nota_actual))
                    nueva_nota.duration.quarterLength = duracion
                    nueva_nota.storedInstrument = instrument.Piano()
                    notas.append(nueva_nota)
                nuevo_acorde = chord.Chord(notas)
                nuevo_acorde.offset = offset
                nuevo_acorde.duration.quarterLength = duracion
                notas_salida.append(nuevo_acorde)
            else:
                # El patrón es una nota
                nueva_nota = note.Note(tono_duracion)
                nueva_nota.offset = offset
                nueva_nota.duration.quarterLength = duracion
                nueva_nota.storedInstrument = instrument.Piano()
                notas_salida.append(nueva_nota)
        else:
            # Manejar casos donde falta la duración (opcional)
            nueva_nota = note.Note(patron)
            nueva_nota.offset = offset
            nueva_nota.storedInstrument = instrument.Piano()
            notas_salida.append(nueva_nota)
            duracion = 0.5  # Duración por defecto

        # Incrementamos el offset por la duración para evitar que las notas se superpongan
        offset += duracion

    # Creamos un stream de música con las notas generadas
    flujo_midi = stream.Stream(notas_salida)

    # Escribimos el stream en un archivo MIDI
    flujo_midi.write('midi', fp='toledo_salida_LSTM_24_nov.mid')
    print("Archivo MIDI generado con éxito")

if __name__ == '__main__':
    generar()  # Llamamos a la función principal para generar el archivo MIDI
