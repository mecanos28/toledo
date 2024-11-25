"""
Este módulo genera notas para un archivo MIDI utilizando el modelo Transformer entrenado con longitudes de notas variables.
"""

# Importamos las librerías necesarias
import pickle  # Para cargar y guardar datos en formato binario
import numpy as np  # Biblioteca para cálculos numéricos y matrices multidimensionales
from music21 import instrument, note, stream, chord  # Librería para trabajar con datos musicales
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, Flatten
from keras.layers import MultiHeadAttention, LayerNormalization, Add
from keras.optimizers import Adam

def generar():
    """ Genera un archivo MIDI de piano utilizando el modelo Transformer """
    # Cargamos las notas utilizadas para entrenar el modelo
    with open('notas_transformer', 'rb') as filepath:
        notas = pickle.load(filepath)

    # Obtenemos todos los nombres de tonos y duraciones
    nombres_notas = sorted(set(notas))
    n_vocab = len(nombres_notas)  # Tamaño del vocabulario

    # Preparamos las secuencias de entrada para el modelo
    entrada_red, int_a_nota = preparar_secuencias(notas, nombres_notas)

    # Creamos la estructura del modelo Transformer y cargamos los pesos entrenados
    modelo = crear_transformer_model(entrada_red, n_vocab)

    # Reemplaza 'pesos_transformer-XX-XXXX.keras' con el nombre real de tu archivo de pesos
    modelo.load_weights('pesos_transformer-57-0.1235.keras')

    # Generamos notas utilizando el modelo entrenado
    salida_prediccion = generar_notas(modelo, entrada_red, int_a_nota, n_vocab)

    # Creamos un archivo MIDI a partir de las notas generadas
    crear_midi(salida_prediccion)

def preparar_secuencias(notas, nombres_notas):
    """ Prepara las secuencias utilizadas por el modelo Transformer """
    # Mapeamos las notas a números enteros y viceversa
    nota_a_int = dict((nota, numero) for numero, nota in enumerate(nombres_notas))
    int_a_nota = dict((numero, nota) for numero, nota in enumerate(nombres_notas))

    longitud_secuencia = 10  # Definimos la longitud de las secuencias de entrada
    entrada_red = []  # Lista para almacenar las secuencias de entrada

    # Creamos las secuencias de entrada a partir de las notas
    for i in range(len(notas) - longitud_secuencia):
        secuencia_entrada = notas[i:i + longitud_secuencia]
        entrada_red.append([nota_a_int[char] for char in secuencia_entrada])

    entrada_red = np.array(entrada_red)

    return entrada_red, int_a_nota  # Devolvemos las secuencias de entrada y el diccionario de conversión

def crear_transformer_model(entrada_red, n_vocab):
    """ Crea la estructura del modelo Transformer para predicción """
    import tensorflow as tf  # Importamos TensorFlow

    d_model = 256
    num_heads = 8
    dff = 512

    # Obtener input_length desde entrada_red
    input_length = entrada_red.shape[1]
    input_length = int(input_length)  # Aseguramos que es un entero

    inputs = Input(shape=(input_length,))

    # Capa de Embedding para las notas
    embedding = Embedding(input_dim=n_vocab, output_dim=d_model)(inputs)

    # Capa de Embedding para Positional Encoding
    positions = tf.range(start=0, limit=input_length, delta=1)
    positions = positions[tf.newaxis, :]  # Shape: (1, input_length)
    position_embedding_layer = Embedding(input_dim=input_length, output_dim=d_model)
    position_embeddings = position_embedding_layer(positions)

    # Sumamos el Embedding y el Positional Encoding
    x = embedding + position_embeddings  # Broadcasting sobre la dimensión del batch

    # Bloque Transformer
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn_output = Dropout(0.1)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

    ffn_output = Dense(dff, activation='relu')(out1)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(0.1)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    # Capa de Salida
    out_flat = Flatten()(out2)
    outputs = Dense(n_vocab, activation='softmax')(out_flat)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy')

    return model

def generar_notas(modelo, entrada_red, int_a_nota, n_vocab):
    """ Genera notas a partir del modelo Transformer basado en una secuencia de notas """
    # Seleccionamos una secuencia aleatoria de la entrada como punto de inicio
    inicio = np.random.randint(0, len(entrada_red) - 1) # Para comparar los modelos, este valor debe ser el mismo
    patron = entrada_red[inicio].tolist()  # Convertimos a lista

    salida_prediccion = []  # Lista para almacenar las notas generadas

    # Generamos 500 notas
    for indice_nota in range(500):
        # Preparamos la entrada para el modelo
        entrada_prediccion = np.array([patron])

        # Realizamos la predicción de la siguiente nota
        prediccion = modelo.predict(entrada_prediccion, verbose=0)

        indice = np.argmax(prediccion)  # Obtenemos el índice con la mayor probabilidad
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
            else:  # 'larga'
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
    flujo_midi.write('midi', fp='toledo_salida_transformer_24_nov.mid')
    print("Archivo MIDI generado con éxito")

if __name__ == '__main__':
    generar()  # Llamamos a la función principal para generar el archivo MIDI
