"""
Este módulo prepara datos de archivos MIDI y los alimenta a un modelo Transformer para entrenamiento con longitudes de notas variables.
"""

# Importamos las librerías necesarias
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, Flatten
from keras.layers import MultiHeadAttention, LayerNormalization, Add
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

def entrenar_transformer():
    """ Entrena un modelo Transformer para generar música """
    notas = obtener_notas()  # Obtenemos todas las notas y acordes de los archivos MIDI

    n_vocab = len(set(notas))  # Tamaño del vocabulario

    entrada_red, salida_red = preparar_secuencias(notas, n_vocab)

    modelo = crear_transformer(entrada_red, n_vocab)

    entrenar_modelo(modelo, entrada_red, salida_red)

def obtener_notas():
    """ Obtiene todas las notas y acordes con sus duraciones de los archivos MIDI en el directorio ./input/midi_songs """
    notas = []

    for archivo in glob.glob("input/midi_songs/*.mid"):
        midi = converter.parse(archivo)
        print(f"Analizando {archivo}")

        notas_a_analizar = None

        try:
            s2 = instrument.partitionByInstrument(midi)
            notas_a_analizar = s2.parts[0].recurse()
        except:
            notas_a_analizar = midi.flat.notes

        for elemento in notas_a_analizar:
            duracion = elemento.duration.quarterLength

            if duracion < 0.75:
                clase_duracion = 'corta'
            elif duracion < 1.5:
                clase_duracion = 'media'
            else:
                clase_duracion = 'larga'

            if isinstance(elemento, note.Note):
                nota_str = f"{str(elemento.pitch)}_{clase_duracion}"
                notas.append(nota_str)
            elif isinstance(elemento, chord.Chord):
                acorde_str = f"{'.'.join(str(n) for n in elemento.normalOrder)}_{clase_duracion}"
                notas.append(acorde_str)

    with open('notas_transformer', 'wb') as filepath:
        pickle.dump(notas, filepath)

    return notas

def preparar_secuencias(notas, n_vocab):
    """ Prepara las secuencias utilizadas por el Transformer """
    longitud_secuencia = 10

    nombres_notas = sorted(set(notas))

    nota_a_int = dict((nota, numero) for numero, nota in enumerate(nombres_notas))

    entrada_red = []
    salida_red = []

    for i in range(len(notas) - longitud_secuencia):
        secuencia_entrada = notas[i:i + longitud_secuencia]
        secuencia_salida = notas[i + longitud_secuencia]
        entrada_red.append([nota_a_int[char] for char in secuencia_entrada])
        salida_red.append(nota_a_int[secuencia_salida])

    entrada_red = np.array(entrada_red)
    salida_red = to_categorical(salida_red, num_classes=n_vocab)

    return entrada_red, salida_red

def crear_transformer(entrada_red, n_vocab):
    """ Crea la estructura del modelo Transformer """
    import tensorflow as tf  # Importamos TensorFlow

    d_model = 256  # Dimensión del embedding
    num_heads = 8  # Número de cabezas de atención
    dff = 512      # Dimensión de la capa feed-forward

    # Obtain input_length from entrada_red
    input_length = entrada_red.shape[1]
    input_length = int(input_length)  # Ensure it's an integer

    # Debugging statements
    print(f"input_length: {input_length}")
    print(f"type(input_length): {type(input_length)}")

    inputs = Input(shape=(input_length,))

    # Capa de Embedding para las notas
    embedding = Embedding(input_dim=n_vocab, output_dim=d_model)(inputs)

    # Positional Embedding Layer
    positions = tf.range(start=0, limit=input_length, delta=1)
    positions = positions[tf.newaxis, :]  # Shape: (1, input_length)
    position_embedding_layer = Embedding(input_dim=input_length, output_dim=d_model)
    position_embeddings = position_embedding_layer(positions)

    # Add positional embeddings to token embeddings
    x = embedding + position_embeddings  # Broadcasting over batch dimension

    # Rest of the Transformer block
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




def entrenar_modelo(modelo, entrada_red, salida_red):
    """ Entrena el modelo Transformer """
    filepath = "pesos_transformer-{epoch:02d}-{loss:.4f}.keras"

    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    modelo.fit(entrada_red, salida_red, epochs=100, batch_size=64, callbacks=callbacks_list)

if __name__ == '__main__':
    entrenar_transformer()
