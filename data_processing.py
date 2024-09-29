# Importar bibliotecas necesarias
import os
from collections import Counter

import tensorflow
from music21 import converter, instrument, note, chord, stream
import numpy as np
from sklearn.model_selection import train_test_split


# Cargar archivos MIDI
def load_midi_files(filepath):
    """Cargar archivos MIDI desde un directorio."""
    all_midis = []
    for i in os.listdir(filepath):
        if i.endswith(".mid"):
            midi = converter.parse(filepath + i)
            all_midis.append(midi)
            print("Archivo MIDI cargado:", i)
    return all_midis


# Extraer notas de archivos MIDI
def extract_notes_from_midi(midi_files):
    """Extraer notas de los archivos MIDI."""
    notes = []
    for file in midi_files:
        parts = instrument.partitionByInstrument(file)
        for part in parts.parts:
            elements = part.recurse()
            for element in elements:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))
    return notes


# Procesar el corpus de notas
def process_corpus(midi_files):
    """Procesar el corpus de notas de los archivos MIDI."""
    corpus = extract_notes_from_midi(midi_files)
    count = Counter(corpus)
    print("Total de notas en el corpus:", len(corpus))
    return corpus, count


# Eliminar notas infrecuentes
def remove_infrequent_notes(corpus, threshold):
    """Eliminar notas que aparecen menos de 'threshold' veces."""
    count = Counter(corpus)
    infrequent_notes = [note for note, count in count.items() if count < threshold]
    filtered_corpus = [note for note in corpus if note not in infrequent_notes]
    return filtered_corpus


# Preparar características y objetivos para el modelo
def prepare_data_for_training(corpus, sequence_length=40):
    """Preparar los datos de entrada y salida para entrenar el modelo."""
    unique_notes = sorted(list(set(corpus)))
    mapping = dict((c, i) for i, c in enumerate(unique_notes))
    reverse_mapping = dict((i, c) for i, c in enumerate(unique_notes))

    features = []
    targets = []
    for i in range(len(corpus) - sequence_length):
        seq_in = corpus[i:i + sequence_length]
        seq_out = corpus[i + sequence_length]
        features.append([mapping[note] for note in seq_in])
        targets.append(mapping[seq_out])

    X = np.reshape(features, (len(targets), sequence_length, 1)) / float(len(unique_notes))
    y = tensorflow.keras.utils.to_categorical(targets)

    X_train, X_seed, y_train, y_seed = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_seed, y_train, y_seed, mapping, reverse_mapping
