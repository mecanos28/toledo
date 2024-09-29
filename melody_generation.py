# Importar bibliotecas necesarias
import numpy as np
from music21 import stream
from utils import convert_to_melody_stream


# Generar música y melodía usando el modelo
def generate_music_and_melody(model, X_seed, reverse_mapping, mapping, sequence_length=40,
                              note_count=100):
    """Generar una melodía usando el modelo entrenado."""
    seed = X_seed[np.random.randint(0, len(X_seed) - 1)]
    generated_notes = []
    for _ in range(note_count):
        seed = seed.reshape(1, sequence_length, 1)
        prediction = model.predict(seed, verbose=0)[0]
        index = np.argmax(prediction)
        generated_notes.append(index)
        next_note = index / float(len(mapping))
        seed = np.insert(seed[0], len(seed[0]), next_note)
        seed = seed[1:]

    music = [reverse_mapping[note] for note in generated_notes]
    melody_stream = convert_to_melody_stream(music)

    return music, melody_stream
