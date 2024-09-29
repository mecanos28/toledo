# Importar bibliotecas necesarias
from music21 import note, chord, stream

# Convertir secuencia de acordes y notas en un flujo de melodía
def convert_to_melody_stream(snippet):
    """Convertir una secuencia de notas y acordes en una melodía."""
    melody = []
    offset = 0
    for element in snippet:
        if "." in element or element.isdigit():
            notes = [note.Note(int(n)) for n in element.split(".")]
            chord_elem = chord.Chord(notes)
            chord_elem.offset = offset
            melody.append(chord_elem)
        else:
            note_elem = note.Note(element)
            note_elem.offset = offset
            melody.append(note_elem)
        offset += 1

    return stream.Stream(melody)
