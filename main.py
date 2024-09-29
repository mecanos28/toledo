# Importar funciones necesarias desde otros archivos
from data_processing import load_midi_files, process_corpus, remove_infrequent_notes, prepare_data_for_training
from model_training import train_lstm_model
from melody_generation import generate_music_and_melody

# Archivo principal que ejecuta todo el flujo

# Cargar archivos MIDI
midi_files = load_midi_files("./input/chopinjust1/")

# Procesar el corpus de notas
corpus, unique_notes = process_corpus(midi_files)

# Eliminar las notas infrecuentes
filtered_corpus = remove_infrequent_notes(corpus, 100)

# Preparar los datos para entrenar
X_train, X_seed, y_train, y_seed, mapping, reverse_mapping = prepare_data_for_training(filtered_corpus)

# Entrenar el modelo
model = train_lstm_model(X_train, y_train)

# Generar la melodía usando el modelo entrenado
music_notes, melody_stream = generate_music_and_melody(model, X_seed, reverse_mapping, mapping)

# Guardar la melodía generada en un archivo MIDI
melody_stream.write('midi', 'Melody_Generated.mid')
