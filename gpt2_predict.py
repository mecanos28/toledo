# gpt2_predict.py

"""
Este módulo genera notas para un archivo MIDI utilizando el modelo GPT-2 mini entrenado.
"""

# Importamos las librerías necesarias
import pickle
import random
from music21 import instrument, note, stream, chord
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextGenerationPipeline

def generar():
    """Genera un archivo MIDI de piano utilizando el modelo GPT-2 mini."""
    notas_generadas = generar_musica_gpt2(longitud_generacion=500)
    crear_midi_gpt2(notas_generadas)

def generar_musica_gpt2(longitud_generacion=500):
    """Genera nuevas secuencias de notas utilizando el modelo GPT-2 mini entrenado."""
    # Cargar el modelo y el tokenizador ajustados
    tokenizer = GPT2Tokenizer.from_pretrained('./gpt2_musica')
    model = GPT2LMHeadModel.from_pretrained('./gpt2_musica')

    # Crear el pipeline de generación
    generador = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        framework='pt',
        do_sample=True
    )

    # Cargar las notas almacenadas
    with open('notas_modelos/notas_gpt2', 'rb') as filepath:
        notas = pickle.load(filepath)

    # Usar las primeras 'longitud_semilla' notas como semilla
    longitud_semilla = 50  # Puedes ajustar la longitud de la semilla
    semilla_lista = notas[:longitud_semilla]
    semilla = ' '.join(semilla_lista)

    print(f"Semilla inicial utilizada: {semilla}")

    notas_generadas = []
    intentos = 0  # Contador de intentos para evitar bucles infinitos

    while len(notas_generadas) < longitud_generacion and intentos < 20:
        # Calculamos el número de tokens necesarios para generar más notas
        tokens_faltantes = (longitud_generacion - len(notas_generadas)) * 2  # Estimación de tokens necesarios
        max_length = min(len(tokenizer.encode(semilla)) + tokens_faltantes, 1000)  # Limitar el max_length

        # Generar texto
        salida = generador(semilla, max_length=max_length, num_return_sequences=1)
        texto_generado = salida[0]['generated_text']

        # Convertir el texto generado a una lista de notas
        nuevas_notas = texto_generado.strip().split()

        # Evitar duplicados de semilla
        nuevas_notas = nuevas_notas[len(semilla_lista):]

        notas_generadas.extend(nuevas_notas)

        # Actualizar la semilla para la siguiente iteración
        semilla_lista = semilla_lista + nuevas_notas
        semilla_lista = semilla_lista[-longitud_semilla:]  # Mantener la longitud de la semilla
        semilla = ' '.join(semilla_lista)

        intentos += 1  # Incrementar el contador de intentos

    # Cortar la lista de notas a la longitud deseada
    notas_generadas = notas_generadas[:longitud_generacion]

    print(f"Total de notas generadas: {len(notas_generadas)}")
    return notas_generadas

def crear_midi_gpt2(notas_generadas):
    """Convierte la lista de notas generadas a un archivo MIDI."""
    offset = 0
    notas_salida = []

    for patron in notas_generadas:
        if not patron.strip():
            continue  # Saltar patrones vacíos
        if '_' in patron:
            partes = patron.split('_')
            if len(partes) != 2:
                print(f"Patrón inválido (formato incorrecto): '{patron}'")
                continue
            tono_duracion, clase_duracion = partes

            if clase_duracion == 'corta':
                duracion = 0.5
            elif clase_duracion == 'media':
                duracion = 1.0
            elif clase_duracion == 'larga':
                duracion = 1.5
            else:
                print(f"Clase de duración inválida: '{clase_duracion}' en patrón '{patron}'")
                continue

            if tono_duracion.isdigit() or '.' in tono_duracion:
                # El patrón es un acorde
                notas_en_acorde = tono_duracion.split('.')
                notas = []
                for nota_actual in notas_en_acorde:
                    try:
                        pitch_value = int(nota_actual)
                        nueva_nota = note.Note(pitch_value)
                        nueva_nota.duration.quarterLength = duracion
                        nueva_nota.storedInstrument = instrument.Piano()
                        notas.append(nueva_nota)
                    except ValueError:
                        print(f"Valor de nota inválido en acorde: '{nota_actual}' en patrón '{patron}'")
                if notas:
                    nuevo_acorde = chord.Chord(notas)
                    nuevo_acorde.offset = offset
                    nuevo_acorde.duration.quarterLength = duracion
                    notas_salida.append(nuevo_acorde)
                else:
                    print(f"No se pudieron crear notas válidas para el acorde en patrón '{patron}'")
            else:
                # El patrón es una nota
                try:
                    nueva_nota = note.Note(tono_duracion)
                    nueva_nota.offset = offset
                    nueva_nota.duration.quarterLength = duracion
                    nueva_nota.storedInstrument = instrument.Piano()
                    notas_salida.append(nueva_nota)
                except Exception as e:
                    print(f"Error al crear nota '{tono_duracion}' en patrón '{patron}': {e}")
                    continue
        else:
            # Manejar casos donde falta la duración
            try:
                nueva_nota = note.Note(patron)
                nueva_nota.offset = offset
                nueva_nota.duration.quarterLength = 0.5  # Duración por defecto
                nueva_nota.storedInstrument = instrument.Piano()
                notas_salida.append(nueva_nota)
            except Exception as e:
                print(f"Error al crear nota '{patron}': {e}")
                continue

        # Incrementamos el offset por la duración para evitar que las notas se superpongan
        offset += duracion

    if notas_salida:
        # Creamos un stream de música con las notas generadas
        flujo_midi = stream.Stream(notas_salida)

        # Escribimos el stream en un archivo MIDI
        flujo_midi.write('midi', fp='toledo_salida_gpt2_26_nov_v2.mid')
        print("Archivo MIDI generado con éxito")
    else:
        print("No se generaron notas válidas para crear el archivo MIDI.")


if __name__ == '__main__':
    generar()
