# gpt2_musica.py

"""
Este módulo prepara datos de archivos MIDI y los utiliza para ajustar el modelo GPT-2 mini para generar música, incluyendo logs detallados del entrenamiento.
"""

# Importamos las librerías necesarias
import glob
import pickle
from music21 import converter, instrument, note, chord
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import logging

# Configuración del registro (logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)

def entrenar_red():
    """Entrena el modelo GPT-2 mini para generar música."""
    logger.info("Iniciando el proceso de entrenamiento.")
    notas = obtener_notas()
    texto_entrenamiento = preparar_datos_gpt2(notas)
    ajustar_gpt2()
    logger.info("Proceso de entrenamiento completado.")

def obtener_notas():
    """Obtiene todas las notas y acordes con sus duraciones de los archivos MIDI en el directorio ./input/midi_songs."""
    notas = []

    for archivo in glob.glob("input/midi_songs/*.mid"):
        logger.info(f"Analizando {archivo}")
        midi = converter.parse(archivo)

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

    # Guardamos las notas en un archivo para usarlas posteriormente
    with open('notas_modelos/notas_gpt2', 'wb') as filepath:
        pickle.dump(notas, filepath)
    logger.info(f"Total de notas extraídas: {len(notas)}")
    return notas

def preparar_datos_gpt2(notas):
    """Convierte la lista de notas en una cadena de texto para GPT-2."""
    logger.info("Preparando los datos para GPT-2.")
    texto = ' '.join(notas)
    with open('datos_gpt2.txt', 'w') as f:
        f.write(texto)
    logger.info("Datos preparados y guardados en 'datos_gpt2.txt'.")
    return texto

def ajustar_gpt2():
    """Ajusta el modelo GPT-2 mini con los datos musicales."""
    logger.info("Iniciando el ajuste fino del modelo GPT-2 mini.")

    # Cargar el tokenizador y el modelo GPT-2 mini preentrenado
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Añadir tokens especiales si es necesario
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    logger.info("Tokenizador y modelo GPT-2 cargados correctamente.")

    # Crear el conjunto de datos
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path='datos_gpt2.txt',
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    logger.info("Conjunto de datos creado.")

    # Configuración del entrenamiento
    training_args = TrainingArguments(
        output_dir='./gpt2_musica',
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=50,  # Registra cada 50 pasos
        logging_dir='./logs',  # Directorio para los logs
        logging_first_step=True,
        evaluation_strategy='no',
    )

    logger.info("Argumentos de entrenamiento configurados.")

    # Entrenador
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    logger.info("Entrenador inicializado. Comenzando el entrenamiento...")

    # Entrenamiento
    trainer.train()

    logger.info("Entrenamiento completado. Guardando el modelo...")

    # Guardar el modelo ajustado
    model.save_pretrained('./gpt2_musica')
    tokenizer.save_pretrained('./gpt2_musica')

    logger.info("Modelo y tokenizador guardados en './gpt2_musica'.")

if __name__ == '__main__':
    entrenar_red()
