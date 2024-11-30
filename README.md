# Toledo 🎶

**Toledo** is a versatile music generation project that leverages state-of-the-art neural network architectures to compose original MIDI music. Whether you're a developer, musician, or AI enthusiast, Toledo offers a comprehensive framework to train and generate music using **LSTM**, **Transformer**, and **GPT-2** models.

#### Note: Code is in Spanish as this was implemented as a university project I had to continually discuss with professor and classmates. 

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Models](#models)
  - [1. LSTM Model](#1-lstm-model)
    - [Training the LSTM Model](#training-the-lstm-model)
    - [Generating Music with LSTM](#generating-music-with-lstm)
  - [2. Transformer Model](#2-transformer-model)
    - [Training the Transformer Model](#training-the-transformer-model)
    - [Generating Music with Transformer](#generating-music-with-transformer)
  - [3. GPT-2 Model](#3-gpt-2-model)
    - [Training the GPT-2 Model](#training-the-gpt-2-model)
    - [Generating Music with GPT-2](#generating-music-with-gpt-2)
- [Switching Between Models](#switching-between-models)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Multiple Neural Architectures**: Train and generate music using LSTM, Transformer, and GPT-2 models.
- **Flexible Data Handling**: Processes MIDI files with variable note lengths.
- **Customizable Training**: Adjust model parameters and training configurations easily in code.
- **Easy Generation**: Generate MIDI files with a single command for each model.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/tu_usuario/toledo.git
   cd toledo
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *If `requirements.txt` does not work, you can install the necessary libraries 
4. manually or try with requirements2.txt*

   ```bash
   pip install numpy music21 keras tensorflow transformers
   ```

## Data Preparation

1. **Add MIDI Files**

   Place all your MIDI files in the `input/midi_songs/` directory. Make sure that the directory exists:

   ```bash
   mkdir -p input/midi_songs
   ```

2. **Verify Data**

   Make sure that the MIDI files are correctly formatted and can be parsed by `music21`.

## Models

Toledo supports three different models for music generation: **LSTM**, **Transformer**, and **GPT-2**. Each model has its own training and prediction scripts.

### 1. LSTM Model

#### Training the LSTM Model

1. **Navigate to the Project Directory**

   Make sure you're in the root directory of the project.

2. **Run the Training Script**

   ```bash
   python lstm.py
   ```

   This script performs the following:
   - **Data Processing**: Parses MIDI files to extract notes and chords with their durations.
   - **Sequence Preparation**: Creates input-output pairs for training.
   - **Model Building**: Constructs an LSTM-based neural network.
   - **Training**: Trains the model and saves the best weights based on training loss.

3. **Training Configuration**

   - **Epochs**: 200
   - **Batch Size**: 128
   - **Model Checkpoints**: Saved as `pesos-mejora-LSTM-12-nov-{epoch}-{loss}-mejor.keras`

#### Generating Music with LSTM

1. **Ensure Model Weights are Trained**

   Make sure you have trained the LSTM model and have the weights saved (e.g., `pesos-mejora-LSTM-12-nov-198-0.5207-mejor.keras`).
   In code, change pre-trained model name in the prediction file.

3. **Run the Prediction Script**

   ```bash
   python predict.py
   ```

   This script:
   - **Loads Notes**: Retrieves the processed notes from `notas`.
   - **Prepares Input Sequences**: Prepares data for the LSTM model.
   - **Loads the Trained Model**: Builds the LSTM architecture and loads the saved weights.
   - **Generates Notes**: Produces a sequence of 500 notes.
   - **Creates MIDI File**: Outputs `toledo_salida_LSTM.mid`. You can change the filename in code.

### 2. Transformer Model

#### Training the Transformer Model

1. **Run the Training Script**

   ```bash
   python transformer.py
   ```

   This script handles:
   - **Data Extraction**: Parses MIDI files to extract relevant musical information.
   - **Sequence Preparation**: Prepares data sequences for the Transformer model.
   - **Model Building**: Constructs a Transformer-based neural network.
   - **Training**: Trains the model and saves the best weights as `pesos_transformer-{epoch}-{loss}.keras`.

2. **Training Configuration**

   - **Epochs**: 100
   - **Batch Size**: 64
   - **Learning Rate**: 0.001

#### Generating Music with Transformer

1. **Ensure Model Weights are Trained**

   Confirm that you have trained the Transformer model and have the weights saved (e.g., `pesos_transformer-57-0.1235.keras`). In code, change pre-trained model name in the prediction file.

2. **Run the Prediction Script**

   ```bash
   python transformer_predict.py
   ```

   This script:
   - **Loads Notes**: Retrieves the processed notes from `notas_transformer`.
   - **Prepares Input Sequences**: Prepares data for the Transformer model.
   - **Loads the Trained Model**: Builds the Transformer architecture and loads the saved weights.
   - **Generates Notes**: Produces a sequence of 500 notes.
   - **Creates MIDI File**: Outputs `toledo_salida_transformer.mid`.

### 3. GPT-2 Model

#### Training the GPT-2 Model

1. **Run the Training Script**

   ```bash
   python gpt_musica.py
   ```

   This script performs:
   - **Data Extraction**: Parses MIDI files to extract notes and chords.
   - **Data Preparation**: Converts the list of notes into a text format suitable for GPT-2.
   - **Model Fine-Tuning**: Adjusts the GPT-2 mini model using the prepared data.
   - **Saving the Model**: Saves the fine-tuned model and tokenizer to `./gpt2_musica`.

2. **Training Configuration**

   - **Epochs**: 5
   - **Batch Size**: 4
   - **Block Size**: 128
   - **Logging**: Detailed logs are saved in `./logs`.
   - **Output Directory**: `./gpt2_musica`

#### Generating Music with GPT-2

1. **Ensure Model is Trained**

   Make sure you have trained the GPT-2 model and have the fine-tuned model saved in `./gpt2_musica`.

2. **Run the Prediction Script**

   ```bash
   python gpt2_predict.py
   ```

   This script:
   - **Loads Notes**: Retrieves the processed notes from `notas_gpt2`.
   - **Loads the Trained Model and Tokenizer**: From `./gpt2_musica`.
   - **Generates Notes**: Produces a sequence of 500 notes using the GPT-2 model.
   - **Creates MIDI File**: Outputs `toledo_salida_gpt2.mid`.

## Switching Between Models

To use a different model for generation, make sure you have trained the desired model and execute its corresponding prediction script. Here's how you can manage and switch between models:

1. **Training a Model**

   Train the model you wish to use (LSTM, Transformer, or GPT-2) by running its respective training script as detailed above.

2. **Generating with a Model**

   After training, generate music using the corresponding prediction script:
   - **LSTM**: `python predict.py`
   - **Transformer**: `python transformer_predict.py`
   - **GPT-2**: `python gpt2_predict.py`

3. **Using Multiple Models Simultaneously**

   Make sure that each model saves its outputs to different directories or filenames to avoid conflicts.

4. **Customizing Generation**

   If you wish to customize which model to use within a single script, modify the generation script to load the desired model's architecture and weights accordingly. However, it's recommended to keep training and prediction scripts separate for clarity and ease of use.

## License

[MIT](https://github.com/mecanos28/toledo/tree/main#MIT-1-ov-file)

## Acknowledgements

- [Music21](https://web.mit.edu/music21/) for music processing.
- [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/) for neural network implementations.
- [Transformers by Hugging Face](https://huggingface.co/transformers/) for the GPT-2 model.
- [Generating Music using an LSTM Neural Network | by David C Exiga | Medium][1]
- [Understanding Neural Networks in the Context of Music Generation | by Emily Thi Tran | Medium][2]
- [How to Generate Music using a LSTM Neural Network in Keras | by Sigurður Skúli | Towards Data Science][3]
- [[2306.05284] Simple and Controllable Music Generation][4]
- [MusicGen: Simple and Controllable Music Generation][5]
- [The Complete LSTM Tutorial With Implementation - Analytics Vidhya][6]
- [The Long Short-Term Memory (LSTM) Network from Scratch | Medium][7]
- [Music Generation: LSTM 🎹][8]

[1]: https://david-exiga.medium.com/music-generation-using-lstm-neural-networks-44f6780a4c5
[2]: https://medium.com/@emilytranthi/understanding-neural-networks-in-the-context-of-music-generation-c9ac5e7b465
[3]: https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
[4]: https://arxiv.org/abs/2306.05284
[5]: https://ai.honu.io/papers/musicgen/
[6]: https://www.analyticsvidhya.com/blog/2022/01/the-complete-lstm-tutorial-with-implementation/
[7]: https://medium.com/@CallMeTwitch/building-a-neural-network-zoo-from-scratch-the-long-short-term-memory-network-1cec5cf31b7
[8]: https://www.kaggle.com/code/karnikakapoor/music-generation-lstm

- The open-source community for invaluable resources and support.

---

**Happy Music Generating!** 🎹🎸🥁