# Importar bibliotecas necesarias
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adamax


# Entrenar el modelo LSTM
def train_lstm_model(X_train, y_train):
    """Entrenar el modelo LSTM con los datos proporcionados."""
    model = Sequential()
    model.add(LSTM(512, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.1))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    # Compilar el modelo
    opt = Adamax(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    # Entrenar el modelo
    model.fit(X_train, y_train, batch_size=256, epochs=10) # TODO: Cambiar a 200 o más

    return model
