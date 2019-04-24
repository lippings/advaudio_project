from SpeechDirectoryIterator import SpeechDirectoryIterator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def get_model(classnames, in_shape):
    model = Sequential()
    model.add(Conv2D(12, (5, 5), activation='relu', input_shape=in_shape))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(25, (5, 5), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(180, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(classnames), activation='softmax'))  # Last layer with one output per class

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
    model.summary()

    return model


def main():
    model = get_model()

    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='auto', min_lr=0.00001)
    model.fit_generator(train_iterator,
                        steps_per_epoch=int(np.ceil(train_iterator.n / batch_size)),
                        epochs=6,
                        validation_data=val_iterator,
                        validation_steps=int(np.ceil(val_iterator.n / batch_size)),
                        verbose=1, callbacks=[early, reduce])