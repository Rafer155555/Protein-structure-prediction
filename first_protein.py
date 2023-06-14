import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Dropout
from tensorflow.keras.models import Sequential
# Load the data
data = np.loadtxt('data/protein_data.csv', delimiter=',')
X = data[:, :-1]
y = data[:, -1]
# Create the model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(128))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# Train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100)
# Evaluate the model
score = model.evaluate(X, y)
print('Accuracy:', score[1])
