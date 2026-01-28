import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import random
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

# Download NLTK resources
nltk.download("punkt")
nltk.download("wordnet")

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize lists and variables
words = []
classes = []
documents = []
ignore_words = ["?", "!"]

# Load intents data from intents.json
data_file = open("intents.json").read()
intents = json.loads(data_file)

# Process intents data
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize words
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents
        documents.append((w, intent["tag"]))
        # Add classes
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Load IPC section intents data from ipc_intents.json
ipc_data_file = open("intents.json").read()
ipc_intents = json.loads(ipc_data_file)

# Process IPC section intents data
for intent in ipc_intents["intents"]:
    for pattern in intent["patterns"]:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize words and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Create training data
for doc in documents:
    # Initialize bag of words
    bag = []
    # Tokenize and lemmatize words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    # Create bag of words
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # Create output row
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert to numpy array
random.shuffle(training)
training = np.array(training, dtype=object)

# Create train and test lists
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Ensure that all sequences have the same length
train_x = pad_sequences(train_x, maxlen=len(words), padding="post")

# Convert to numpy arrays
train_x = np.array(train_x)
train_y = np.array(train_y)

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(len(words),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation="softmax"))

# Compile model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True, weight_decay=1e-6)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Fit and save the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save("chatbot_model.h5", hist)

print("Model created")
