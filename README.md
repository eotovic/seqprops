## Sequential properties - peptide representation scheme
This package contains implementation of sequential properties representation scheme from the paper "Sequential properties representation scheme for recurrent neural network based prediction of therapeutic peptides". If you use this package in your work, please cite it as below or use the citation option in the side menu.

*Otović, E., Njirjak, M., Kalafatovic, D., & Mauša, G. (2022). Sequential Properties Representation Scheme for Recurrent Neural Network-Based Prediction of Therapeutic Peptides. Journal of Chemical Information and Modeling, 62(12), 2961-2972.*

### Usage
````
from seqprops import SequentialPropertiesEncoder
encoder = SequentialPropertiesEncoder()
encoder.encode(["AA", "HTTA"])
````

### Minimal working example
````
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from seqprops import SequentialPropertiesEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM

# Some input data
sequences = ["AAC", "ACACA", "AHHHTK", "HH"]
y = np.array([0, 1, 1, 0])

# Encode sequences
encoder = SequentialPropertiesEncoder(scaler=MinMaxScaler(feature_range=(-1, 1)))
X = encoder.encode(sequences)

# Define a model
model_input = Input(shape=X.shape[1:], name="input_1")
x = LSTM(32, unroll=True, name="bi_lstm")(model_input)
x = Dense(1, activation='sigmoid', name="output_dense")(x)
model = Model(inputs=model_input, outputs=x)

# Model training
adam_optimizer = keras.optimizers.Adam()
model.compile(loss="binary_crossentropy", optimizer=adam_optimizer)
model.fit(
    X, y, 
)
````

### Available properties
You can list available properties with:
````
print(encoder.get_available_properties())
````

To manually select specific properties:
````
encoder.select_properties(['MSWHIM_MSWHIM3', 'tScales_T1'])
````

For automatic feature selection, the users are referred to function <em>feature_selection</em> and usage example [here](https://github.com/eotovic/seqprops_therapeutic)