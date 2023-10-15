import numpy as np
import trax
from trax import layers as tl


def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x))



def show_layers(model, layer_prefix="Serial.sublayers"):
    print(f"Total layers: {len(model.sublayers)}\n")
    for i in range(len(model.sublayers)):
        print('========')
        print(f'{layer_prefix}_{i}: {model.sublayers[i]}\n')