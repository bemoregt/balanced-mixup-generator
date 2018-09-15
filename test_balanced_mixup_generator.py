import numpy as np
import keras
X = [
    [[[1.0,0.0,0.0]]],
    [[[0.8,0.0,0.1]]],
    [[[0.5,0.5,0.0]]],
    [[[0.4,0.4,0.0]]],
    [[[0.6,0.7,0.0]]],
    [[[0.1,0.5,0.5]]],
]
label_y = [
    'kylin',
    'kylin',
    'tiger',
    'tiger',
    'tiger',
    'eleph',
]
classes = sorted(list(set(label_y)))
class2int = {c:i for i, c in enumerate(classes)}
num_classes = len(classes)

X = np.array(X)
y = keras.utils.to_categorical(np.array([class2int[l] for l in label_y]))

from balanced_mixup_generator import BalancedMixupGenerator

bmgen = BalancedMixupGenerator(X, y, batch_size=12)()
dX, dy = next(bmgen)
print(dX.shape, dy.shape)
print(dy)