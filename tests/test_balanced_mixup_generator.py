import unittest
import numpy as np
import keras
from mixup_generator.balanced_mixup_generator import BalancedMixupGenerator

class TestCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def check_only_two_non_zeros(self, one_y):
        self.assertEqual(2, np.sum(one_y != 0.0))

    def test_naive_sample(self):
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

        bmgen = BalancedMixupGenerator(X, y, batch_size=12)()
        dX, dy = next(bmgen)
        self.assertEqual(dX.shape, (12, 1, 1, 3))
        self.assertEqual(dy.shape, (12, 3))
        for one_y in dy:
            self.check_only_two_non_zeros(one_y)

if __name__ == '__main__':
    unittest.main()


