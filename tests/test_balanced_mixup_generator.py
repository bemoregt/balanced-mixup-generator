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

    def check_only_n_non_zeros(self, one_y, only_n=2):
        self.assertEqual(only_n, np.sum(one_y != 0.0))

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
            self.check_only_n_non_zeros(one_y)

    def test_non_image(self):
        N = 200
        X = np.random.beta(1, 1, (N, 10))
        y = keras.utils.to_categorical(np.random.randint(0, 5, (N, 1)))
        bmgen = BalancedMixupGenerator(X, y, batch_size=11)()
        dX, dy = next(bmgen)
        self.assertEqual(dX.shape, (11, 10))
        self.assertEqual(dy.shape, (11, 5))
        for one_y in dy:
            self.check_only_n_non_zeros(one_y)

    def test_mix_0(self):
        N = 200
        X = np.random.beta(1, 1, (N, 10, 2, 5, 3, 8))
        y = keras.utils.to_categorical(np.random.randint(0, 9, (N, 1)))
        bmgen = BalancedMixupGenerator(X, y, batch_size=11, alpha=0.0)()
        dX, dy = next(bmgen)
        self.assertEqual(dX.shape, (11, 10, 2, 5, 3, 8))
        self.assertEqual(dy.shape, (11, 9))
        for one_y in dy:
            self.check_only_n_non_zeros(one_y, only_n=1)


if __name__ == '__main__':
    unittest.main()


