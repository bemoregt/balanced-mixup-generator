# Class balanced mixup generator

This repository provides a class [BalancedMixupGenerator](balanced_mixup_generator.py) inherited from [MixupGenerator](https://github.com/yu4u/mixup-generator).

## Motivation

[MixupGenerator](https://github.com/yu4u/mixup-generator) implements the mixup algorithm [1] efficiently, but I need generator to be aware of class balance __even in a single batch__.

This implementation fullfills this requirement.

## BalancedMixupGenerator

This class generates class balanced batches that are mixup applied.

## Limitation

- Batch size has to be greater than or equal to number of classes.

## Install

```sh
python setup.py install
```

## References
[1] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond Empirical Risk Minimization," in arXiv:1710.09412, 2017.

[2] Z. Zhong, L. Zheng, G. Kang, S. Li, and Y. Yang, "Random Erasing Data Augmentation," in arXiv:1708.04896, 2017.
