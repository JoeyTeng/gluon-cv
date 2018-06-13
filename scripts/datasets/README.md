# Prepare large datasets for vision
[Gluon](https://mxnet.incubator.apache.org/gluon/) itself provides self-managed
tiny datasets such as MNIST, CIFAR-10/100, Fashion-MNIST.
However, downloading and unzipping large scale datasets are very time consuming
processes which are not appropriate to be initialized during class instantiation.
Therefore we provide convenient example scripts for existing/non-existing datasets.

All datasets requires one-time setup, and will be automatically recognized by `gluoncv`
package in the future.

## Instrctions
Please refer to our official [tutorials](http://gluon-cv.mxnet.io/build/examples_datasets/index.html)

## References
Some of the datasets are collected by other researchers. Please cite their papers if you use the data.

- For translation datasets [[Details](https://github.com/junyanz/CycleGAN/blob/master/README.md#datasets)]
  - `facades`: 400 images from the [CMP Facades dataset](http://cmp.felk.cvut.cz/~tylecr1/facade/). [[Citation](datasets/bibtex/facades.tex)]
  - `cityscapes`: 2975 images from the [Cityscapes training set](https://www.cityscapes-dataset.com/). [[Citation](datasets/bibtex/cityscapes.tex)]
