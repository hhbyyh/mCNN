# CNN based on Spark

A Spark machine learning package containing the implementation of classical [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network).

## Requirements

This documentation is for Spark 1.3+. Other version will probably work yet not tested.

## Features

`mCNN` supports training of a Convolutional Neural Network. Currently `ConvolutionLayer`, `MeanPoolingLayer` are included. `MaxPooling` and `SumPooling` are under test.
A version compatible to the ANN interface in Spark 1.5 is also under development in the communityInterface folder.

## Example

### Scala API

```scala
    // training for Mnist data set.
    val topology = new CNNTopology
    topology.addLayer(CNNLayer.buildConvolutionLayer(1, 6, new Scale(5, 5)))
    topology.addLayer(CNNLayer.buildMeanPoolingLayer(6, 6, new Scale(2, 2)))
    topology.addLayer(CNNLayer.buildConvolutionLayer(6, 12, new Scale(5, 5)))
    topology.addLayer(CNNLayer.buildMeanPoolingLayer(12, 12, new Scale(2, 2)))
    topology.addLayer(CNNLayer.buildConvolutionLayer(12, 12, new Scale(4, 4)))
    val cnn: CNN = new CNN(topology).setMaxIterations(1000).setMiniBatchSize(16)
    cnn.trainOneByOne(data)
```
