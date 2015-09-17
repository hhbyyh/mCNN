# CNN based on Spark

A Spark machine learning package containing the implementation of classical [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network).

## Requirements

This documentation is for Spark 1.3+. Other version will probably work yet not tested.

## Features

`mCNN` supports training of a Convolutional Neural Network. Currently `ConvolutionLayer`, `MeanPoolingLayer` and 'MaxPoolingLayer` are included.

## Example

### Scala API

```scala
    // training for Mnist data set.
    val topology = new CNNTopology
    topology.addLayer(CNNLayer.buildInputLayer(new Scale(28, 28)))
    topology.addLayer(CNNLayer.buildConvLayer(6, new Scale(5, 5)))
    topology.addLayer(CNNLayer.buildSampLayer(new Scale(2, 2)))
    topology.addLayer(CNNLayer.buildConvLayer(12, new Scale(5, 5)))
    topology.addLayer(CNNLayer.buildSampLayer(new Scale(2, 2)))
    topology.addLayer(CNNLayer.buildConvLayer(12, new Scale(4, 4)))
    val cnn: CNN = new CNN(topology).setMaxIterations(1000).setMiniBatchSize(16)
    val start = System.nanoTime()
    cnn.trainOneByOne(data)
    println("Training time: " + (System.nanoTime() - start) / 1e9)
```
