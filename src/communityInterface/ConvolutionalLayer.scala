/*
* Licensed to the Apache Software Foundation (ASF) under one or more
* contributor license agreements.  See the NOTICE file distributed with
* this work for additional information regarding copyright ownership.
* The ASF licenses this file to You under the Apache License, Version 2.0
* (the "License"); you may not use this file except in compliance with
* the License.  You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
package org.apache.spark.ml.ann

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, fliplr, flipud, sum}
import org.apache.spark.mllib.linalg.{DenseVector, Vector}

/**
 * Layer properties of Convolutional Layer
 * @param numInMap number of inputs
 * @param numOutMap number of outputs
 */
private[ann] class ConvolutionalLayer(
    val numInMap: Int,
    val numOutMap: Int,
    val kernelSize: MapSize,
    val inputSize: MapSize) extends Layer {

  override def getInstance(weights: Vector, position: Int): LayerModel = {
    ConvolutionalLayerModel(this, weights, position)
  }

  override def getInstance(seed: Long = 11L): LayerModel = {
    ConvolutionalLayerModel(this, seed)
  }
}

/**
 * ConvolutionLayerModel contains multiple kernels and applies a 2D convolution over the input maps.
 * @param inMapNum number of input feature maps
 * @param outMapNum number of output feature maps
 * @param kernels weights in the form of inNum * outNum * kernelSize
 * @param bias bias for each outMap, with length equal to outMapNum
 * @param inputMapSize size of each input feature map
 */
private[ann] class ConvolutionalLayerModel private(
    inMapNum: Int,
    outMapNum: Int,
    kernels: Array[Array[BDM[Double]]],
    bias: Array[Double],
    inputMapSize: MapSize) extends LayerModel {

  require(kernels.length == inMapNum)
  require(kernels.length > 0 && kernels(0).length == outMapNum)
  require(bias.length == outMapNum)

  private val kernelSize = new MapSize(kernels(0)(0).rows, kernels(0)(0).cols)
  private val outputSize = inputMapSize.subtract(kernelSize, 1)

  override val size = kernelSize.x * kernelSize.y * inMapNum * outMapNum + bias.length

  /**
    * @param data contains only one column with all the data for one sample, so its size is
    *             inMapNum * inputMapSize. CNN does not benefit from stacked input
    */
  override def eval(data: BDM[Double]): BDM[Double] = {
    require(data.rows == this.inMapNum * inputMapSize.x * inputMapSize.y)
    require(data.cols == 1)
    // local copy
    val inMapNum = this.inMapNum
    val outMapNum = this.outMapNum
    val kernels = this.kernels
    val bias = this.bias

    val inputMaps = FeatureMapRolling.extractMaps(data, inputMapSize)
    val output = new Array[BDM[Double]](this.outMapNum)
    var j = 0
    while (j < outMapNum) {
      var sum: BDM[Double] = ConvolutionalLayerModel.convValid(inputMaps(0), kernels(0)(j))
      var i = 1
      while (i < inMapNum) {
        sum += ConvolutionalLayerModel.convValid(inputMaps(i), kernels(i)(j))
        i += 1
      }
      output(j) = sum + bias(j)
      j += 1
    }
    // reorganize feature maps to a single column in a dense matrix.
    val outBDM = FeatureMapRolling.mergeMaps(output)
    outBDM
  }

  override def prevDelta(nextDelta: BDM[Double], input: BDM[Double]): BDM[Double] = {
    require(nextDelta.cols == 1)
    // local copy
    val inMapNum = this.inMapNum
    val outMapNum = this.outMapNum
    val kernels = this.kernels

    val nextDeltaMaps = FeatureMapRolling.extractMaps(nextDelta, outputSize)
    val deltas = new Array[BDM[Double]](inMapNum)
    var i = 0
    while (i < inMapNum) {
      // rotate kernel by 180 degrees and get full convolution
      var sum: BDM[Double] = ConvolutionalLayerModel.convFull(nextDeltaMaps(0),
        flipud(fliplr(kernels(i)(0))))
      var j = 1
      while (j < outMapNum) {
        sum += ConvolutionalLayerModel.convFull(nextDeltaMaps(j), flipud(fliplr(kernels(i)(j))))
        j += 1
      }
      deltas(i) = sum
      i += 1
    }
    // reorganize delta maps to a single column in a dense matrix.
    FeatureMapRolling.mergeMaps(deltas)
  }

  override def grad(delta: BDM[Double], input: BDM[Double]): Array[Double] = {
    val inputMaps = FeatureMapRolling.extractMaps(input, inputMapSize)
    val deltaMaps = FeatureMapRolling.extractMaps(delta, outputSize)

    val kernelGradient = getKernelsGradient(inputMaps, deltaMaps)
    val biasGradient = getBiasGradient(deltaMaps)
    ConvolutionalLayerModel.roll(kernelGradient, biasGradient)
  }

  /**
   * get kernels gradient
   */
  private def getKernelsGradient(input: Array[BDM[Double]],
      delta: Array[BDM[Double]]): Array[Array[BDM[Double]]] = {
    val outMapNum = this.outMapNum
    val inMapNum = this.inMapNum
    val kernelGradient = Array.ofDim[BDM[Double]](inMapNum, outMapNum)
    var j = 0
    while (j < outMapNum) {
      var i = 0
      while (i < inMapNum) {
        kernelGradient(i)(j) = ConvolutionalLayerModel.convValid(input(i), delta(j))
        i += 1
      }
      j += 1
    }
    kernelGradient
  }

  /**
   * get bias gradient
   *
   * @param deltas errors of this layer
   */
  private def getBiasGradient(deltas: Array[BDM[Double]]): Array[Double] = {
    val outMapNum = this.outMapNum
    
    val gradient = new Array[Double](outMapNum)
    var j: Int = 0
    while (j < outMapNum) {
      gradient(j) = sum(deltas(j))
      j += 1
    }
    gradient
  }

  override def weights(): Vector = new DenseVector(ConvolutionalLayerModel.roll(kernels, bias))

}

/**
* Fabric for Convolutional layer models
*/
private[ann] object ConvolutionalLayerModel {

  /**
   * Creates a model of Convolutional layer
   * @param layer layer properties
   * @param weights vector with weights
   * @param position position of weights in the vector
   * @return model of Convolutional layer
   */
  def apply(layer: ConvolutionalLayer, weights: Vector, position: Int): ConvolutionalLayerModel = {
    val (w, b) = unroll(weights, position,
      layer.numInMap,
      layer.numOutMap,
      layer.kernelSize,
      layer.inputSize)
    new ConvolutionalLayerModel(layer.numInMap, layer.numOutMap, w, b, layer.inputSize)
  }

  /**
   * Creates a model of Affine layer
   * @param layer layer properties
   * @param seed seed
   * @return model of Affine layer
   */
  def apply(layer: ConvolutionalLayer, seed: Long): ConvolutionalLayerModel = {
    val bias = new Array[Double](layer.numOutMap) // bias init to 0
    val kernel = Array.ofDim[BDM[Double]](layer.numInMap, layer.numOutMap)
    for (i <- 0 until layer.numInMap)
      for (j <- 0 until layer.numOutMap)
        kernel(i)(j) = (BDM.rand[Double](layer.kernelSize.x, layer.kernelSize.y) - 0.05) / 10.0

    new ConvolutionalLayerModel(layer.numInMap, layer.numOutMap, kernel, bias, layer.inputSize)
  }

  private[ann] def roll(kernels: Array[Array[BDM[Double]]], bias: Array[Double]): Array[Double] = {
    val rows = kernels.length
    val cols = kernels(0).length
    val m = kernels(0)(0).rows * kernels(0)(0).cols
    val result = new Array[Double](m * rows * cols + bias.length)
    var offset = 0
    var i = 0
    while(i < rows){
      var j = 0
      while(j < cols){
        System.arraycopy(kernels(i)(j).toArray, 0, result, offset, m)
        offset += m
        j += 1
      }
      i += 1
    }
    System.arraycopy(bias, 0, result, offset, bias.length)
    result
  }

  /**
   * Unrolls the weights from the vector
   * @param weights vector with weights
   * @param position position of weights for this layer
   * @param numIn number of layer inputs
   * @param numOut number of layer outputs
   * @return matrix A and vector b
   */
  def unroll(weights: Vector,
      position: Int,
      numIn: Int,
      numOut: Int,
      kernelSize: MapSize,
      inputSize: MapSize): (Array[Array[BDM[Double]]], Array[Double]) = {
    val weightsCopy = weights.toArray
    var offset = position
    val kernels = new Array[Array[BDM[Double]]](numIn)
    for(i <- 0 until numIn){
      kernels(i) = new Array[BDM[Double]](numOut)
      for(j <- 0 until numOut){
        val a = new BDM[Double](kernelSize.x, kernelSize.y, weightsCopy, offset)
        kernels(i)(j) = a
        offset += kernelSize.x * kernelSize.y
      }
    }

    val b = new BDV[Double](weightsCopy, offset, 1, numOut).toArray
    (kernels, b)
  }

  /**
   * full conv
   *
   * @param matrix
   * @param kernel
   * @return
   */
  private[ann] def convFull(matrix: BDM[Double], kernel: BDM[Double]): BDM[Double] = {
    val m: Int = matrix.rows
    val n: Int = matrix.cols
    val km: Int = kernel.rows
    val kn: Int = kernel.cols
    val extendMatrix = new BDM[Double](m + 2 * (km - 1), n + 2 * (kn - 1))
    var i = 0
    var j = 0
    while (i < m) {
      while (j < n) {
        extendMatrix(i + km - 1, j + kn - 1) = matrix(i, j)
        j += 1
      }
      i += 1
    }
    convValid(extendMatrix, kernel)
  }

  /**
   * valid conv
   *
   * @param matrix
   * @param kernel
   * @return
   */
  private[ann] def convValid(matrix: BDM[Double], kernel: BDM[Double]): BDM[Double] = {
    val m: Int = matrix.rows
    val n: Int = matrix.cols
    val km: Int = kernel.rows
    val kn: Int = kernel.cols
    val kns: Int = n - kn + 1
    val kms: Int = m - km + 1
    val outMatrix: BDM[Double] = new BDM[Double](kms, kns)
    var i = 0
    while (i < kms) {
      var j = 0
      while (j < kns) {
        var sum = 0.0
        for (ki <- 0 until km) {
          for (kj <- 0 until kn)
            sum += matrix(i + ki, j + kj) * kernel(ki, kj)
        }
        outMatrix(i, j) = sum
        j += 1
      }
      i += 1
    }
    outMatrix
  }
}