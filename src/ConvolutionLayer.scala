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

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum, flipud, fliplr}
import breeze.numerics._
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.util.random.XORShiftRandom

/**
* Layer properties of affine transformations, that is y=A*x+b
* @param numInMap number of inputs
* @param numOutMap number of outputs
*/
private[ann] class ConvolutionLayer(
    val numInMap: Int,
    val numOutMap: Int,
    val kernelSize: Scale,
    val inputSize: Scale) extends Layer {

  override def getInstance(weights: Vector, position: Int): LayerModel = {
    ConvolutionLayerModel(this, weights, position)
  }

  override def getInstance(seed: Long = 11L): LayerModel = {
    ConvolutionLayerModel(this, seed)
  }
}


/**
* Model of Affine layer y=A*x+b
* @param kernels kernels (matrix A)
* @param bias bias (vector b)
*/
private[ann] class ConvolutionLayerModel private(
    inMapNum: Int,
    outMapNum: Int,
    kernels: Array[Array[BDM[Double]]],
    bias: Array[Double],
    inputSize: Scale) extends LayerModel {

  type Tensor4D = Array[Array[BDM[Double]]]
  type Tensor3D = Array[BDM[Double]]
  type Tensor2D = Array[Double]

  require(inMapNum > 0)
  require(outMapNum > 0)
  require(kernels.length == inMapNum)
  require(kernels(0).length == inMapNum)
  val outputSize = new Scale(kernels(0)(0).rows, kernels(0)(0).cols)

  override val size = weights().size

  override def eval(data: BDM[Double]): BDM[Double] = {
    require(data.cols == inMapNum)

    val inputMaps = new Array[BDM[Double]](inMapNum)
    for(i <- 0 to inMapNum){
      val v = data(::, i)
      inputMaps(i) = new BDM(inputSize.x, inputSize.y, v.data)
    }

    val output = new Array[BDM[Double]](outMapNum)
    var j = 0
    val oldBias = this.bias
    while (j < outMapNum) {
      var sum: BDM[Double] = null
      var i = 0
      while (i < inMapNum) {
        val lastMap = inputMaps(i)
        val kernel = this.kernels(i)(j)
        if (sum == null) {
          sum = ConvolutionLayerModel.convnValid(lastMap, kernel)
        }
        else {
          sum += ConvolutionLayerModel.convnValid(lastMap, kernel)
        }
        i += 1
      }
      sum = sigmoid(sum + oldBias(j))
      output(j) = sum
      j += 1
    }
    val outBDM = new BDM[Double](outputSize.x * outputSize.y, outMapNum)
    for(i <- 0 to outMapNum){
      outBDM(::, i) := output(i).toDenseVector
    }
    outBDM
  }

  override def prevDelta(nextDelta: BDM[Double], input: BDM[Double]): BDM[Double] = {
    val inputMaps = new Array[BDM[Double]](inMapNum)
    for(i <- 0 to inMapNum){
      val v = input(::, i)
      inputMaps(i) = new BDM(inputSize.x, inputSize.y, v.data)
    }

    require(nextDelta.cols == outMapNum)
    val nextDeltaMaps = new Array[BDM[Double]](outMapNum)
    for(i <- 0 to outMapNum){
      val v = nextDelta(::, i)
      nextDeltaMaps(i) = new BDM(outputSize.x, outputSize.y, v.data)
    }

    val nextMapNum: Int = outMapNum
    val errors = new Array[BDM[Double]](inMapNum)
    var i = 0
    while (i < inMapNum) {
      var sum: BDM[Double] = null // sum for every kernel
      var j = 0
      while (j < nextMapNum) {
        val nextError = nextDeltaMaps(j)
        val kernel = kernels(i)(j)
        // rotate kernel by 180 degrees and get full convolution
        if (sum == null) {
          sum = ConvolutionLayerModel.convnFull(nextError, flipud(fliplr(kernel)))
        }
        else {
          sum += ConvolutionLayerModel.convnFull(nextError, flipud(fliplr(kernel)))
        }
        j += 1
      }
      errors(i) = sum
      i += 1
    }

    val outBDM = new BDM[Double](inputSize.x * inputSize.y, inMapNum)
    for(i <- 0 to outMapNum){
      outBDM(::, i) := errors(i).toDenseVector
    }
    outBDM
  }

  override def grad(delta: BDM[Double], input: BDM[Double]): Array[Double] = {
    val inputMaps = new Array[BDM[Double]](inMapNum)
    for(i <- 0 to inMapNum){
      val v = input(::, i)
      inputMaps(i) = new BDM(inputSize.x, inputSize.y, v.data)
    }

    val deltaMaps = new Array[BDM[Double]](outMapNum)
    for(i <- 0 to outMapNum){
      val v = delta(::, i)
      deltaMaps(i) = new BDM(outputSize.x, outputSize.y, v.data)
    }

    val kernelGradient = getKernelsGradient(inputMaps, deltaMaps)
    val biasGradient = getBiasGradient(deltaMaps)
    roll(kernelGradient, biasGradient)
  }

  private def roll(kernels: Tensor4D, bias: Tensor2D): Array[Double] ={
    val rows = kernels.length
    val cols = kernels(0).length
    val m = kernels(0)(0).rows * kernels(0)(0).cols
    val result = new Array[Double](m * rows * cols + bias.length)
    var offset = 0
    var i = 0
    while(i < rows){
      var j = 0
      while(j < cols){
        System.arraycopy(kernels(i)(j), 0, result, offset, m)
        offset += m
        j += 1
      }
      i += 1
    }
    System.arraycopy(bias, 0, result, offset, bias.length)
    result
  }

  /**
   * get kernels gradient
   */
  private def getKernelsGradient(input: Tensor3D, layerError: Tensor3D): Tensor4D = {
    val mapNum: Int = outMapNum
    val lastMapNum: Int = input.length
    val kernelGradient = Array.ofDim[BDM[Double]](lastMapNum, mapNum)
    var j = 0
    while (j < mapNum) {
      var i = 0
      while (i < lastMapNum) {
        val error = layerError(j)
        val deltaKernel = ConvolutionLayerModel.convnValid(input(i), error)
        kernelGradient(i)(j) = deltaKernel
        i += 1
      }
      j += 1
    }
    kernelGradient
  }

  /**
   * get bias gradient
   *
   * @param errors errors of this layer
   */
  private def getBiasGradient(errors: Tensor3D): Array[Double] = {
    val mapNum: Int = outMapNum
    var j: Int = 0
    val gradient = new Array[Double](mapNum)
    while (j < mapNum) {
      val error: BDM[Double] = errors(j)
      val deltaBias: Double = sum(error)
      gradient(j) = deltaBias
      j += 1
    }
    gradient
  }

  override def weights(): Vector = new DenseVector(roll(kernels, bias))

}

/**
* Fabric for Affine layer models
*/
private[ann] object ConvolutionLayerModel {

  /**
   * Creates a model of Affine layer
   * @param layer layer properties
   * @param weights vector with weights
   * @param position position of weights in the vector
   * @return model of Affine layer
   */
  def apply(layer: ConvolutionLayer, weights: Vector, position: Int): ConvolutionLayerModel = {
    val (w, b) = unroll(weights, position,
      layer.numInMap, layer.numOutMap, layer.kernelSize, layer.inputSize)
    new ConvolutionLayerModel(w, b)
  }

  /**
   * Creates a model of Affine layer
   * @param layer layer properties
   * @param seed seed
   * @return model of Affine layer
   */
  def apply(layer: ConvolutionLayer, seed: Long): AffineLayerModel = {
    val (w, b) = randomWeights(layer.numIn, layer.numOut, seed)
    new AffineLayerModel(w, b)
  }

  /**
   * Unrolls the weights from the vector
   * @param weights vector with weights
   * @param position position of weights for this layer
   * @param numIn number of layer inputs
   * @param numOut number of layer outputs
   * @return matrix A and vector b
   */
  def unroll(
      weights: Vector,
      position: Int,
      numIn: Int,
      numOut: Int,
      kernelSize: Scale,
      inputSize: Scale): (Array[Array[BDM[Double]]], Array[Double]) = {
    val weightsCopy = weights.toArray
    // TODO: the array is not copied to BDMs, make sure this is OK!
    val a = new BDM[Double](numOut, numIn, weightsCopy, position)
    val b = new BDV[Double](weightsCopy, position + (numOut * numIn), 1, numOut)





    (a, b)
  }

  /**
   * Generate random weights for the layer
   * @param numIn number of inputs
   * @param numOut number of outputs
   * @param seed seed
   * @return (matrix A, vector b)
   */
  def randomWeights(numIn: Int, numOut: Int, seed: Long = 11L): (BDM[Double], BDV[Double]) = {
    val rand: XORShiftRandom = new XORShiftRandom(seed)
    val weights = BDM.fill[Double](numOut, numIn){ (rand.nextDouble * 4.8 - 2.4) / numIn }
    val bias = BDV.fill[Double](numOut){ (rand.nextDouble * 4.8 - 2.4) / numIn }
    (weights, bias)
  }

  /**
   * full conv
   *
   * @param matrix
   * @param kernel
   * @return
   */
  private[ann] def convnFull(matrix: BDM[Double], kernel: BDM[Double]): BDM[Double] = {
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
    convnValid(extendMatrix, kernel)
  }

  /**
   * valid conv
   *
   * @param matrix
   * @param kernel
   * @return
   */
  private[ann] def convnValid(matrix: BDM[Double], kernel: BDM[Double]): BDM[Double] = {
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