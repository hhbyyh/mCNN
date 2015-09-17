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
package hhbyyh.mCNN

import java.io.Serializable

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, fliplr, flipud, sum}
import breeze.numerics._

object CNNLayer {

  def buildInputLayer(mapSize: Scale): CNNLayer = {
    val layer: CNNLayer = new InputCNNLayer
    layer.mapNum = 1
    layer.setMapSize(mapSize)
    layer
  }

  def buildConvLayer(outMapNum: Int, kernelSize: Scale): CNNLayer = {
    val layer = new ConvolutionLayer
    layer.mapNum = outMapNum
    layer.setKernelSize(kernelSize)
    layer
  }

  def buildSampLayer(scaleSize: Scale): CNNLayer = {
    val layer = new MeanPoolingLayer
    layer.setScaleSize(scaleSize)
    layer
  }
}

/**
 * scale size for conv and sampling, can have different x and y
 */
class Scale(var x: Int, var y: Int) extends Serializable {

  /**
   * divide a scale with other scale
   *
   * @param scaleSize
   * @return
   */
  private[mCNN] def divide(scaleSize: Scale): Scale = {
    val x: Int = this.x / scaleSize.x
    val y: Int = this.y / scaleSize.y
    if (x * scaleSize.x != this.x || y * scaleSize.y != this.y){
      throw new RuntimeException(this + "can not be divided" + scaleSize)
    }
    new Scale(x, y)
  }

  private[mCNN] def multiply(scaleSize: Scale): Scale = {
    val x: Int = this.x * scaleSize.x
    val y: Int = this.y * scaleSize.y
    new Scale(x, y)
  }

  /**
   * subtract a scale and add append
   */
  private[mCNN] def subtract(other: Scale, append: Int): Scale = {
    val x: Int = this.x - other.x + append
    val y: Int = this.y - other.y + append
    new Scale(x, y)
  }
}

abstract class CNNLayer private[mCNN] extends Serializable {

  protected var layerType: String = null
  protected var mapNum: Int = 0
  private var mapSize: Scale = null

  def getOutMapNum: Int = mapNum
  def setOutMapNum(value: Int): this.type = {
    this.mapNum = value
    this
  }

  def getMapSize: Scale = mapSize
  def setMapSize(mapSize: Scale): this.type = {
    this.mapSize = mapSize
    this
  }

  def getType: String = layerType

  def forward(input: Array[BDM[Double]]): Array[BDM[Double]] = input

  def prevDelta(
      nextDelta: Array[BDM[Double]],
      input: Array[BDM[Double]]): Array[BDM[Double]] = null

  def grad(
    delta: Array[BDM[Double]],
    layerInput: Array[BDM[Double]]): (Array[Array[BDM[Double]]], Array[Double]) = null
}

class InputCNNLayer extends CNNLayer{
  this.layerType = "input"
}

class ConvolutionLayer private[mCNN] extends CNNLayer{
  private var bias: BDV[Double] = null
  private var kernel: Array[Array[BDM[Double]]] = null
  private var kernelSize: Scale = null

  this.layerType = "conv"
  private[mCNN] def initBias(frontMapNum: Int) {
    this.bias = BDV.zeros[Double](mapNum)
  }

  private[mCNN] def initKernel(frontMapNum: Int) {
    this.kernel = Array.ofDim[BDM[Double]](frontMapNum, mapNum)
    for (i <- 0 until frontMapNum)
      for (j <- 0 until mapNum)
        kernel(i)(j) = (BDM.rand[Double](kernelSize.x, kernelSize.y) - 0.05) / 10.0
  }

  def getBias: BDV[Double] = bias
  def setBias(mapNo: Int, value: Double): this.type = {
    bias(mapNo) = value
    this
  }

  def getKernelSize: Scale = kernelSize
  def setKernelSize(value: Scale): this.type = {
    this.kernelSize = value
    this
  }
  def getKernel(i: Int, j: Int): BDM[Double] = kernel(i)(j)

  override def forward(input: Array[BDM[Double]]): Array[BDM[Double]] = {
    val mapNum: Int = this.mapNum
    val lastMapNum: Int = input.length
    val output = new Array[BDM[Double]](mapNum)
    var j = 0
    val oldBias = this.getBias
    while (j < mapNum) {
      var sum: BDM[Double] = null
      var i = 0
      while (i < lastMapNum) {
        val lastMap = input(i)
        val kernel = this.getKernel(i, j)
        if (sum == null) {
          sum = CNN.convnValid(lastMap, kernel)
        }
        else {
          sum += CNN.convnValid(lastMap, kernel)
        }
        i += 1
      }
      sum = sigmoid(sum + oldBias(j))
      output(j) = sum
      j += 1
    }
    output
  }

  override def prevDelta(
      nextDelta: Array[BDM[Double]],
      layerInput: Array[BDM[Double]]): Array[BDM[Double]] = {

    val mapNum: Int = layerInput.length
    val nextMapNum: Int = this.getOutMapNum
    val errors = new Array[BDM[Double]](mapNum)
    var i = 0
    while (i < mapNum) {
      var sum: BDM[Double] = null // sum for every kernel
      var j = 0
      while (j < nextMapNum) {
        val nextError = nextDelta(j)
        val kernel = this.getKernel(i, j)
        // rotate kernel by 180 degrees and get full convolution
        if (sum == null) {
          sum = CNN.convnFull(nextError, flipud(fliplr(kernel)))
        }
        else {
          sum += CNN.convnFull(nextError, flipud(fliplr(kernel)))
        }
        j += 1
      }
      errors(i) = sum
      i += 1
    }
    errors
  }

  override def grad(
      layerError: Array[BDM[Double]],
      input: Array[BDM[Double]]): (Array[Array[BDM[Double]]], Array[Double]) = {
    val kernelGradient = getKernelsGradient(layerError, input)
    val biasGradient = getBiasGradient(layerError)
    (kernelGradient, biasGradient)
  }

  /**
   * get kernels gradient
   */
  private def getKernelsGradient(
      layerError: Array[BDM[Double]],
      input: Array[BDM[Double]]): Array[Array[BDM[Double]]] = {
    val mapNum: Int = this.getOutMapNum
    val lastMapNum: Int = input.length
    val delta = Array.ofDim[BDM[Double]](lastMapNum, mapNum)
    var j = 0
    while (j < mapNum) {
      var i = 0
      while (i < lastMapNum) {
        val error = layerError(j)
        val deltaKernel = CNN.convnValid(input(i), error)
        delta(i)(j) = deltaKernel
        i += 1
      }
      j += 1
    }
    delta
  }

  /**
   * get bias gradient
   *
   * @param errors errors of this layer
   */
  private def getBiasGradient(errors: Array[BDM[Double]]): Array[Double] = {
    val mapNum: Int = this.getOutMapNum
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
}

class MeanPoolingLayer private[mCNN] extends CNNLayer{
  private var scaleSize: Scale = null
  this.layerType = "samp"

  def getScaleSize: Scale = scaleSize
  def setScaleSize(value: Scale): this.type = {
    this.scaleSize = value
    this
  }

  override def forward(input: Array[BDM[Double]]): Array[BDM[Double]] = {
    val lastMapNum: Int = input.length
    val output = new Array[BDM[Double]](lastMapNum)
    var i: Int = 0
    while (i < lastMapNum) {
      val lastMap: BDM[Double] = input(i)
      val scaleSize: Scale = this.getScaleSize
      output(i) = CNN.scaleMatrix(lastMap, scaleSize)
      i += 1
    }
    output
  }

  override def prevDelta(
      nextDelta: Array[BDM[Double]],
      layerInput: Array[BDM[Double]]): Array[BDM[Double]] = {
    val mapNum: Int = layerInput.length
    val errors = new Array[BDM[Double]](mapNum)
    var m: Int = 0
    val scale: Scale = this.getScaleSize
    while (m < mapNum) {
      val nextError: BDM[Double] = nextDelta(m)
      val map: BDM[Double] = layerInput(m)
      var outMatrix: BDM[Double] = (1.0 - map)
      outMatrix = map :* outMatrix
      outMatrix = outMatrix :* CNN.kronecker(nextError, scale)
      errors(m) = outMatrix
      m += 1
    }
    errors
  }

}
