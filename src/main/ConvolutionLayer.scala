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

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, fliplr, flipud, sum}
import breeze.numerics._

class ConvolutionLayer private[mCNN](inMapNum: Int, outMapNum: Int, kernelSize: Scale)
  extends CNNLayer{

  private var bias: BDV[Double] = null
  private var kernel: Array[Array[BDM[Double]]] = null
  initBias(inMapNum)
  initKernel(inMapNum)

  def getOutMapNum: Int = outMapNum

  private[mCNN] def initBias(frontMapNum: Int) {
    this.bias = BDV.zeros[Double](outMapNum)
  }

  private[mCNN] def initKernel(frontMapNum: Int) {
    this.kernel = Array.ofDim[BDM[Double]](frontMapNum, outMapNum)
    for (i <- 0 until frontMapNum)
      for (j <- 0 until outMapNum)
        kernel(i)(j) = (BDM.rand[Double](kernelSize.x, kernelSize.y) - 0.05) / 10.0
  }

  def getBias: BDV[Double] = bias
  def setBias(mapNo: Int, value: Double): this.type = {
    bias(mapNo) = value
    this
  }

  def getKernelSize: Scale = kernelSize
  def getKernel(i: Int, j: Int): BDM[Double] = kernel(i)(j)

  override def forward(input: Array[BDM[Double]]): Array[BDM[Double]] = {
    val mapNum: Int = this.outMapNum
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
          sum = ConvolutionLayer.convnValid(lastMap, kernel)
        }
        else {
          sum += ConvolutionLayer.convnValid(lastMap, kernel)
        }
        i += 1
      }
      sum = sigmoid(sum + oldBias(j))
      output(j) = sum
      j += 1
    }
    output
  }

  override def prevDelta(nextDelta: Array[BDM[Double]],
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
          sum = ConvolutionLayer.convnFull(nextError, flipud(fliplr(kernel)))
        }
        else {
          sum += ConvolutionLayer.convnFull(nextError, flipud(fliplr(kernel)))
        }
        j += 1
      }
      errors(i) = sum
      i += 1
    }
    errors
  }

  override def grad(layerError: Array[BDM[Double]],
      input: Array[BDM[Double]]): (Array[Array[BDM[Double]]], Array[Double]) = {
    val kernelGradient = getKernelsGradient(layerError, input)
    val biasGradient = getBiasGradient(layerError)
    (kernelGradient, biasGradient)
  }

  /**
   * get kernels gradient
   */
  private def getKernelsGradient(layerError: Array[BDM[Double]],
      input: Array[BDM[Double]]): Array[Array[BDM[Double]]] = {
    val mapNum: Int = this.getOutMapNum
    val lastMapNum: Int = input.length
    val delta = Array.ofDim[BDM[Double]](lastMapNum, mapNum)
    var j = 0
    while (j < mapNum) {
      var i = 0
      while (i < lastMapNum) {
        val error = layerError(j)
        val deltaKernel = ConvolutionLayer.convnValid(input(i), error)
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

object ConvolutionLayer{

  /**
   * full conv
   *
   * @param matrix
   * @param kernel
   * @return
   */
  private[mCNN] def convnFull(matrix: BDM[Double], kernel: BDM[Double]): BDM[Double] = {
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
  private[mCNN] def convnValid(matrix: BDM[Double], kernel: BDM[Double]): BDM[Double] = {
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
