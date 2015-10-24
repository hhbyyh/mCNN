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

import breeze.linalg.{DenseMatrix => BDM, _}
import org.apache.spark.mllib.linalg.{Vector, Vectors}

/**
 * Layer properties of Mean Pooling transformations
 * @param poolingSize number of inputs
 */
private[ann] class MeanPoolingLayer(val poolingSize: MapSize, val inputSize: MapSize) extends Layer {

  override def getInstance(weights: Vector, position: Int): LayerModel = getInstance(0L)

  override def getInstance(seed: Long = 11L): LayerModel = MeanPoolingLayerModel(this, inputSize)
}

private[ann] class MeanPoolingLayerModel private(poolingSize: MapSize,
    inputSize: MapSize) extends LayerModel {

  val outputSize = inputSize.divide(poolingSize)

  override val size = 0

  override def eval(data: BDM[Double]): BDM[Double] = {
    val inMapNum = data.rows / (inputSize.x * inputSize.y)
    val batchOutput = new BDM[Double](outputSize.x * outputSize.y * inMapNum , data.cols)
    (0 until data.cols).foreach { col =>
      val inputMaps = FeatureMapRolling.extractMaps(data(::, col), inputSize)
      val inputMapNum = inputMaps.length
      val scaleSize: MapSize = this.poolingSize

      val output = new Array[BDM[Double]](inputMapNum)
      var i = 0
      while (i < inputMapNum) {
        val inputMap: BDM[Double] = inputMaps(i)
        output(i) = MeanPoolingLayerModel.avgPooling(inputMap, scaleSize)
        i += 1
      }
      batchOutput(::, col) := FeatureMapRolling.mergeMaps(output)
    }
    batchOutput
  }

  override def prevDelta(nextDelta: BDM[Double], output: BDM[Double]): BDM[Double] = {

    val inMapNum = output.rows / (outputSize.x * outputSize.y)
    val batchDelta = new BDM[Double](inputSize.x * inputSize.y * inMapNum, output.cols)
    (0 until output.cols).foreach { col =>
      val nextDeltaMaps = FeatureMapRolling.extractMaps(nextDelta(::, col), outputSize)
      val mapNum: Int = nextDeltaMaps.length
      val errors = new Array[BDM[Double]](mapNum)
      var m: Int = 0
      val scale: MapSize = this.poolingSize
      while (m < mapNum) {
        val nextError: BDM[Double] = nextDeltaMaps(m)
        val ones = BDM.ones[Double](scale.x, scale.y)
        val outMatrix = kron(nextError, ones)
        errors(m) = outMatrix
        m += 1
      }

      batchDelta(::, col) := FeatureMapRolling.mergeMaps(errors)
    }
    batchDelta
  }

  override def grad(delta: BDM[Double], input: BDM[Double]): Array[Double] = {
    new Array[Double](0)
  }

  override def weights(): Vector = Vectors.dense(new Array[Double](0))

}

/**
 * Fabric for mean pooling layer models
 */
private[ann] object MeanPoolingLayerModel {

  /**
   * Creates a model of Mean Pooling layer
   * @param layer layer properties
   * @return model of Mean Pooling layer
   */
  def apply(layer: MeanPoolingLayer, inputSize: MapSize): MeanPoolingLayerModel = {
    new MeanPoolingLayerModel(layer.poolingSize, inputSize: MapSize)
  }

  /**
   * return a new matrix that has been scaled down
   *
   * @param matrix
   */
  private[ann] def avgPooling(matrix: BDM[Double], scale: MapSize): BDM[Double] = {
    val m: Int = matrix.rows
    val n: Int = matrix.cols
    val scaleX = scale.x
    val scaleY = scale.y
    val sm: Int = m / scaleX
    val sn: Int = n / scaleY
    val outMatrix = new BDM[Double](sm, sn)
    val size = scaleX * scaleY

    var i = 0  // iterate through blocks
    while (i < sm) {
      var j = 0
      while (j < sn) {
        var sum = 0.0 // initial to left up corner of the block
        var bi = i * scaleX // block i
        val biMax = (i + 1) * scaleX
        val bjMax = (j + 1) * scaleY
        while (bi < biMax) {
          var bj = j * scaleY // block j
          while (bj < bjMax) {
            sum += matrix(bi, bj)
            bj += 1
          }
          bi += 1
        }
        outMatrix(i, j) = sum / size
        j += 1
      }
      i += 1
    }
    outMatrix
  }
}