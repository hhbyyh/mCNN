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
  def buildConvLayer(inMapNum: Int, outMapNum: Int, kernelSize: Scale): CNNLayer = {
    val layer = new ConvolutionLayer(inMapNum, outMapNum, kernelSize)
    layer
  }

  def buildSampLayer(inMapNum: Int, outMapNum: Int, scaleSize: Scale): CNNLayer = {
    val layer = new MeanPoolingLayer(inMapNum, outMapNum, scaleSize)
    layer
  }
}


abstract class CNNLayer private[mCNN](
    inMapNum: Int,
    outMapNum: Int,
    kernelSize: Scale) extends Serializable {

  def getOutMapNum: Int = outMapNum

  def forward(input: Array[BDM[Double]]): Array[BDM[Double]] = input

  def prevDelta(nextDelta: Array[BDM[Double]], input: Array[BDM[Double]]): Array[BDM[Double]]

  def grad(delta: Array[BDM[Double]],
    layerInput: Array[BDM[Double]]): (Array[Array[BDM[Double]]], Array[Double]) = null
}


