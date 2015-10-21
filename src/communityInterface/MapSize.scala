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

import java.io.Serializable
import breeze.linalg.{DenseMatrix => BDM}

/**
 * Utility class for managing feature map size. x and y can be different
 */
class MapSize(var x: Int, var y: Int) extends Serializable {

  /**
   * divide a scale with other scale
   *
   * @param scaleSize
   * @return
   */
  private[ann] def divide(scaleSize: MapSize): MapSize = {
    val x: Int = this.x / scaleSize.x
    val y: Int = this.y / scaleSize.y
    if (x * scaleSize.x != this.x || y * scaleSize.y != this.y){
      throw new RuntimeException(this + "can not be divided" + scaleSize)
    }
    new MapSize(x, y)
  }

  private[ann] def multiply(scaleSize: MapSize): MapSize = {
    val x: Int = this.x * scaleSize.x
    val y: Int = this.y * scaleSize.y
    new MapSize(x, y)
  }

  /**
   * subtract a scale and add append
   */
  private[ann] def subtract(other: MapSize, append: Int): MapSize = {
    val x: Int = this.x - other.x + append
    val y: Int = this.y - other.y + append
    new MapSize(x, y)
  }
}

/**
 * Utility class for converting feature maps to and from a vector (one column in a matrix).
 * The conversion is necessary for compatibility with current Layer interface.
 */
object FeatureMapRolling{
  private[ann] def extractMaps(bdm: BDM[Double], size: MapSize): Array[BDM[Double]] = {
    require(bdm.cols == 1)

    val v = bdm.data
    val mapSize = size.x * size.y
    val mapNum = v.length / mapSize
    val maps = new Array[BDM[Double]](mapNum)
    var i = 0
    var offset = 0
    while(i < mapNum){
      maps(i) = new BDM(size.x, size.y, v, offset)
      offset += mapSize
      i += 1
    }
    maps
  }

  private[ann] def mergeMaps(data: Array[BDM[Double]]): BDM[Double] = {
    require(data.length > 0)
    val num = data.length
    val size = data(0).size
    val arr = new Array[Double](size * num)
    var offset = 0
    var i = 0
    while (i < num){
      System.arraycopy(data(i).toArray, 0, arr, offset, size)
      offset += size
      i += 1
    }
    val outBDM = new BDM[Double](num * size, 1, arr)
    outBDM
  }
}