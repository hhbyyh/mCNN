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

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.{SparkConf, SparkContext}
import breeze.linalg.{DenseMatrix => BDM, _}

object Example {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val conf = new SparkConf().setMaster("local[8]").setAppName("ttt")
    val sc = new SparkContext(conf)
    val lines = sc.textFile("dataset/train.format", 8)
    val data = lines.map(line => line.split(",")).map(arr => arr.map(_.toDouble))
      .map(arr => (arr(784), Vector2Tensor(Vectors.dense(arr.slice(0, 784)))))

    val topology = new CNNTopology
    topology.addLayer(CNNLayer.buildConvolutionLayer(1, 6, new Scale(5, 5)))
    topology.addLayer(CNNLayer.buildMeanPoolingLayer(6, 6, new Scale(2, 2)))
    topology.addLayer(CNNLayer.buildConvolutionLayer(6, 12, new Scale(5, 5)))
    topology.addLayer(CNNLayer.buildMeanPoolingLayer(12, 12, new Scale(2, 2)))
    topology.addLayer(CNNLayer.buildConvolutionLayer(12, 12, new Scale(4, 4)))
    val cnn: CNN = new CNN(topology).setMaxIterations(1000).setMiniBatchSize(16)
    val start = System.nanoTime()
    cnn.trainOneByOne(data)
    println("Training time: " + (System.nanoTime() - start) / 1e9)
  }

  /**
   * set inlayer output
   * @param record
   */
  private def Vector2Tensor(record: Vector): Array[BDM[Double]] = {
    val mapSize = new Scale(28, 28)
    val m = new BDM[Double](mapSize.x, mapSize.y)
    var i: Int = 0
    while (i < mapSize.x) {
      var j: Int = 0
      while (j < mapSize.y) {
        m(i, j) = record(mapSize.x * i + j)
        j += 1
      }
      i += 1
    }
    Array(m)
  }
}


