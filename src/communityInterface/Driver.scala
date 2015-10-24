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

import org.apache.log4j.{Logger, Level}
import breeze.linalg.{DenseMatrix => BDM}
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.{SparkContext, SparkConf}

object CNNDriver {

  def main(args: Array[String]) {

    val myLayers = new Array[Layer](8)
    myLayers(0) = new ConvolutionalLayer(1, 6, kernelSize = new MapSize(5, 5), inputMapSize = new MapSize(28, 28))
    myLayers(1) = new FunctionalLayer(new SigmoidFunction())
    myLayers(2) = new MeanPoolingLayer(new MapSize(2, 2), new MapSize(24, 24))
    myLayers(3) = new ConvolutionalLayer(6, 12, new MapSize(5, 5), new MapSize(12, 12))
    myLayers(4) = new FunctionalLayer(new SigmoidFunction())
    myLayers(5) = new MeanPoolingLayer(new MapSize(2, 2), new MapSize(8, 8))
    myLayers(6) = new ConvolutionalLayer(12, 12, new MapSize(4, 4), new MapSize(4, 4))
    myLayers(7) = new FunctionalLayer(new SigmoidFunction())
    val topology = FeedForwardTopology(myLayers)

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val conf = new SparkConf().setMaster("local[8]").setAppName("ttt")
    val sc = new SparkContext(conf)
    val lines = sc.textFile("dataset/train.format", 8)
    val data = lines.map(line => line.split(",")).map(arr => arr.map(_.toDouble))
      .map(arr => {
      val target = new Array[Double](12)
      target(arr(784).toInt) = 1
      val in = Vector2BDM(Vectors.dense(arr.slice(0, 784)))
      (Vectors.fromBreeze(in.toDenseVector), Vectors.dense(target))
    }).cache()

    val feedForwardTrainer = new FeedForwardTrainer(topology, 784, 12)

    feedForwardTrainer.setStackSize(4) // CNN does not benefit from the stacked data
//    .LBFGSOptimizer.setNumIterations(20)
      .SGDOptimizer
      .setMiniBatchFraction(0.002)
      .setConvergenceTol(0)
      .setNumIterations(1000)
      .setUpdater(new CNNUpdater(0.85))

    for(iter <- 1 to 1000){
      val start = System.nanoTime()
      val mlpModel = feedForwardTrainer.train(data)
      feedForwardTrainer.setWeights(mlpModel.weights())

      println(s"Training time $iter: " + (System.nanoTime() - start) / 1e9)

      // predict
      val right = data.filter(v => mlpModel.predict(v._1).argmax == v._2.argmax).count()
      val precision = right.toDouble / data.count()
      println(s"right: $right, count: ${data.count()}, precision: $precision")
    }
  }

  def Vector2BDM(record: Vector): BDM[Double] = {
    val mapSize = new MapSize(28, 28)
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
    m
  }

}
