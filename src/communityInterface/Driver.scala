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
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.apache.spark.{SparkContext, SparkConf}

object CNNDriver {

  def main(args: Array[String]) {

    val myLayers = new Array[Layer](5)
    myLayers(0) = new ConvolutionLayer(1, 6, new Scale(5, 5), new Scale(28, 28))
    myLayers(1) = new MeanPoolingLayer(new Scale(2, 2), new Scale(24, 24))
    myLayers(2) = new ConvolutionLayer(6, 12, new Scale(5, 5), new Scale(12, 12))
    myLayers(3) = new MeanPoolingLayer(new Scale(2, 2), new Scale(8, 8))
    myLayers(4) = new ConvolutionLayer(12, 12, new Scale(4, 4), new Scale(4, 4))
    val topology = CNNTopology(myLayers)

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val conf = new SparkConf().setMaster("local[8]").setAppName("ttt")
    val sc = new SparkContext(conf)
    val lines = sc.textFile("dataset/train.format", 8)
    val data = lines.map(line => line.split(",")).map(arr => arr.map(_.toDouble))
      .map(arr => {
      val target = new Array[Double](12)
      target(arr(784).toInt) = 1
      (Vectors.dense(arr.slice(0, 784)), Vectors.dense(target))
    })

    val start = System.nanoTime()
    val FeedForwardTrainer = new FeedForwardTrainer(topology, 784, 12)
    FeedForwardTrainer.SGDOptimizer.setMiniBatchFraction(0.001).setConvergenceTol(1e-3).setNumIterations(1000000)
    FeedForwardTrainer.setStackSize(1)
    val mlpModel = FeedForwardTrainer.train(data)

    // predict
    val right = data.map(v => {
      val pre = mlpModel.predict(v._1)
      pre.argmax == v._2.argmax
    }).filter(b => b).count()

    val precision = right.toDouble / data.count()

    println(precision)

    println("Training time: " + (System.nanoTime() - start) / 1e9)
  }

}
