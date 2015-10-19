package hhbyyh.mCNN

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkContext, SparkConf}
import breeze.linalg.{DenseMatrix => BDM, kron}

/**
 * Created by yuhao on 9/22/15.
 */
object printMatrix {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val conf = new SparkConf().setMaster("local[8]").setAppName("ttt")
    val sc = new SparkContext(conf)
    val lines = sc.textFile("dataset/mnist/mnist_train.csv", 8)
    val data = lines.map(line => line.split(",")).map(arr => arr.map(_.toDouble))
      .map(arr => (arr(0), Example.Vector2Tensor(Vectors.dense(arr.slice(1, 785).map(v => if(v > 200) 1.0 else 0)))(0)))

    val lines2 = sc.textFile("dataset/train.format", 8)
    val data2 = lines2.map(line => line.split(",")).map(arr => arr.map(_.toDouble))
      .map(arr => (arr(784), Example.Vector2Tensor(Vectors.dense(arr.slice(0, 784)))(0)))

    data2.take(10).foreach(record =>{
      println("label: " + record._1)
      val intm = new BDM[Int](28, 28, record._2.toArray.map(d => d.toInt))
      val str = intm.toString(1000, 1000).replace('0', '.').replace('0', '*')
      println(str)
    })

  }
}
