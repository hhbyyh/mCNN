package torch.cnn

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by yuhaoyan on 8/25/2015.
 */
object TorchDriver {

  def main(args: Array[String]) {

    val mlp = new Sequential
    mlp.add(new ConvLayer(6, new Scale(5, 5)))
    mlp.add(new SampLayer(new Scale(2, 2)))
    mlp.add(new ConvLayer(12, new Scale(5, 5)))
    mlp.add(new SampLayer(new Scale(2, 2)))
    mlp.add(new ConvLayer(12, new Scale(4, 4)))

    val trainer = new StochasticGradient(mlp, new Criterion)
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val conf = new SparkConf().setMaster("local[8]").setAppName("ttt")
    val sc = new SparkContext(conf)
    val lines = sc.textFile("dataset/train.format", 8)
    val data = lines.map(line => line.split(",")).map(arr => arr.map(_.toDouble))
      .map(arr => new LabeledPoint(arr(784), Vectors.dense(arr.slice(0, 784))))

    val start = System.nanoTime()
    trainer.train(data)
    println("Training time: " + (System.nanoTime() - start) / 1e9)
  }
}
