import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.neuralNetwork.{CNN, Scale, CNNLayer, CNNTopology}

object MnistCSVDriver {

  def main(args: Array[String]) {
    val topology = new CNNTopology
    topology.addLayer(CNNLayer.buildInputLayer(new Scale(28, 28)))
    topology.addLayer(CNNLayer.buildConvLayer(6, new Scale(5, 5)))
    topology.addLayer(CNNLayer.buildSampLayer(new Scale(2, 2)))
    topology.addLayer(CNNLayer.buildConvLayer(12, new Scale(5, 5)))
    topology.addLayer(CNNLayer.buildSampLayer(new Scale(2, 2)))
    topology.addLayer(CNNLayer.buildConvLayer(12, new Scale(4, 4)))
    val cnn: CNN = new CNN(topology).setMaxIterations(500000).setMiniBatchSize(16)

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val conf = new SparkConf().setMaster("local[8]").setAppName("ttt")
    val sc = new SparkContext(conf)
    val lines = sc.textFile("dataset/mnist/mnist_train.csv", 8)
    val data = lines.map(line => line.split(",")).map(arr => arr.map(_.toDouble))
      .map(arr => new LabeledPoint(arr(0), Vectors.dense(arr.slice(1, 785).map(v => if(v > 0) 1.0 else 0))))

    val start = System.nanoTime()
    cnn.trainOneByOne(data)
    println("Training time: " + (System.nanoTime() - start) / 1e9)
  }

}
