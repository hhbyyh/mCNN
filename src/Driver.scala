import mycnn.{Size, Layer, CNN}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by yuhao on 7/19/15.
 */
object Driver {
  def main(args: Array[String]) {
    val builder: CNN.LayerBuilder = new CNN.LayerBuilder
    builder.addLayer(Layer.buildInputLayer(new Size(28, 28)))
    builder.addLayer(Layer.buildConvLayer(6, new Size(5, 5)))
    builder.addLayer(Layer.buildSampLayer(new Size(2, 2)))
    builder.addLayer(Layer.buildConvLayer(12, new Size(5, 5)))
    builder.addLayer(Layer.buildSampLayer(new Size(2, 2)))
    builder.addLayer(Layer.buildOutputLayer(10))
    val cnn: CNN = new CNN(builder, 50)

    //导入数据集
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val conf = new SparkConf().setMaster("local").setAppName("ttt")
    val sc = new SparkContext(conf)
    val lines = sc.textFile("dataset/train.format")
    val data = lines.map(line => line.split(",")).map(arr => arr.map(_.toDouble))
      .map(arr => new LabeledPoint(arr(784), Vectors.dense(arr.slice(0, 784))))

    val start = System.nanoTime()
    cnn.train(data, 5)
    println("Training time: " + (System.nanoTime() - start) / 1e9 )

    //预测
    // CNN cnn = CNN.loadModel(modelName);
    val testLines = sc.textFile("dataset/test.predict")
    val testdata = testLines.map(line => line.split(",")).map(arr => arr.map(_.toDouble))
      .map(arr => Vectors.dense(arr))
//    val result = cnn.predict(testdata)
//    println(result.collect().mkString("\n"))
  }

}
