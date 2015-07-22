/**
 * Created by yuhao on 7/18/15.
 */
package mycnn

import java.util.{ArrayList, List}

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics.sigmoid
import org.apache.spark.mllib.optimization.Gradient
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vectors, Vector}

object CNN {

  /**
   * The second layer from the last can not be a conv layer
   */
  class LayerBuilder {
    var mLayers: List[Layer] = new ArrayList[Layer]

    def this(layer: Layer) {
      this()
      mLayers.add(layer)
    }

    def addLayer(layer: Layer): CNN.LayerBuilder = {
      mLayers.add(layer)
      return this
    }
  }

  def combineGradient(
    g1: Array[(Array[Array[BDM[Double]]], Array[Double])],
    g2: Array[(Array[Array[BDM[Double]]], Array[Double])]) :  Array[(Array[Array[BDM[Double]]], Array[Double])] = {

    val l = g1.length
    var li = 0
    while(li < l){
      if(g1(li) != null){
        // kernel
        val layer = g1(li)._1
        val x = layer.length
        var xi = 0
        while(xi < x){
          val line: Array[BDM[Double]] = layer(xi)
          val y = line.length
          var yi = 0
          while(yi < y){
            line(yi) += g2(li)._1(xi)(yi)
            yi += 1
          }
          xi += 1
        }

        // bias
        val b = g1(li)._2
        val len = b.length
        var bi = 0
        while(bi < len){
          b(bi) = b(bi) + g2(li)._2(bi)
          bi += 1
        }
      }
      li += 1
    }
    g1
  }
}

class CNN extends Serializable {
  var LAMBDA: Double = 0
  var ALPHA: Double = 0.85
  private var layers: List[Layer] = null
  private var layerNum: Int = 0

  def this(layerBuilder: CNN.LayerBuilder, batchSize: Int) {
    this()
    layers = layerBuilder.mLayers
    layerNum = layers.size
    setup
  }

  def train(trainset: RDD[LabeledPoint], repeat: Int) {
    var t = 0
    val trainSize = trainset.count().toInt
    val dataArr = trainset.collect()
    while (t < repeat) {
      val epochsNum = trainSize
      var right = 0
      var count = 0
      var i = 0
      while (i < epochsNum) {
        val start = System.nanoTime()
        val record = dataArr(i)
        val result = train(record)
        if (result._1) right += 1
        count += 1
        val gradient: Array[(Array[Array[BDM[Double]]], Array[Double])] = result._2
        updateParams(gradient, 1)
        i += 1
//        println("epochsNum: " + i + "\t" + (System.nanoTime() - start) / 1e9 )
      }
      val p = 1.0 * right / count
      if (t % 10 == 1 && p > 0.96) {
        ALPHA = 0.001 + ALPHA * 0.9
      }
      t += 1
      println("precision " + right + "/" + count + "=" + p)
    }
  }

  def train1(trainset: RDD[LabeledPoint], repeat: Int) {
    var t = 0
    val trainSize = trainset.count().toInt
    val gZero = train(trainset.first())._2
    gZero.foreach(tu =>{
      if(tu != null){
        tu._1.foreach(m => m.foreach(x => x -= x))
        val len = tu._2.length
        for(i <- 0 until len){
          tu._2(i) = 0
        }
      }
    })
    while (t < repeat) {
      val (gradientSum, right, count) = trainset.sample(false, 50.0/trainSize, 42 + t)
        .treeAggregate((gZero, 0, 0))(
          seqOp = (c, v) => {
            val result = train(v)
            val gradient = result._2
            val right = if(result._1) 1 else 0
            (CNN.combineGradient(c._1, gradient), c._2 + right, c._3 + 1)
          },
          combOp = (c1, c2) => {
            // c: (grad, loss, count)
            (CNN.combineGradient(c1._1, c2._1), c1._2 + c2._2, c1._3 + c2._3)
          })

      updateParams(gradientSum, count)
      val p = 1.0 * right / count
      if (t % 10 == 1 && p > 0.96) {
        ALPHA = 0.001 + ALPHA * 0.9
      }
      t += 1
      println(t + "\tprecision " + right + "/" + count + "=" + p)
    }
  }


//  def predict(testset: RDD[Vector]): RDD[Int] = {
//    testset.map(record => {
//      forward(record)
//      val outputLayer = layers.get(layerNum - 1)
//      val mapNum = outputLayer.outMapNum
//      val out = new Array[Double](mapNum)
//      for (m <- 0 until mapNum) {
//        val outmap = outputLayer.getMap(m)
//        out(m) = outmap(0, 0)
//      }
//      Util.getMaxIndex(out)
//    })
//  }

  private def train(record: LabeledPoint): (Boolean, Array[(Array[Array[BDM[Double]]], Array[Double])]) = {
    val outputs: Array[Array[BDM[Double]]] = forward(record.features)
    val result = backPropagation(record, outputs)
    val right = result._1
    val errors: Array[Array[BDM[Double]]] = result._2
    val gradient: Array[(Array[Array[BDM[Double]]], Array[Double])] = getGradient(outputs, errors)

    return (right, gradient)
  }

  private def backPropagation(record: LabeledPoint, outputs: Array[Array[BDM[Double]]]): (Boolean, Array[Array[BDM[Double]]]) = {
    val errors = new Array[Array[BDM[Double]]](layers.size)
    val result = setOutLayerErrors(record, outputs(layerNum - 1))
    errors(layerNum - 1) = result._2
    var l: Int = layerNum - 2
    while (l > 0) {
      val layer: Layer = layers.get(l)
      val nextLayer: Layer = layers.get(l + 1)
      errors(l) = layer.getType match {
        case "samp" =>
          setSampErrors(layer, nextLayer, errors(l + 1))
        case "conv" =>
          setConvErrors(layer, nextLayer, errors(l + 1), outputs(l))
        case _ => null
      }
      l -= 1
    }
    return (result._1, errors)
  }


  private def getGradient(
      outputs: Array[Array[BDM[Double]]],
      errors: Array[Array[BDM[Double]]]) : Array[(Array[Array[BDM[Double]]], Array[Double])] = {
    var l: Int = 1
    val gradient = new Array[(Array[Array[BDM[Double]]], Array[Double])](layerNum)
    while (l < layerNum) {
      val layer: Layer = layers.get(l)
      val lastLayer: Layer = layers.get(l - 1)
      gradient(l) = layer.getType match {
        case "conv" =>
          val kernelGradient: Array[Array[BDM[Double]]] = getKernelsGradient(layer, lastLayer, errors(l), outputs(l - 1))
          val biasGradient: Array[Double] = getBiasGradient(layer, errors(l))
          (kernelGradient, biasGradient)
        case "output" =>
          val kernelGradient = getKernelsGradient(layer, lastLayer, errors(l), outputs(l - 1))
          val biasGradient = getBiasGradient(layer, errors(l))
          (kernelGradient, biasGradient)
        case _ => null
      }
      l += 1
    }
    gradient
  }

  private def updateParams(gradient: Array[(Array[Array[BDM[Double]]], Array[Double])], batchSize: Int): Unit = {
    var l: Int = 1
    while (l < layerNum) {
      val layer: Layer = layers.get(l)
      val lastLayer: Layer = layers.get(l - 1)
      layer.getType match {
        case "conv" =>
          updateKernels(layer, gradient(l)._1, batchSize)
          udpateBias(layer, gradient(l)._2, batchSize)
        case "output" =>
          updateKernels(layer, gradient(l)._1, batchSize)
          udpateBias(layer, gradient(l)._2, batchSize)
        case _ =>
      }
      l += 1
    }
  }

  /**
   * 更新偏置
   *
   * @param layer
   */
  private def getBiasGradient(layer: Layer, errors: Array[BDM[Double]]): Array[Double] = {
    val mapNum: Int = layer.outMapNum
    var j: Int = 0
    val gradient = new Array[Double](mapNum)
    while (j < mapNum) {
      val error: BDM[Double] = Util.sum(errors, j)
      val deltaBias: Double = sum(error)
      gradient(j) = deltaBias
      j += 1
    }
    gradient
  }

  private def udpateBias(layer: Layer, gradient: Array[Double], batchSize: Int): Unit = {
    val len = gradient.length
    var j: Int = 0
    while (j < len) {
      val bias: Double = layer.getBias(j) + ALPHA * gradient(j) / batchSize
      layer.setBias(j, bias)
      j += 1
    }
  }

  /**
   * 更新layer层的卷积核（权重）和偏置
   *
   * @param layer
	 * 当前层
   * @param lastLayer
	 * 前一层
   */
  private def getKernelsGradient(
      layer: Layer,
      lastLayer: Layer,
      layerError: Array[BDM[Double]],
      lastOutput: Array[BDM[Double]]): Array[Array[BDM[Double]]] = {
    val mapNum: Int = layer.outMapNum
    val lastMapNum: Int = lastLayer.outMapNum
    val delta = Array.ofDim[BDM[Double]](lastMapNum, mapNum)
    var j = 0
    while (j < mapNum) {
      var i = 0
      while (i < lastMapNum) {
        // 对batch的每个记录delta求和
        var deltaKernel: BDM[Double] = null
        val error = layerError(j)
        deltaKernel = Util.convnValid(lastOutput(i), error)
        delta(i)(j) = deltaKernel
        i += 1
      }
      j += 1
    }
    delta
  }

  private def updateKernels(layer: Layer, gradient: Array[Array[BDM[Double]]], batchSize: Int): Unit = {
    val len = gradient.length
    val width = gradient(0).length
    var j = 0
    while (j < width) {
      var i = 0
      while (i < len) {
        // 更新卷积核
        val deltaKernel = gradient(i)(j) / batchSize.toDouble * ALPHA
        layer.setKernel(i, j, layer.getKernel(i, j) + deltaKernel)
        i += 1
      }
      j += 1
    }
  }

  /**
   * 设置采样层的残差
   *
   * @param layer
   * @param nextLayer
   */
  private def setSampErrors(layer: Layer, nextLayer: Layer, nextLayerError: Array[BDM[Double]]): Array[BDM[Double]] = {
    val mapNum: Int = layer.outMapNum
    val nextMapNum: Int = nextLayer.outMapNum
    val errors = new Array[BDM[Double]](mapNum)
    var i = 0
    while (i < mapNum) {
      var sum: BDM[Double] = null // 对每一个卷积进行求和
      var j = 0
      while (j < nextMapNum) {
        val nextError = nextLayerError(j)
        val kernel = nextLayer.getKernel(i, j)
        // 对卷积核进行180度旋转，然后进行full模式下得卷积
        if (sum == null)
          sum = Util.convnFull(nextError, flipud(fliplr(kernel)))
        else
          sum += Util.convnFull(nextError, flipud(fliplr(kernel)))
        j += 1
      }
      errors(i) = sum
      i += 1
    }
    errors
  }

  /**
   * 设置卷积层的残差
   *
   * @param layer
   * @param nextLayer
   */
  private def setConvErrors(
      layer: Layer,
      nextLayer: Layer,
      nextLayerError: Array[BDM[Double]],
      layerOutput: Array[BDM[Double]]): Array[BDM[Double]] = {
    val mapNum: Int = layer.outMapNum
    val errors = new Array[BDM[Double]](mapNum)
    var m: Int = 0
    while (m < mapNum) {
      val scale: Size = nextLayer.getScaleSize
      val nextError: BDM[Double] = nextLayerError(m)
      val map: BDM[Double] = layerOutput(m)
      var outMatrix: BDM[Double] = (1.0 - map)
      outMatrix = map :* outMatrix
      outMatrix = outMatrix :* Util.kronecker(nextError, scale)
      errors(m) = outMatrix
      m += 1
    }
    errors
  }

  /**
   * 设置输出层的残差值,输出层的神经单元个数较少，暂不考虑多线程
   *
   * @param record
   * @return
   */
  private def setOutLayerErrors(record: LabeledPoint, output: Array[BDM[Double]]): (Boolean, Array[BDM[Double]]) = {
    val outputLayer: Layer = layers.get(layerNum - 1)
    val mapNum: Int = outputLayer.outMapNum
    val layerError = new Array[BDM[Double]](mapNum)
    val target: Array[Double] = new Array[Double](mapNum)
    val outValues: Array[Double] = new Array[Double](mapNum)
    var m = 0
    while (m < mapNum) {
      val outmap = output(m)
      outValues(m) = outmap(0, 0)
      m += 1
    }
    val label = record.label.toInt
    target(label) = 1
    m = 0
    while (m < mapNum) {
      val errorMatrix = new BDM[Double](1, 1)
      errorMatrix(0, 0) = outValues(m) * (1 - outValues(m)) * (target(m) - outValues(m))
      layerError(m) = errorMatrix
      m += 1
    }
    val outClass = Util.getMaxIndex(outValues)
    return (label == outClass, layerError)
  }


  /**
   * 前向计算一条记录
   *
   * @param record
   */
  private def forward(record: Vector): Array[Array[BDM[Double]]] = {
    val outputs = new Array[Array[BDM[Double]]](layers.size)
    outputs(0) = setInLayerOutput(record)
    var l: Int = 1
    while (l < layers.size) {
      val layer: Layer = layers.get(l)
      val lastLayer: Layer = layers.get(l - 1)
      outputs(l) =
        layer.getType match {
          case "conv" =>
            setConvOutput(layer, outputs(l - 1))
          case "samp" =>
            setSampOutput(layer, outputs(l - 1))
          case "output" =>
            setConvOutput(layer, outputs(l - 1))
          case _ => null
        }
      l += 1
    }
    outputs
  }

  /**
   * 根据记录值，设置输入层的输出值
   *
   * @param record
   */
  private def setInLayerOutput(record: Vector): Array[BDM[Double]] = {
    val inputLayer: Layer = layers.get(0)
    val mapSize: Size = inputLayer.getMapSize
    val attr = record
    if (attr.size != mapSize.x * mapSize.y)
      throw new RuntimeException("数据记录的大小与定义的map大小不一致!")

    val m = new BDM[Double](mapSize.x, mapSize.y)
    var i: Int = 0
    while (i < mapSize.x) {
      var j: Int = 0
      while (j < mapSize.y) {
        m(i, j) = attr(mapSize.x * i + j)
        j += 1
      }
      i += 1
    }
    return Array(m)
  }

  private def setConvOutput(layer: Layer, outputs: Array[BDM[Double]]): Array[BDM[Double]] = {
    val mapNum: Int = layer.outMapNum
    val lastMapNum: Int = outputs.length
    val output = new Array[BDM[Double]](mapNum)
    var j = 0
    while (j < mapNum) {
      var sum: BDM[Double] = null // 对每一个输入map的卷积进行求和
      var i = 0
      while (i < lastMapNum) {
        val lastMap = outputs(i)
        val kernel = layer.getKernel(i, j)
        if (sum == null)
          sum = Util.convnValid(lastMap, kernel)
        else
          sum += Util.convnValid(lastMap, kernel)
        i += 1
      }
      val bias = layer.getBias(j)
      sum = sigmoid(sum + bias)
      output(j) = sum
      j += 1
    }
    return output
  }

  /**
   * 设置采样层的输出值，采样层是对卷积层的均值处理
   *
   * @param layer
   */
  private def setSampOutput(layer: Layer, outputs: Array[BDM[Double]]): Array[BDM[Double]] = {
    val lastMapNum: Int = outputs.length
    val output = new Array[BDM[Double]](lastMapNum)
    var i: Int = 0
    while (i < lastMapNum) {
      val lastMap: BDM[Double] = outputs(i)
      val scaleSize: Size = layer.getScaleSize
      val sampMatrix: BDM[Double] = Util.scaleMatrix(lastMap, scaleSize)
      output(i) = sampMatrix
      i += 1
    }
    output
  }

  /**
   * 设置cnn网络的每一层的参数
   *
   */
  def setup {
    val inputLayer: Layer = layers.get(0)

    var i: Int = 1
    while (i < layers.size) {
      val layer: Layer = layers.get(i)
      val frontLayer: Layer = layers.get(i - 1)
      val frontMapNum: Int = frontLayer.outMapNum
      layer.getType match {
        case "input" =>
        case "conv" =>
          layer.setMapSize(frontLayer.getMapSize.subtract(layer.getKernelSize, 1))
          layer.initKernel(frontMapNum)
          layer.initBias(frontMapNum)
        case "samp" =>
          layer.outMapNum = frontMapNum
          layer.setMapSize(frontLayer.getMapSize.divide(layer.getScaleSize))
        case "output" =>
          layer.initOutputKerkel(frontMapNum, frontLayer.getMapSize)
          layer.initBias(frontMapNum)
      }
      i += 1
    }
  }
}
