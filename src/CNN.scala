/**
 * Created by yuhao on 7/18/15.
 */
package mycnn

import java.util.{ArrayList, List}

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics.sigmoid
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vector

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

}

class CNN extends Serializable {
  var LAMBDA: Double = 0
  var ALPHA: Double = 0.85
  private var layers: List[Layer] = null
  private var layerNum: Int = 0
  private var actualBatchSize: Int = 0

  def this(layerBuilder: CNN.LayerBuilder, batchSize: Int) {
    this()
    layers = layerBuilder.mLayers
    layerNum = layers.size
    this.actualBatchSize = batchSize
    setup(batchSize)
  }

  def train(trainset: RDD[LabeledPoint], repeat: Int) {
    var t = 0
    val trainSize = trainset.count().toInt
    while (t < repeat) {
      var epochsNum = trainSize / actualBatchSize
      if (trainSize % actualBatchSize != 0)
        epochsNum += 1
      var right = 0
      var count = 0
      var i = 0
      while (i < epochsNum) {
        Layer.prepareForNewBatch
        val batch = trainset.sample(true, actualBatchSize.toDouble * 2 / trainSize).take(actualBatchSize)
        batch.foreach(re => {
          val isRight = train(re)
          if (isRight)
            right += 1
          count += 1
          Layer.prepareForNewRecord
        })
        updateParas
        i += 1
      }
      val p = 1.0 * right / count
      if (t % 10 == 1 && p > 0.96) {
        ALPHA = 0.001 + ALPHA * 0.9
      }
      t += 1
      println("precision " + right + "/" + count + "=" + p)
    }
  }


  def predict(testset: RDD[Vector]): RDD[Int] = {
    Layer.prepareForNewBatch
    testset.map(record => {
      forward(record)
      val outputLayer = layers.get(layerNum - 1)
      val mapNum = outputLayer.getOutMapNum
      val out = new Array[Double](mapNum)
      for (m <- 0 until mapNum) {
        val outmap = outputLayer.getMap(m)
        out(m) = outmap(0, 0)
      }
      Util.getMaxIndex(out)
    })
  }

  private def train(record: LabeledPoint): Boolean = {
    forward(record.features)
    val result: Boolean = backPropagation(record)
    return result
  }

  private def backPropagation(record: LabeledPoint): Boolean = {
    val result: Boolean = setOutLayerErrors(record)
    setHiddenLayerErrors
    return result
  }


  private def updateParas {
    {
      var l: Int = 1
      while (l < layerNum) {
        {
          val layer: Layer = layers.get(l)
          val lastLayer: Layer = layers.get(l - 1)
          layer.getType match {
            case "conv" =>
              updateKernels(layer, lastLayer)
              updateBias(layer, lastLayer)
            case "output" =>
              updateKernels(layer, lastLayer)
              updateBias(layer, lastLayer)
            case _ =>
          }
        }
        l += 1
      }
    }
  }

  /**
   * 更新偏置
   *
   * @param layer
   * @param lastLayer
   */
  private def updateBias(layer: Layer, lastLayer: Layer) {
    val errors: Array[Array[BDM[Double]]] = layer.getErrors
    val mapNum: Int = layer.getOutMapNum

    var j: Int = 0
    while (j < mapNum) {
      val error: BDM[Double] = Util.sum(errors, j)
      val deltaBias: Double = sum(error) / actualBatchSize
      val bias: Double = layer.getBias(j) + ALPHA * deltaBias
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
  private def updateKernels(layer: Layer, lastLayer: Layer) {
    val mapNum: Int = layer.getOutMapNum
    val lastMapNum: Int = lastLayer.getOutMapNum
    var j = 0
    while (j < mapNum) {
      var i = 0
      while (i < lastMapNum) {
        // 对batch的每个记录delta求和
        var deltaKernel: BDM[Double] = null
        for (r <- 0 until actualBatchSize) {
          val error = layer.getError(r, j)
          if (deltaKernel == null)
            deltaKernel = Util.convnValid(lastLayer.getMap(r, i), error)
          else {
            // 累积求和
            val tsum = Util.convnValid(lastLayer.getMap(r, i), error)
            deltaKernel += tsum
          }
        }

        // 除以batchSize
        deltaKernel /= actualBatchSize.toDouble
        // 更新卷积核
        val kernel = layer.getKernel(i, j)
        deltaKernel = kernel * (1 - LAMBDA * ALPHA) + deltaKernel * ALPHA
        layer.setKernel(i, j, deltaKernel)
        i += 1
      }
      j += 1
    }
  }

  /**
   * 设置中将各层的残差
   */
  private def setHiddenLayerErrors {
    {
      var l: Int = layerNum - 2
      while (l > 0) {

        val layer: Layer = layers.get(l)
        val nextLayer: Layer = layers.get(l + 1)
        layer.getType match {
          case "samp" =>
            setSampErrors(layer, nextLayer)
          case "conv" =>
            setConvErrors(layer, nextLayer)
          case _ =>
        }
        l -= 1
      }
    }
  }

  /**
   * 设置采样层的残差
   *
   * @param layer
   * @param nextLayer
   */
  private def setSampErrors(layer: Layer, nextLayer: Layer) {
    val mapNum: Int = layer.getOutMapNum
    val nextMapNum: Int = nextLayer.getOutMapNum
    var i = 0
    while (i < mapNum) {
      var sum: BDM[Double] = null // 对每一个卷积进行求和
      var j = 0
      while (j < nextMapNum) {
        val nextError = nextLayer.getError(j)
        val kernel = nextLayer.getKernel(i, j)
        // 对卷积核进行180度旋转，然后进行full模式下得卷积
        if (sum == null)
          sum = Util.convnFull(nextError, flipud(fliplr(kernel)))
        else
          sum += Util.convnFull(nextError, flipud(fliplr(kernel)))
        j += 1
      }
      layer.setError(i, sum)
      i += 1
    }
  }

  /**
   * 设置卷积层的残差
   *
   * @param layer
   * @param nextLayer
   */
  private def setConvErrors(layer: Layer, nextLayer: Layer) {
    val mapNum: Int = layer.getOutMapNum

    var m: Int = 0
    while (m < mapNum) {

      val scale: Size = nextLayer.getScaleSize
      val nextError: BDM[Double] = nextLayer.getError(m)
      val map: BDM[Double] = layer.getMap(m)

      var outMatrix: BDM[Double] = (1.0 - map)
      outMatrix = map :* outMatrix
      outMatrix = outMatrix :* Util.kronecker(nextError, scale)
      layer.setError(m, outMatrix)
      m += 1
    }

  }

  /**
   * 设置输出层的残差值,输出层的神经单元个数较少，暂不考虑多线程
   *
   * @param record
   * @return
   */
  private def setOutLayerErrors(record: LabeledPoint): Boolean = {
    val outputLayer: Layer = layers.get(layerNum - 1)
    val mapNum: Int = outputLayer.getOutMapNum
    val target: Array[Double] = new Array[Double](mapNum)
    val outmaps: Array[Double] = new Array[Double](mapNum)
    var m = 0
    while (m < mapNum) {
      val outmap = outputLayer.getMap(m)
      outmaps(m) = outmap(0, 0)
      m += 1
    }
    val label = record.label.toInt
    target(label) = 1
    m = 0
    while (m < mapNum) {
      outputLayer.setError(m, 0, 0, outmaps(m) * (1 - outmaps(m)) * (target(m) - outmaps(m)))
      m += 1
    }
    val outClass = Util.getMaxIndex(outmaps)
    return label == outClass
  }


  /**
   * 前向计算一条记录
   *
   * @param record
   */
  private def forward(record: Vector) {
    setInLayerOutput(record)

    var l: Int = 1
    while (l < layers.size) {

      val layer: Layer = layers.get(l)
      val lastLayer: Layer = layers.get(l - 1)
      layer.getType match {
        case "conv" =>
          setConvOutput(layer, lastLayer)
        case "samp" =>
          setSampOutput(layer, lastLayer)
        case "output" =>
          setConvOutput(layer, lastLayer)
        case _ =>
      }

      l += 1
    }

  }

  /**
   * 根据记录值，设置输入层的输出值
   *
   * @param record
   */
  private def setInLayerOutput(record: Vector) {
    val inputLayer: Layer = layers.get(0)
    val mapSize: Size = inputLayer.getMapSize
    val attr = record
    if (attr.size != mapSize.x * mapSize.y)
      throw new RuntimeException("数据记录的大小与定义的map大小不一致!")

    var i: Int = 0
    while (i < mapSize.x) {
      var j: Int = 0
      while (j < mapSize.y) {
        inputLayer.setMapValue(0, i, j, attr(mapSize.x * i + j))
        j += 1
      }
      i += 1
    }

  }

  private def setConvOutput(layer: Layer, lastLayer: Layer) {
    val mapNum: Int = layer.getOutMapNum
    val lastMapNum: Int = lastLayer.getOutMapNum
    var j = 0
    while (j < mapNum) {
      var sum: BDM[Double] = null // 对每一个输入map的卷积进行求和
      var i = 0
      while (i < lastMapNum) {
        val lastMap = lastLayer.getMap(i)
        val kernel = layer.getKernel(i, j)
        if (sum == null)
          sum = Util.convnValid(lastMap, kernel)
        else
          sum += Util.convnValid(lastMap, kernel)
        i += 1
      }
      val bias = layer.getBias(j)
      sum = sigmoid(sum + bias)
      layer.setMapValue(j, sum)
      j += 1
    }
  }

  /**
   * 设置采样层的输出值，采样层是对卷积层的均值处理
   *
   * @param layer
   * @param lastLayer
   */
  private def setSampOutput(layer: Layer, lastLayer: Layer) {
    val lastMapNum: Int = lastLayer.getOutMapNum

    var i: Int = 0
    while (i < lastMapNum) {
      val lastMap: BDM[Double] = lastLayer.getMap(i)
      val scaleSize: Size = layer.getScaleSize
      val sampMatrix: BDM[Double] = Util.scaleMatrix(lastMap, scaleSize)
      layer.setMapValue(i, sampMatrix)
      i += 1
    }
  }

  /**
   * 设置cnn网络的每一层的参数
   *
   */
  def setup(batchSize: Int) {
    val inputLayer: Layer = layers.get(0)
    inputLayer.initOutmaps(batchSize)

    var i: Int = 1
    while (i < layers.size) {
      val layer: Layer = layers.get(i)
      val frontLayer: Layer = layers.get(i - 1)
      val frontMapNum: Int = frontLayer.getOutMapNum
      layer.getType match {
        case "input" =>
        case "conv" =>
          layer.setMapSize(frontLayer.getMapSize.subtract(layer.getKernelSize, 1))
          layer.initKernel(frontMapNum)
          layer.initBias(frontMapNum)
          layer.initErros(batchSize)
          layer.initOutmaps(batchSize)
        case "samp" =>
          layer.setOutMapNum(frontMapNum)
          layer.setMapSize(frontLayer.getMapSize.divide(layer.getScaleSize))
          layer.initErros(batchSize)
          layer.initOutmaps(batchSize)
        case "output" =>
          layer.initOutputKerkel(frontMapNum, frontLayer.getMapSize)
          layer.initBias(frontMapNum)
          layer.initErros(batchSize)
          layer.initOutmaps(batchSize)
      }
      i += 1
    }
  }
}