package com.intel.webscaleml.algorithms.neuralNetwork

import java.util.{ArrayList, List}

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics.sigmoid
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * Builder class to define CNN layers structure
 * The second layer from the last can not be a conv layer,
 * Typical layers: input, conv, samp, conv, samp, output
 */
class CNNTopology {
  var mLayers: List[Layer] = new ArrayList[Layer]

  def this(layer: Layer) {
    this()
    mLayers.add(layer)
  }

  def addLayer(layer: Layer): CNNTopology = {
    mLayers.add(layer)
    this
  }
}

object CNN {
  private def combineGradient(
    g1: Array[(Array[Array[BDM[Double]]], Array[Double])],
    g2: Array[(Array[Array[BDM[Double]]], Array[Double])]):
  Array[(Array[Array[BDM[Double]]], Array[Double])] = {

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

/**
 * Convolution neural network
 */
class CNN private extends Serializable {
  var ALPHA: Double = 0.85
  private var layers: List[Layer] = null
  private var layerNum: Int = 0
  private var maxIterations = 10
  private var batchSize = 100

  def this(layerBuilder: CNNTopology) {
    this()
    layers = layerBuilder.mLayers
    layerNum = layers.size
    setup
  }

  def setMiniBatchSize(batchSize: Int): this.type = {
    this.batchSize = batchSize
    this
  }

  /**
   * Maximum number of iterations for learning.
   */
  def getMaxIterations: Int = maxIterations

  /**
   * Maximum number of iterations for learning.
   * (default = 20)
   */
  def setMaxIterations(maxIterations: Int): this.type = {
    this.maxIterations = maxIterations
    this
  }

  def trainOnebyOne(trainset: RDD[LabeledPoint]) {
    var t = 0
    val trainSize = trainset.count().toInt
    val dataArr = trainset.collect()
    while (t < maxIterations) {
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

  def train(trainset: RDD[LabeledPoint]) {
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
    var totalCount = 0
    var totalRight = 0
    while (t < maxIterations) {
      val (gradientSum, right, count) = trainset.sample(false, batchSize.toDouble/trainSize, 42 + t)
        .treeAggregate((gZero, 0, 0))(
          seqOp = (c, v) => {
            val result = train(v)
            val gradient = result._2
            val right = if(result._1) 1 else 0
            (CNN.combineGradient(c._1, gradient), c._2 + right, c._3 + 1)
          },
          combOp = (c1, c2) => {
            (CNN.combineGradient(c1._1, c2._1), c1._2 + c2._2, c1._3 + c2._3)
          })

      t += 1
      if(count > 0){
        updateParams(gradientSum, count)
        val p = 1.0 * totalRight / totalCount
        if (t % 10 == 1 && p > 0.96) {
          ALPHA = 0.001 + ALPHA * 0.9
        }
        totalCount += count
        totalRight += right
        if(totalCount > 10000){
          println(t + "\tprecision " + totalRight + "/" + totalCount + "=" + p)
          totalCount = 0
          totalRight = 0
        }
      }
    }
  }

  def predict(testSet: RDD[Vector]): RDD[Int] = {
    testSet.map(record => {
      val outputs: Array[Array[BDM[Double]]] = forward(record)
      val outputLayer = layers.get(layerNum - 1)
      val mapNum = outputLayer.outMapNum
      val out = new Array[Double](mapNum)
      for (m <- 0 until mapNum) {
        val outMap = outputs(layerNum - 1)(m)
        out(m) = outMap(0, 0)
      }
      Util.getMaxIndex(out)
    })
  }

  private def train(
      record: LabeledPoint): (Boolean, Array[(Array[Array[BDM[Double]]], Array[Double])]) = {
    val outputs: Array[Array[BDM[Double]]] = forward(record.features)
    val (right, errors) = backPropagation(record, outputs)
    val gradient: Array[(Array[Array[BDM[Double]]], Array[Double])] = getGradient(outputs, errors)
    (right, gradient)
  }

  private def backPropagation(
      record: LabeledPoint,
      outputs: Array[Array[BDM[Double]]]): (Boolean, Array[Array[BDM[Double]]]) = {
    val errors = new Array[Array[BDM[Double]]](layers.size)
    val result = setOutLayerErrors(record, outputs(layerNum - 1))
    errors(layerNum - 1) = result._2
    var l: Int = layerNum - 2
    while (l > 0) {
      val layer: Layer = layers.get(l)
      val nextLayer: Layer = layers.get(l + 1)
      errors(l) = layer.getType match {
        case "samp" =>
          setSampErrors(layer, nextLayer.asInstanceOf[ConvLayer], errors(l + 1))
        case "conv" =>
          setConvErrors(layer, nextLayer.asInstanceOf[SampLayer], errors(l + 1), outputs(l))
        case _ => null
      }
      l -= 1
    }
    (result._1, errors)
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
          val kernelGradient = getKernelsGradient(layer, lastLayer, errors(l), outputs(l - 1))
          val biasGradient = getBiasGradient(layer, errors(l))
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

  private def updateParams(
      gradient: Array[(Array[Array[BDM[Double]]],
      Array[Double])], 
      batchSize: Int): Unit = {
    var l: Int = 1
    while (l < layerNum) {
      val layer: Layer = layers.get(l)
      layer.getType match {
        case "conv" =>
          updateKernels(layer.asInstanceOf[ConvLayer], gradient(l)._1, batchSize)
          updateBias(layer.asInstanceOf[ConvLayer], gradient(l)._2, batchSize)
        case "output" =>
          updateKernels(layer.asInstanceOf[ConvLayer], gradient(l)._1, batchSize)
          updateBias(layer.asInstanceOf[ConvLayer], gradient(l)._2, batchSize)
        case _ =>
      }
      l += 1
    }
  }

  /**
   * get bias gradient
   *
   * @param layer layer to be updated
   * @param errors errors of this layer
   */
  private def getBiasGradient(layer: Layer, errors: Array[BDM[Double]]): Array[Double] = {
    val mapNum: Int = layer.outMapNum
    var j: Int = 0
    val gradient = new Array[Double](mapNum)
    while (j < mapNum) {
      val error: BDM[Double] = errors(j) //Util.sum(errors, j)
      val deltaBias: Double = sum(error)
      gradient(j) = deltaBias
      j += 1
    }
    gradient
  }

  private def updateBias(layer: ConvLayer, gradient: Array[Double], batchSize: Int): Unit = {
    val gv = new BDV[Double](gradient)
    layer.getBias += gv * ALPHA / batchSize.toDouble
  }

  /**
   * get kernels gradient
   *
   * @param layer
   * @param lastLayer
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
        val error = layerError(j)
        val deltaKernel = Util.convnValid(lastOutput(i), error)
        delta(i)(j) = deltaKernel
        i += 1
      }
      j += 1
    }
    delta
  }

  private def updateKernels(layer: ConvLayer, gradient: Array[Array[BDM[Double]]], batchSize: Int): Unit = {
    val len = gradient.length
    val width = gradient(0).length
    var j = 0
    while (j < width) {
      var i = 0
      while (i < len) {
        // update kernel
        val deltaKernel = gradient(i)(j) / batchSize.toDouble * ALPHA
        layer.getKernel(i, j) += deltaKernel
        i += 1
      }
      j += 1
    }
  }

  /**
   * set errors for sampling layer
   *
   * @param layer
   * @param nextLayer
   */
  private def setSampErrors(layer: Layer, nextLayer: ConvLayer, nextLayerError: Array[BDM[Double]]): Array[BDM[Double]] = {
    val mapNum: Int = layer.outMapNum
    val nextMapNum: Int = nextLayer.outMapNum
    val errors = new Array[BDM[Double]](mapNum)
    var i = 0
    while (i < mapNum) {
      var sum: BDM[Double] = null // sum for every kernel
      var j = 0
      while (j < nextMapNum) {
        val nextError = nextLayerError(j)
        val kernel = nextLayer.getKernel(i, j)
        // rotate kernel by 180 degrees and get full convolution
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
   * set errors for convolution layer
   *
   * @param layer
   * @param nextLayer
   */
  private def setConvErrors(
      layer: Layer,
      nextLayer: SampLayer,
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
   * set errors for output layer
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
    (label == outClass, layerError)
  }


  /**
   * forward for one record
   *
   * @param record
   */
  private def forward(record: Vector): Array[Array[BDM[Double]]] = {
    val outputs = new Array[Array[BDM[Double]]](layers.size)
    outputs(0) = setInLayerOutput(record)
    var l: Int = 1
    while (l < layers.size) {
      val layer: Layer = layers.get(l)
      outputs(l) =
        layer.getType match {
          case "conv" =>
            setConvOutput(layer.asInstanceOf[ConvLayer], outputs(l - 1))
          case "samp" =>
            setSampOutput(layer.asInstanceOf[SampLayer], outputs(l - 1))
          case "output" =>
            setConvOutput(layer.asInstanceOf[ConvLayer], outputs(l - 1))
          case _ => null
        }
      l += 1
    }
    outputs
  }

  /**
   * set inlayer output
   * @param record
   */
  private def setInLayerOutput(record: Vector): Array[BDM[Double]] = {
    val inputLayer: Layer = layers.get(0)
    val mapSize: Size = inputLayer.getMapSize
    val attr = record
    if (attr.size != mapSize.x * mapSize.y)
      throw new RuntimeException("data size and map size mismatch!")

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
    Array(m)
  }

  private def setConvOutput(layer: ConvLayer, outputs: Array[BDM[Double]]): Array[BDM[Double]] = {
    val mapNum: Int = layer.outMapNum
    val lastMapNum: Int = outputs.length
    val output = new Array[BDM[Double]](mapNum)
    var j = 0
    val oldBias = layer.getBias
    while (j < mapNum) {
      var sum: BDM[Double] = null
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
      sum = sigmoid(sum + oldBias(j))
      output(j) = sum
      j += 1
    }
    output
  }

  /**
   * set output for sampling layer
   *
   * @param layer
   */
  private def setSampOutput(layer: SampLayer, outputs: Array[BDM[Double]]): Array[BDM[Double]] = {
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

  def setup {
    var i: Int = 1
    while (i < layers.size) {
      val layer: Layer = layers.get(i)
      val frontLayer: Layer = layers.get(i - 1)
      val frontMapNum: Int = frontLayer.outMapNum
      layer.getType match {
        case "input" =>
        case "conv" =>
          val convLayer = layer.asInstanceOf[ConvLayer]
          convLayer.setMapSize(frontLayer.getMapSize.subtract(convLayer.getKernelSize, 1))
          convLayer.initKernel(frontMapNum)
          convLayer.initBias(frontMapNum)
        case "samp" =>
          val sampLayer = layer.asInstanceOf[SampLayer]
          sampLayer.outMapNum = frontMapNum
          sampLayer.setMapSize(frontLayer.getMapSize.divide(sampLayer.getScaleSize))
        case "output" =>
          val outputLayer = layer.asInstanceOf[OutputLayer]
          outputLayer.initOutputKernels(frontMapNum, frontLayer.getMapSize)
          outputLayer.asInstanceOf[OutputLayer].initBias(frontMapNum)
      }
      i += 1
    }
  }
}
