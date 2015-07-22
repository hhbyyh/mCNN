package com.intel.webscaleml.algorithms.neuralNetwork

import java.io.Serializable
import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}

object Layer {

  def buildInputLayer(mapSize: Size): Layer = {
    val layer: Layer = new InputLayer
    layer.outMapNum = 1
    layer.setMapSize(mapSize)
    layer
  }

  def buildConvLayer(outMapNum: Int, kernelSize: Size): Layer = {
    val layer = new ConvLayer
    layer.outMapNum = outMapNum
    layer.setKernelSize(kernelSize)
    layer
  }

  def buildSampLayer(scaleSize: Size): Layer = {
    val layer = new SampLayer
    layer.setScaleSize(scaleSize)
    layer
  }

  def buildOutputLayer(classNum: Int): Layer = {
    val layer = new OutputLayer
    layer.mapSize = new Size(1, 1)
    layer.outMapNum = classNum
    layer
  }
}

/**
 * scale size for conv and sampling, can have different x and y
 */
class Size extends Serializable {
  final var x: Int = 0
  final var y: Int = 0

  def this(x: Int, y: Int) {
    this()
    this.x = x
    this.y = y
  }

  override def toString: String = {
    val s: StringBuilder = new StringBuilder("Size(").append(" x = ").append(x).append(" y= ").append(y).append(")")
    s.toString
  }

  /**
   * divide a size with other size
   *
   * @param scaleSize
   * @return
   */
  def divide(scaleSize: Size): Size = {
    val x: Int = this.x / scaleSize.x
    val y: Int = this.y / scaleSize.y
    if (x * scaleSize.x != this.x || y * scaleSize.y != this.y) throw new RuntimeException(this + "can not be divided" + scaleSize)
    new Size(x, y)
  }

  /**
   * subtract a size and add append
   *
   * @param size
   * @param append
   * @return
   */
  def subtract(size: Size, append: Int): Size = {
    val x: Int = this.x - size.x + append
    val y: Int = this.y - size.y + append
    new Size(x, y)
  }
}

abstract class Layer extends Serializable {

  protected var layerType: String = null
  var outMapNum: Int = 0
  private var mapSize: Size = null

  def getMapSize: Size = {
    mapSize
  }

  def setMapSize(mapSize: Size) {
    this.mapSize = mapSize
  }

  def getType: String = {
    layerType
  }
}

class InputLayer extends Layer{
  this.layerType = "input"
}

class ConvLayer extends Layer{

  private var bias: BDV[Double] = null
  this.layerType = "conv"
  private var kernel: Array[Array[BDM[Double]]] = null
  private var kernelSize: Size = null
    
  def initBias(frontMapNum: Int) {
    this.bias = Util.randomArray(outMapNum)
  }

  def initKernel(frontMapNum: Int) {
    this.kernel = Array.ofDim[BDM[Double]](frontMapNum, outMapNum)
    for (i <- 0 until frontMapNum)
      for (j <- 0 until outMapNum)
        kernel(i)(j) = Util.randomMatrix(kernelSize.x, kernelSize.y)
  }

  def getBias = bias

  def setBias(mapNo: Int, value: Double) {
    bias(mapNo) = value
  }
  def getKernelSize = kernelSize

  def setKernelSize(value: Size): this.type = {
    this.kernelSize = value
    this
  }

  def getKernel(i: Int, j: Int): BDM[Double] = {
    kernel(i)(j)
  }
}

class SampLayer extends Layer{
  private var scaleSize: Size = null
  this.layerType = "samp"
  
  def getScaleSize = scaleSize

  def setScaleSize(value: Size): this.type = {
    this.scaleSize = value
    this
  }
}

class OutputLayer extends ConvLayer{
  this.layerType = "output"
  /**
   * kernel size for output layer is equal to map size of last layer
   *
   * @param frontMapNum
   * @param size
   */
  def initOutputKernels(frontMapNum: Int, size: Size) {
    this.setKernelSize(size)
    this.initKernel(frontMapNum)
  }
}