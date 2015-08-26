package torch.cnn

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import scala.collection.mutable.ArrayBuffer

abstract class Module extends Serializable{
  type Tensor = Array[BDM[Double]]
  type Gradient = (Array[Array[BDM[Double]]], Array[Double])

  var output: Tensor = null
  var gradient: Gradient = null
  val modules: ArrayBuffer[Module] = new ArrayBuffer[Module]()

  protected var mapNum: Int = 0
  private var mapSize: Scale = null

  def getOutMapNum: Int = mapNum
  def setOutMapNum(value: Int): this.type = {
    this.mapNum = value
    this
  }

  def getMapSize: Scale = mapSize
  def setMapSize(mapSize: Scale): this.type = {
    this.mapSize = mapSize
    this
  }

  def forward(input: Tensor): Tensor = {
    updateOutput(input)
  }

  def backward(input: Tensor, nextError: Tensor): Tensor = {
    prevError(input, nextError)
  }

  def updateOutput(input: Tensor): Tensor = {
    this.output = input
    input
  }

  def prevError(input: Tensor, nextError: Tensor): Tensor

  def accGradParameters(input: Tensor, gradOutput: Tensor, scale: Double): Unit ={}

  def zeroGradParameters(): Unit = {}

  def updateParameters(learningRate: Double): Unit = {}
}
