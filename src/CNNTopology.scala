
package org.apache.spark.ml.ann

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV, _}
import org.apache.spark.mllib.neuralNetwork.CNNLayer
import org.apache.spark.mllib.optimization.Updater
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * Feed forward ANN
 * @param layers
 */
private[ann] class CNNTopology private(val layers: Array[Layer]) extends Topology {
  override def getInstance(weights: Vector): TopologyModel = CNNTopologyModel(this, weights)

  override def getInstance(seed: Long): TopologyModel = CNNTopologyModel(this, seed)
}

/**
 * Factory for some of the frequently-used topologies
 */
private[ml] object CNNTopology {
  /**
   * Creates a feed forward topology from the array of layers
   * @param layers array of layers
   * @return feed forward topology
   */
  def apply(layers: Array[Layer]): CNNTopology = {
    new CNNTopology(layers)
  }
}

/**
 * Model of Feed Forward Neural Network.
 * Implements forward, gradient computation and can return weights in vector format.
 * @param layerModels models of layers
 * @param topology topology of the network
 */
private[ml] class CNNTopologyModel private(
    val layerModels: Array[LayerModel],
    val topology: CNNTopology) extends TopologyModel {

  override def forward(data: BDM[Double]): Array[BDM[Double]] = {
    val outputs = new Array[BDM[Double]](layerModels.length)
    outputs(0) = layerModels(0).eval(data)
    for (i <- 1 until layerModels.length) {
      outputs(i) = layerModels(i).eval(outputs(i-1))
    }
    outputs
  }


  private def setOutLayerErrors(
      label: BDM[Double],
      output: BDM[Double]): (Double, BDM[Double]) = {
    val mapNum: Int = output.cols
    val outValues: Array[Double] = output(0,::).t.toArray
    val target: Array[Double] = label(::, 0).toArray
    val layerError: BDM[Double] = new BDM(1, mapNum)
    for(i <- 0 until mapNum){
      layerError(0, i) = outValues(i) * (1 - outValues(i)) * (target(i) - outValues(i))
    }
    (sum(layerError), layerError)
  }

  private def backPropagation(
     lastError: BDM[Double],
     outputs: Array[BDM[Double]]): Array[BDM[Double]] = {
    val errors = new Array[BDM[Double]](layerModels.size)
    errors(layerModels.length - 1) = lastError
    var l: Int = layerModels.size - 2
    while (l >= 0) {
      val layer: LayerModel = layerModels(l + 1)
      errors(l) = layer.prevDelta(errors(l + 1), outputs(l))
      l -= 1
    }
    errors
  }

  override def computeGradient(
      data: BDM[Double],
      target: BDM[Double],
      cumGradient: Vector,
      realBatchSize: Int): Double = {

    val outputs = forward(data)
    val (loss, lastError) = setOutLayerErrors(target, outputs.last)
    val errors = backPropagation(lastError, outputs)

    val grads = new Array[Array[Double]](layerModels.length)
    for (i <- 0 until layerModels.length) {
      val input = if (i == 0) data else outputs(i - 1)
      grads(i) = layerModels(i).grad(errors(i), input)
    }
    // update cumGradient
    val cumGradientArray = cumGradient.toArray
    var offset = 0
    // TODO: extract roll
    for (i <- 0 until grads.length) {
      val gradArray = grads(i)
      var k = 0
      while (k < gradArray.length) {
        cumGradientArray(offset + k) += gradArray(k)
        k += 1
      }
      offset += gradArray.length
    }
    loss
  }

  // TODO: do we really need to copy the weights? they should be read-only
  override def weights(): Vector = {
    // TODO: extract roll
    var size = 0
    for (i <- 0 until layerModels.length) {
      size += layerModels(i).size
    }
    val array = new Array[Double](size)
    var offset = 0
    for (i <- 0 until layerModels.length) {
      val layerWeights = layerModels(i).weights().toArray
      System.arraycopy(layerWeights, 0, array, offset, layerWeights.length)
      offset += layerWeights.length
    }
    Vectors.dense(array)
  }

  override def predict(data: Vector): Vector = {
    val size = data.size
    val result = forward(new BDM[Double](size, 1, data.toArray))
    Vectors.dense(result.last.toArray)
  }
}

/**
 * Fabric for feed forward ANN models
 */
private[ann] object CNNTopologyModel {

  /**
   * Creates a model from a topology and weights
   * @param topology topology
   * @param weights weights
   * @return model
   */
  def apply(topology: CNNTopology, weights: Vector): CNNTopologyModel = {
    val layers = topology.layers
    val layerModels = new Array[LayerModel](layers.length)
    var offset = 0
    for (i <- 0 until layers.length) {
      layerModels(i) = layers(i).getInstance(weights, offset)
      offset += layerModels(i).size
    }
    new CNNTopologyModel(layerModels, topology)
  }

  /**
   * Creates a model given a topology and seed
   * @param topology topology
   * @param seed seed for generating the weights
   * @return model
   */
  def apply(topology: CNNTopology, seed: Long = 11L): CNNTopologyModel = {
    val layers = topology.layers
    val layerModels = new Array[LayerModel](layers.length)
    for(i <- 0 until layers.length){
      layerModels(i) = layers(i).getInstance(seed)
    }
    new CNNTopologyModel(layerModels, topology)
  }

  private[ann] def getMaxIndex(out: Array[Double]): Int = {
    var max: Double = out(0)
    var index: Int = 0
    var i: Int = 1
    while (i < out.length) {
      if (out(i) > max) {
        max = out(i)
        index = i
      }
      i += 1
    }
    index
  }
}

/**
 * Simple updater
 */
private[ann] class CNNUpdater extends Updater {

  override def compute(
                        weightsOld: Vector,
                        gradient: Vector,
                        stepSize: Double,
                        iter: Int,
                        regParam: Double): (Vector, Double) = {
    val thisIterStepSize = stepSize
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    axpy(thisIterStepSize, gradient.toBreeze, brzWeights)
    (Vectors.fromBreeze(brzWeights), 0)
  }
}
