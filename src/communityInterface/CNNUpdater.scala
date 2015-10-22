package org.apache.spark.ml.ann

import breeze.linalg.{*, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, axpy => Baxpy,
sum => Bsum}
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.optimization.Updater

private[ann] class CNNUpdater(alpha: Double) extends Updater {

  override def compute(
      weightsOld: Vector,
      gradient: Vector,
      stepSize: Double,
      iter: Int,
      regParam: Double): (Vector, Double) = {
    val thisIterStepSize = stepSize
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    Baxpy(-thisIterStepSize, gradient.toBreeze * alpha, brzWeights)
    (Vectors.fromBreeze(brzWeights), 0)
  }
}
