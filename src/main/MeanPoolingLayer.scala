package hhbyyh.mCNN

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, fliplr, flipud, sum}


class MeanPoolingLayer private[mCNN](
                                      inMapNum: Int, outMapNum: Int, scaleSize: Scale)
  extends CNNLayer(inMapNum: Int, outMapNum: Int, scaleSize: Scale){

  def getScaleSize: Scale = scaleSize

  override def forward(input: Array[BDM[Double]]): Array[BDM[Double]] = {
    val lastMapNum: Int = input.length
    val output = new Array[BDM[Double]](lastMapNum)
    var i: Int = 0
    while (i < lastMapNum) {
      val lastMap: BDM[Double] = input(i)
      val scaleSize: Scale = this.getScaleSize
      output(i) = CNN.scaleMatrix(lastMap, scaleSize)
      i += 1
    }
    output
  }

  override def prevDelta(
                          nextDelta: Array[BDM[Double]],
                          layerInput: Array[BDM[Double]]): Array[BDM[Double]] = {
    val mapNum: Int = layerInput.length
    val errors = new Array[BDM[Double]](mapNum)
    var m: Int = 0
    val scale: Scale = this.getScaleSize
    while (m < mapNum) {
      val nextError: BDM[Double] = nextDelta(m)
      val map: BDM[Double] = layerInput(m)
      var outMatrix: BDM[Double] = (1.0 - map)
      outMatrix = map :* outMatrix
      outMatrix = outMatrix :* CNN.kronecker(nextError, scale)
      errors(m) = outMatrix
      m += 1
    }
    errors
  }

}
