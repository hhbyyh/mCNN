package torch.cnn

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}

/**
  * Created by yuhaoyan on 8/24/2015.
  */
class SampLayer(scaleSize: Scale) extends Module{
   def getScaleSize: Scale = scaleSize

   override def updateOutput(input: Array[BDM[Double]]): Array[BDM[Double]] = {
     val lastMapNum: Int = input.length
     val output = new Array[BDM[Double]](lastMapNum)
     var i: Int = 0
     while (i < lastMapNum) {
       val lastMap: BDM[Double] = input(i)
       val scaleSize: Scale = this.getScaleSize
       output(i) = StochasticGradient.scaleMatrix(lastMap, scaleSize)
       i += 1
     }
     this.output = output
     output
   }

   override def prevError(
       layerInput: Array[BDM[Double]],
       nextDelta: Array[BDM[Double]]): Array[BDM[Double]] = {
     val mapNum: Int = layerInput.length
     val errors = new Array[BDM[Double]](mapNum)
     var m: Int = 0
     val scale: Scale = this.getScaleSize
     while (m < mapNum) {
       val nextError: BDM[Double] = nextDelta(m)
       val map: BDM[Double] = layerInput(m)
       var outMatrix: BDM[Double] = (1.0 - map)
       outMatrix = map :* outMatrix
       outMatrix = outMatrix :* StochasticGradient.kronecker(nextError, scale)
       errors(m) = outMatrix
       m += 1
     }
     errors
   }
 }
