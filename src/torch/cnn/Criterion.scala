package torch.cnn

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.mllib.regression.LabeledPoint

/**
  * Created by yuhaoyan on 8/23/2015.
  */
class Criterion extends Serializable{
   def setOutLayerErrors(
       record: LabeledPoint,
       output: Array[BDM[Double]]): (Boolean, Array[BDM[Double]]) = {
     val mapNum: Int = output.length
     val target: Array[Double] = new Array[Double](mapNum)
     val outValues: Array[Double] = output.map(m => m(0, 0))

     val label = record.label.toInt
     target(label) = 1
     val layerError: Array[BDM[Double]] = (0 until mapNum).map(i => {
       val errorMatrix = new BDM[Double](1, 1)
       errorMatrix(0, 0) = outValues(i) * (1 - outValues(i)) * (target(i) - outValues(i))
       errorMatrix
     }).toArray
     val outClass = StochasticGradient.getMaxIndex(outValues)
     (label == outClass, layerError)

   }
 }
