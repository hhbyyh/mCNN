package torch.cnn

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, kron}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by yuhaoyan on 8/23/2015.
  */

class StochasticGradient(topology: Module, criterion: Criterion) extends Serializable{
   type Tensor = Array[BDM[Double]]
   type Gradient = (Array[Array[BDM[Double]]], Array[Double])
   val maxIterations = 25
   val batchSize = 1000
   private var ALPHA: Double = 0.85

   def setup {
     val layers = topology.modules
     var frontMapNum = 1
     var frontMapSize = new Scale(28, 28)
     var i = 0
     while (i < layers.length) {
       val layer: Module = layers(i)
       if(layer.isInstanceOf[ConvLayer]){
         val convLayer = layer.asInstanceOf[ConvLayer]
         convLayer.setMapSize(frontMapSize.subtract(convLayer.getKernelSize, 1))
         convLayer.initKernel(frontMapNum)
         convLayer.initBias(frontMapNum)
       }
       else if (layer.isInstanceOf[SampLayer]){
         val sampLayer = layer.asInstanceOf[SampLayer]
         sampLayer.setOutMapNum(frontMapNum)
         sampLayer.setMapSize(frontMapSize.divide(sampLayer.getScaleSize))
       }
       frontMapNum = layer.getOutMapNum
       frontMapSize = layer.getMapSize
       i += 1
     }
   }

   def train(trainSet: RDD[LabeledPoint]): Unit = {
     setup
     var t = 0
     val trainSize = trainSet.count().toInt
     val gZero = train(trainSet.first)._2
     gZero.foreach(tu => if (tu != null){
       tu._1.foreach(m => m.foreach(x => x -= x))
       (0 until tu._2.length).foreach(i => tu._2(i) = 0)
     })
     var totalCount = 0
     var totalRight = 0
     while (t < maxIterations) {
       val (gradientSum, count) = trainSet
         .sample(false, batchSize.toDouble/trainSize, 42 + t)
         .treeAggregate((gZero, 0))(
           seqOp = (c, v) => {
             val (right, result) = train(v)
             val gradient = result
             (StochasticGradient.combineGradient(c._1, gradient), c._2 + 1)
           },
           combOp = (c1, c2) => {
             (StochasticGradient.combineGradient(c1._1, c2._1), c1._2 + c2._2)
           })

       t += 1
       if (count > 0){
         updateParams(gradientSum, count)
         val p = 1.0 * totalRight / totalCount
         if (t % 10 == 1 && p > 0.96) {
           ALPHA = 0.001 + ALPHA * 0.9
         }
         totalCount += count
         if (totalCount > 10000){
           totalCount = 0
           totalRight = 0
         }
       }
     }
   }

   def trainOneByOne(trainSet: RDD[LabeledPoint]) {
     setup
     var t = 0
     val trainSize = trainSet.count().toInt
     val dataArr = trainSet.collect()
     while (t < maxIterations) {
       val epochsNum = trainSize
       var right = 0
       var count = 0
       var i = 0
       while (i < epochsNum) {
         val record = dataArr(i)
         val result = train(record)
         if (result._1) right += 1
         count += 1
         val gradient: Array[(Array[Array[BDM[Double]]], Array[Double])] = result._2
         updateParams(gradient, 1)
         i += 1
         if(i % 1000 == 0) println(s"$t:\t$i\tsamplesprecision $right/$count = " + 1.0 * right / count)
       }
       val p = 1.0 * right / count
       if (t % 10 == 1 && p > 0.96) {
         ALPHA = 0.001 + ALPHA * 0.9
       }
       t += 1
       println(s"precision $right/$count = $p")
     }
   }

   private def train(record: LabeledPoint): (Boolean, Array[Gradient]) = {
     val input = Vector2Tensor(record.features)
     val output = topology.forward(input)
     val (right, error) = this.criterion.setOutLayerErrors(record, output)
     topology.backward(input, error)
     (right, topology.modules.map(m => m.gradient).toArray)
   }

   /**
    * set inlayer output
    * @param record
    */
   private def Vector2Tensor(record: Vector): Tensor = {
     val mapSize = new Scale(28, 28)
     if (record.size != mapSize.x * mapSize.y) {
       throw new RuntimeException("data size and map size mismatch!")
     }
     val m = new BDM[Double](mapSize.x, mapSize.y)
     var i: Int = 0
     while (i < mapSize.x) {
       var j: Int = 0
       while (j < mapSize.y) {
         m(i, j) = record(mapSize.x * i + j)
         j += 1
       }
       i += 1
     }
     Array(m)
   }

   private def updateParams(
       gradient: Array[(Array[Array[BDM[Double]]], Array[Double])],
       batchSize: Int): Unit = {
     var l: Int = 0
     val layers = topology.modules
     while (l < topology.modules.length) {
       val layer: Module = layers(l)
       if(layer.isInstanceOf[ConvLayer]) {
         updateKernels(layer.asInstanceOf[ConvLayer], gradient(l)._1, batchSize)
         updateBias(layer.asInstanceOf[ConvLayer], gradient(l)._2, batchSize)
       }
       l += 1
     }
   }

   private def updateKernels(
       layer: ConvLayer,
       gradient: Array[Array[BDM[Double]]], batchSize: Int): Unit = {
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

   private def updateBias(layer: ConvLayer, gradient: Array[Double], batchSize: Int): Unit = {
     val gv = new BDV[Double](gradient)
     layer.getBias += gv * ALPHA / batchSize.toDouble
   }
 }

object StochasticGradient {

   private[cnn] def kronecker(matrix: BDM[Double], scale: Scale): BDM[Double] = {
     val ones = BDM.ones[Double](scale.x, scale.y)
     kron(matrix, ones)
   }

   /**
    * return a new matrix that has been scaled down
    *
    * @param matrix
    */
   private[cnn] def scaleMatrix(matrix: BDM[Double], scale: Scale): BDM[Double] = {
     val m: Int = matrix.rows
     val n: Int = matrix.cols
     val sm: Int = m / scale.x
     val sn: Int = n / scale.y
     val outMatrix = new BDM[Double](sm, sn)
     val size = scale.x * scale.y
     var i = 0
     while (i < sm) {
       var j = 0
       while (j < sn) {
         var sum = 0.0
         var si = i * scale.x
         while (si < (i + 1) * scale.x) {
           var sj = j * scale.y
           while (sj < (j + 1) * scale.y) {
             sum += matrix(si, sj)
             sj += 1
           }
           si += 1
         }
         outMatrix(i, j) = sum / size
         j += 1
       }
       i += 1
     }
     outMatrix
   }

   /**
    * full conv
    *
    * @param matrix
    * @param kernel
    * @return
    */
   private[cnn] def convnFull(matrix: BDM[Double], kernel: BDM[Double]): BDM[Double] = {
     val m: Int = matrix.rows
     val n: Int = matrix.cols
     val km: Int = kernel.rows
     val kn: Int = kernel.cols
     val extendMatrix = new BDM[Double](m + 2 * (km - 1), n + 2 * (kn - 1))
     var i = 0
     var j = 0
     while (i < m) {
       while (j < n) {
         extendMatrix(i + km - 1, j + kn - 1) = matrix(i, j)
         j += 1
       }
       i += 1
     }
     convnValid(extendMatrix, kernel)
   }

   /**
    * valid conv
    *
    * @param matrix
    * @param kernel
    * @return
    */
   private[cnn] def convnValid(matrix: BDM[Double], kernel: BDM[Double]): BDM[Double] = {
     val m: Int = matrix.rows
     val n: Int = matrix.cols
     val km: Int = kernel.rows
     val kn: Int = kernel.cols
     val kns: Int = n - kn + 1
     val kms: Int = m - km + 1
     val outMatrix: BDM[Double] = new BDM[Double](kms, kns)
     var i = 0
     while (i < kms) {
       var j = 0
       while (j < kns) {
         var sum = 0.0
         for (ki <- 0 until km) {
           for (kj <- 0 until kn)
             sum += matrix(i + ki, j + kj) * kernel(ki, kj)
         }
         outMatrix(i, j) = sum
         j += 1
       }
       i += 1
     }
     outMatrix
   }

   private[cnn] def getMaxIndex(out: Array[Double]): Int = {
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

   private[cnn] def combineGradient(
       g1: Array[(Array[Array[BDM[Double]]], Array[Double])],
       g2: Array[(Array[Array[BDM[Double]]], Array[Double])]):
   Array[(Array[Array[BDM[Double]]], Array[Double])] = {

     val l = g1.length
     var li = 0
     while(li < l){
       if (g1(li) != null){
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
