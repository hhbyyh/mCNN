package torch.cnn

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, fliplr, flipud, sum}
import breeze.numerics._

/**
  * Created by yuhaoyan on 8/23/2015.
  */
class ConvLayer private extends Module {

   private var bias: BDV[Double] = null
   private var kernel: Array[Array[BDM[Double]]] = null
   private var kernelSize: Scale = null

   def this(mapNum: Int, kernelSize: Scale) {
     this
     this.mapNum = mapNum
     this.kernelSize = kernelSize
   }

   private[cnn] def initBias(frontMapNum: Int) {
     this.bias = BDV.zeros[Double](mapNum)
   }

   private[cnn] def initKernel(frontMapNum: Int) {
     this.kernel = Array.ofDim[BDM[Double]](frontMapNum, mapNum)
     for (i <- 0 until frontMapNum)
       for (j <- 0 until mapNum)
         kernel(i)(j) = (BDM.rand[Double](kernelSize.x, kernelSize.y) - 0.05) / 10.0
   }

   def getBias: BDV[Double] = bias
   def setBias(mapNo: Int, value: Double): this.type = {
     bias(mapNo) = value
     this
   }

   def getKernelSize: Scale = kernelSize
   def getKernel(i: Int, j: Int): BDM[Double] = kernel(i)(j)

   override def updateOutput(input: Tensor): Tensor = {
     val mapNum: Int = this.mapNum
     val lastMapNum: Int = input.length
     val output = new Array[BDM[Double]](mapNum)
     var j = 0
     val oldBias = this.bias
     while (j < mapNum) {
       var sum: BDM[Double] = null
       var i = 0
       while (i < lastMapNum) {
         val lastMap = input(i)
         val kernel = this.getKernel(i, j)
         if (sum == null) {
           sum = StochasticGradient.convnValid(lastMap, kernel)
         }
         else {
           sum += StochasticGradient.convnValid(lastMap, kernel)
         }
         i += 1
       }
       sum = sigmoid(sum + oldBias(j))
       output(j) = sum
       j += 1
     }
     this.output = output
     output
   }

   override def prevError(input: Tensor, nextDelta: Tensor): Tensor = {
     val mapNum: Int = input.length
     val nextMapNum: Int = this.getOutMapNum
     val errors = new Array[BDM[Double]](mapNum)
     var i = 0
     while (i < mapNum) {
       var sum: BDM[Double] = null // sum for every kernel
       var j = 0
       while (j < nextMapNum) {
         val nextError = nextDelta(j)
         val kernel = this.getKernel(i, j)
         // rotate kernel by 180 degrees and get full convolution
         if (sum == null) {
           sum = StochasticGradient.convnFull(nextError, flipud(fliplr(kernel)))
         }
         else {
           sum += StochasticGradient.convnFull(nextError, flipud(fliplr(kernel)))
         }
         j += 1
       }
       errors(i) = sum
       i += 1
     }
     this.gradient = grad(nextDelta, input)
     errors
   }

    def grad(layerError: Array[BDM[Double]],
             input: Array[BDM[Double]]): (Array[Array[BDM[Double]]], Array[Double]) = {
     val kernelGradient = getKernelsGradient(layerError, input)
     val biasGradient = getBiasGradient(layerError)
     (kernelGradient, biasGradient)
   }

   /**
    * get kernels gradient
    */
   private def getKernelsGradient(
       layerError: Array[BDM[Double]],
       input: Array[BDM[Double]]): Array[Array[BDM[Double]]] = {
     val mapNum: Int = this.getOutMapNum
     val lastMapNum: Int = input.length
     val delta = Array.ofDim[BDM[Double]](lastMapNum, mapNum)
     var j = 0
     while (j < mapNum) {
       var i = 0
       while (i < lastMapNum) {
         val error = layerError(j)
         val deltaKernel = StochasticGradient.convnValid(input(i), error)
         delta(i)(j) = deltaKernel
         i += 1
       }
       j += 1
     }
     delta
   }

   /**
    * get bias gradient
    *
    * @param errors errors of this layer
    */
   private def getBiasGradient(errors: Array[BDM[Double]]): Array[Double] = {
     val mapNum: Int = this.getOutMapNum
     var j: Int = 0
     val gradient = new Array[Double](mapNum)
     while (j < mapNum) {
       val error: BDM[Double] = errors(j)
       val deltaBias: Double = sum(error)
       gradient(j) = deltaBias
       j += 1
     }
     gradient
   }
 }
