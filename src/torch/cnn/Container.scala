package torch.cnn

abstract class Container extends Module {

   def add(module: Module): this.type = {
     modules += module
     this
   }

   override def zeroGradParameters(): Unit = {
     modules.foreach(_.zeroGradParameters())
   }

   override def updateParameters(learningRate: Double): Unit = {
     modules.foreach(_.updateParameters(learningRate))
   }
 }





