package torch.cnn

import java.io.Serializable

/**
  * Created by yuhaoyan on 8/24/2015.
  */
class TorchUtil {

 }

/**
  * scale size for conv and sampling, can have different x and y
  */
class Scale(var x: Int, var y: Int) extends Serializable {

   /**
    * divide a scale with other scale
    *
    * @param scaleSize
    * @return
    */
   private[cnn] def divide(scaleSize: Scale): Scale = {
     val x: Int = this.x / scaleSize.x
     val y: Int = this.y / scaleSize.y
     if (x * scaleSize.x != this.x || y * scaleSize.y != this.y){
       throw new RuntimeException(this + "can not be divided" + scaleSize)
     }
     new Scale(x, y)
   }

   private[cnn] def multiply(scaleSize: Scale): Scale = {
     val x: Int = this.x * scaleSize.x
     val y: Int = this.y * scaleSize.y
     new Scale(x, y)
   }

   /**
    * subtract a scale and add append
    */
   private[cnn] def subtract(other: Scale, append: Int): Scale = {
     val x: Int = this.x - other.x + append
     val y: Int = this.y - other.y + append
     new Scale(x, y)
   }
 }