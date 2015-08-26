package torch.cnn

/**
  * Created by yuhaoyan on 8/24/2015.
  */
class Sequential extends Container {

   override def updateOutput(input: Tensor): Tensor = {
     var i = 0
     var result = input
     while(i < modules.length){
       result = modules(i).forward(result)
       i += 1
     }
     result
   }

   override def prevError(input: Tensor, nextError: Tensor) : Tensor = {
     var i = modules.length - 1
     var error = nextError
     while(i > 0){
       val input = modules(i - 1).output
       error = modules(i).backward(input, error)
       i -= 1
     }
     modules(0).backward(input, error)
     error
   }

 }
