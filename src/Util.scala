package mycnn

import java.io.Serializable
import java.util.{Arrays, Random, Set}

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, kron}

object Util {

  def randomMatrix(row: Int, col: Int): BDM[Double] = {
    val r = new Random(7)
    val m = BDM.zeros[Double](row, col)
    var i: Int = 0
    while (i < row) {
      {
        {
          var j: Int = 0
          while (j < col) {
            {
              m(i, j) = (r.nextDouble - 0.05) / 10
            }
            j += 1
          }
        }
      }
      i += 1
    }
    m
  }

  /**
   * 随机初始化一维向量
   *
   * @param len
   * @return
   */
  def randomArray(len: Int): BDV[Double] = {
    BDV.zeros[Double](len)
  }

  /**
   * 克罗内克积,对矩阵进行扩展
   *
   * @param matrix
   * @param scale
   * @return
   */
  def kronecker(matrix: BDM[Double], scale: Size): BDM[Double] = {
    val ones = BDM.ones[Double](scale.x, scale.y)
    return kron(matrix, ones)
  }

  /**
   * 对矩阵进行均值缩小
   *
   * @param matrix
   */
  def scaleMatrix(matrix: BDM[Double], scale: Size): BDM[Double] = {
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
        var si =  i * scale.x
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
    return outMatrix
  }

  /**
   * 计算full模式的卷积
   *
   * @param matrix
   * @param kernel
   * @return
   */
  def convnFull(matrix: BDM[Double], kernel: BDM[Double]): BDM[Double] = {
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
    return convnValid(extendMatrix, kernel)
  }

  /**
   * 计算valid模式的卷积
   *
   * @param matrix
   * @param kernel
   * @return
   */
  def convnValid(matrix: BDM[Double], kernel: BDM[Double]): BDM[Double] = {
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
    return outMatrix
  }

  /**
   * 对errors[...][j]元素求和
   *
   * @param errors
   * @param j
   * @return
   */
  def sum(errors: Array[Array[BDM[Double]]], j: Int): BDM[Double] = {
    val result = errors(0)(j).copy
    for (i <- 1 until errors.size) {
      result += errors(i)(j)
    }
    return result
  }

  def getMaxIndex(out: Array[Double]): Int = {
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
    return index
  }

}
