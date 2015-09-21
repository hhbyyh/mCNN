/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package hhbyyh.mCNN

import java.io.Serializable


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
  private[mCNN] def divide(scaleSize: Scale): Scale = {
    val x: Int = this.x / scaleSize.x
    val y: Int = this.y / scaleSize.y
    if (x * scaleSize.x != this.x || y * scaleSize.y != this.y){
      throw new RuntimeException(this + "can not be divided" + scaleSize)
    }
    new Scale(x, y)
  }

  private[mCNN] def multiply(scaleSize: Scale): Scale = {
    val x: Int = this.x * scaleSize.x
    val y: Int = this.y * scaleSize.y
    new Scale(x, y)
  }

  /**
   * subtract a scale and add append
   */
  private[mCNN] def subtract(other: Scale, append: Int): Scale = {
    val x: Int = this.x - other.x + append
    val y: Int = this.y - other.y + append
    new Scale(x, y)
  }
}
