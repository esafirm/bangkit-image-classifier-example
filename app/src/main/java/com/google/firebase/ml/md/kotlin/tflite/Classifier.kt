/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package com.google.firebase.ml.md.kotlin.tflite

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import com.google.firebase.ml.md.kotlin.Logger.d
import com.google.firebase.ml.md.kotlin.Logger.v
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.MappedByteBuffer
import java.util.*
import kotlin.math.min

/**
 * A classifier specialized to label images using TensorFlow Lite.
 */
abstract class Classifier protected constructor(activity: Activity?, device: Device?, numThreads: Int) {
    /**
     * The model type used for classification.
     */
    enum class Model {
        QUANTIZED_EFFICIENTNET,
        QUANTIZED_MOBILENET,
        QUANTIZED_HOTDOG
    }

    /**
     * The runtime device type used for executing classification.
     */
    enum class Device {
        CPU, NNAPI, GPU
    }

    /**
     * The loaded TensorFlow Lite model.
     */
    private var tfliteModel: MappedByteBuffer?

    /**
     * Get the image size along the x axis.
     */
    /**
     * Image size along the x axis.
     */
    val imageSizeX: Int

    /**
     * Get the image size along the y axis.
     */
    /**
     * Image size along the y axis.
     */
    val imageSizeY: Int

    /**
     * Optional GPU delegate for accleration.
     */
    private var gpuDelegate: GpuDelegate? = null

    /**
     * Optional NNAPI delegate for accleration.
     */
    private var nnApiDelegate: NnApiDelegate? = null

    /**
     * An instance of the driver class to run model inference with Tensorflow Lite.
     */
    protected var tflite: Interpreter?

    /**
     * Options for configuring the Interpreter.
     */
    private val tfliteOptions = Interpreter.Options()

    /**
     * Labels corresponding to the output of the vision model.
     */
    private val labels: List<String>

    /**
     * Input image TensorBuffer.
     */
    private var inputImageBuffer: TensorImage

    /**
     * Output probability TensorBuffer.
     */
    private val outputProbabilityBuffer: TensorBuffer

    /**
     * Processer to apply post processing of the output probability.
     */
    private val probabilityProcessor: TensorProcessor

    /**
     * Runs inference and returns the classification results.
     */
    fun recognizeImage(bitmap: Bitmap, sensorOrientation: Int): List<Recognition> {
        val startTimeForLoadImage = SystemClock.uptimeMillis()
        inputImageBuffer = loadImage(bitmap, sensorOrientation)
        val endTimeForLoadImage = SystemClock.uptimeMillis()
        v("Timecost to load the image: " + (endTimeForLoadImage - startTimeForLoadImage))

        // Runs the inference call.
        val startTimeForReference = SystemClock.uptimeMillis()
        tflite!!.run(inputImageBuffer.buffer, outputProbabilityBuffer.buffer.rewind())
        val endTimeForReference = SystemClock.uptimeMillis()
        v("Timecost to run model inference: " + (endTimeForReference - startTimeForReference))

        // Gets the map of label and probability.
        val labeledProbability = TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
            .mapWithFloatValue

        // Gets top-k results.
        return getTopKProbability(labeledProbability)
    }

    /**
     * Closes the interpreter and model to release resources.
     */
    fun close() {
        tflite?.close()
        tflite = null

        gpuDelegate?.close()
        gpuDelegate = null

        nnApiDelegate?.close()
        nnApiDelegate = null

        tfliteModel = null
    }

    /**
     * Loads input image, and applies preprocessing.
     */
    private fun loadImage(bitmap: Bitmap, sensorOrientation: Int): TensorImage {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap)

        // Creates processor for the TensorImage.
        val cropSize = min(bitmap.width, bitmap.height)
        val numRotation = sensorOrientation / 90
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(ResizeOp(imageSizeX, imageSizeY, ResizeMethod.NEAREST_NEIGHBOR))
            .add(Rot90Op(numRotation))
            .add(preprocessNormalizeOp)
            .build()
        return imageProcessor.process(inputImageBuffer)
    }

    /**
     * Gets the name of the model file stored in Assets.
     */
    protected abstract val modelPath: String

    /**
     * Gets the name of the label file stored in Assets.
     */
    protected abstract val labelPath: String

    /**
     * Gets the TensorOperator to nomalize the input image in preprocessing.
     */
    protected abstract val preprocessNormalizeOp: TensorOperator?

    /**
     * Gets the TensorOperator to dequantize the output probability in post processing.
     *
     *
     * For quantized model, we need de-quantize the prediction with NormalizeOp (as they are all
     * essentially linear transformation). For float model, de-quantize is not required. But to
     * uniform the API, de-quantize is added to float model too. Mean and std are set to 0.0f and
     * 1.0f, respectively.
     */
    protected abstract val postprocessNormalizeOp: TensorOperator?

    companion object {
        /**
         * Number of results to show in the UI.
         */
        private const val MAX_RESULTS = 3

        /**
         * Creates a classifier with the provided configuration.
         *
         * @param activity   The current Activity.
         * @param model      The model to use for classification.
         * @param device     The device to use for classification.
         * @param numThreads The number of threads to use for classification.
         * @return A classifier with the desired configuration.
         */
        @Throws(IOException::class)
        fun create(activity: Activity?, model: Model, device: Device?, numThreads: Int): Classifier {
            return when (model) {
                Model.QUANTIZED_EFFICIENTNET -> ClassifierQuantizedEfficientNet(activity, device, numThreads)
                Model.QUANTIZED_MOBILENET -> ClassifierQuantizedMobileNet(activity, device, numThreads)
                Model.QUANTIZED_HOTDOG -> ClassifierHotDog(activity, device, numThreads)
            }
        }

        /**
         * Gets the top-k results.
         */
        private fun getTopKProbability(labelProb: Map<String, Float>): List<Recognition> {
            // Find the best classifications.
            val pq = PriorityQueue(
                MAX_RESULTS,
                Comparator<Recognition> { lhs, rhs -> // Intentionally reversed to put high confidence at the head of the queue.
                    (rhs.confidence!!).compareTo(lhs.confidence!!)
                })
            for ((key, value) in labelProb) {
                pq.add(Recognition("" + key, key, value, null))
            }
            val recognitions = ArrayList<Recognition>()
            val recognitionsSize = min(pq.size, MAX_RESULTS)
            for (i in 0 until recognitionsSize) {
                recognitions.add(pq.poll())
            }
            return recognitions
        }
    }

    /**
     * Initializes a `Classifier`.
     */
    init {
        tfliteModel = FileUtil.loadMappedFile(activity!!, modelPath)
        when (device) {
            Device.NNAPI -> {
                nnApiDelegate = NnApiDelegate()
                tfliteOptions.addDelegate(nnApiDelegate)
            }
            Device.GPU -> {
                gpuDelegate = GpuDelegate()
                tfliteOptions.addDelegate(gpuDelegate)
            }
            Device.CPU -> {
            }
        }
        tfliteOptions.setNumThreads(numThreads)
        tflite = Interpreter(tfliteModel!!, tfliteOptions)

        // Loads labels out from the label file.
        labels = FileUtil.loadLabels(activity, labelPath!!)

        // Reads type and shape of input and output tensors, respectively.
        val imageTensorIndex = 0
        val imageShape = tflite!!.getInputTensor(imageTensorIndex).shape() // {1, height, width, 3}
        imageSizeY = imageShape[1]
        imageSizeX = imageShape[2]
        val imageDataType = tflite!!.getInputTensor(imageTensorIndex).dataType()
        val probabilityTensorIndex = 0
        val probabilityShape = tflite!!.getOutputTensor(probabilityTensorIndex).shape() // {1, NUM_CLASSES}
        val probabilityDataType = tflite!!.getOutputTensor(probabilityTensorIndex).dataType()

        // Creates the input tensor.
        inputImageBuffer = TensorImage(imageDataType)

        // Creates the output tensor and its processor.
        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType)

        // Creates the post processor for the output probability.
        probabilityProcessor = TensorProcessor.Builder().add(postprocessNormalizeOp).build()
        d("Created a Tensorflow Lite Image Classifier.")
    }

    /**
     * An immutable result returned by a Classifier describing what was recognized.
     */
    class Recognition(
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        val id: String?,
        /**
         * Display name for the recognition.
         */
        val title: String?,
        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        val confidence: Float?,
        /**
         * Optional location within the source image for the location of the recognized object.
         */
        private var location: RectF?) {

        fun getLocation(): RectF {
            return RectF(location)
        }

        fun setLocation(location: RectF?) {
            this.location = location
        }

        override fun toString(): String {
            var resultString = ""
            if (id != null) {
                resultString += "[$id] "
            }
            if (title != null) {
                resultString += "$title "
            }
            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f)
            }
            if (location != null) {
                resultString += location.toString() + " "
            }
            return resultString.trim { it <= ' ' }
        }

        fun formattedString() = "${title?.capitalize()} - ${String.format("(%.1f%%) ", confidence!! * 100.0f)}"
    }
}