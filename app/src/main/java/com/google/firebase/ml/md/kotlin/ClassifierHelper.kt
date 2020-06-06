package com.google.firebase.ml.md.kotlin

import android.app.Activity
import android.graphics.Bitmap
import com.google.firebase.ml.md.kotlin.tflite.Classifier
import java.io.IOException
import java.util.concurrent.Executors
import kotlin.system.measureTimeMillis

data class ClassifierSpec(
        val model: Classifier.Model,
        val device: Classifier.Device,
        val numThreads: Int
)

class ClassifierHelper(
        private val activity: Activity,
        private val spec: ClassifierSpec
) {

    private var classifier: Classifier? = null
    private val executors = Executors.newSingleThreadExecutor()

    fun execute(
            bitmap: Bitmap,
            onError: (Exception) -> Unit,
            onResult: (List<Classifier.Recognition>) -> Unit
    ) {
        val mainOnError = { e: Exception -> activity.runOnUiThread { onError(e) } }
        val mainOnResult = { r: List<Classifier.Recognition> -> activity.runOnUiThread { onResult(r) } }

        executors.execute {
            createClassifier(mainOnError)
            processImage(bitmap, mainOnResult)
        }
    }

    private fun createClassifier(onError: (Exception) -> Unit) {
        if (classifier != null) return

        val (model, device, numThreads) = spec

        if (device === Classifier.Device.GPU
                && (model === Classifier.Model.QUANTIZED_MOBILENET || model === Classifier.Model.QUANTIZED_EFFICIENTNET)) {
            Logger.d("Not creating classifier: GPU doesn't support quantized models.")
            onError(IllegalStateException("Error regarding GPU support for Quant models[CHAR_LIMIT=60]"))
            return
        }
        try {
            Logger.d("Creating classifier (model=$model, device=$device, numThreads=$numThreads)")
            classifier = Classifier.create(activity, model, device, numThreads)
        } catch (e: IOException) {
            Logger.e("Failed to create classifier: $e")
        }
    }

    private fun processImage(bitmap: Bitmap, onResult: (List<Classifier.Recognition>) -> Unit) {
        val currentClassifier = classifier ?: throw  IllegalStateException("Classifier not ready!")

        measureTimeMillis {
            val results = currentClassifier.recognizeImage(bitmap, 0)
            onResult(results)
            Logger.d("Result ready: $results")
        }.also {
            Logger.v("Detect: $it ms")
        }
    }

    fun close() {
        classifier?.close()
    }
}