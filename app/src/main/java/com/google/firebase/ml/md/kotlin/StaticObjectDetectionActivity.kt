/*
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.firebase.ml.md.kotlin

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.firebase.ml.md.R
import com.google.firebase.ml.md.kotlin.productsearch.BottomSheetScrimView
import com.google.firebase.ml.md.kotlin.tflite.Classifier
import com.google.firebase.ml.md.kotlin.tflite.Classifier.*
import java.io.IOException
import java.util.concurrent.Executors
import kotlin.system.measureTimeMillis

/** Demonstrates the object detection and visual search workflow using static image.  */
open class StaticObjectDetectionActivity : AppCompatActivity(), View.OnClickListener {

    private var loadingView: View? = null
    private var inputImageView: ImageView? = null
    private var dotViewContainer: ViewGroup? = null

    private var bottomSheetBehavior: BottomSheetBehavior<View>? = null
    private var bottomSheetScrimView: BottomSheetScrimView? = null
    private var bottomSheetCaptionText: TextView? = null
    private var bottomSheetBestText: TextView? = null
    private var productRecyclerView: RecyclerView? = null

    private var inputBitmap: Bitmap? = null
    private var dotViewSize: Int = 0
    private var currentSelectedObjectIndex = 0

    private var classifier: Classifier? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_static_object_kotlin)

        loadingView = findViewById<View>(R.id.loading_view).apply {
            setOnClickListener(this@StaticObjectDetectionActivity)
        }

        inputImageView = findViewById(R.id.input_image_view)
        dotViewContainer = findViewById(R.id.dot_view_container)
        dotViewSize = resources.getDimensionPixelOffset(R.dimen.static_image_dot_view_size)

        setUpBottomSheet()

        findViewById<View>(R.id.close_button).setOnClickListener(this)
        findViewById<View>(R.id.photo_library_button).setOnClickListener(this)

        intent?.data?.let(::detectObjects)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode == Utils.REQUEST_CODE_PHOTO_LIBRARY && resultCode == Activity.RESULT_OK) {
            data?.data?.let(::detectObjects)
        } else {
            super.onActivityResult(requestCode, resultCode, data)
        }
    }

    override fun onBackPressed() {
        if (bottomSheetBehavior?.state != BottomSheetBehavior.STATE_HIDDEN) {
            bottomSheetBehavior?.setState(BottomSheetBehavior.STATE_HIDDEN)
        } else {
            super.onBackPressed()
        }
    }

    override fun onClick(view: View) {
        when (view.id) {
            R.id.close_button -> onBackPressed()
            R.id.photo_library_button -> Utils.openImagePicker(this)
            R.id.bottom_sheet_scrim_view -> bottomSheetBehavior?.state = BottomSheetBehavior.STATE_HIDDEN
        }
    }

    @SuppressLint("SetTextI18n")
    private fun showSearchResults(results: List<Recognition>) {
        loadingView?.visibility = View.GONE

        // Create caption, the unclean way
        if (results.size > 1) {
            val resultString = results
                    .subList(1, results.size)
                    .foldIndexed("") { index, acc, recognition ->
                        "${acc}${index + 2}. ${recognition.title}\n"
                    }
            bottomSheetCaptionText?.text = resultString
        }

        bottomSheetBestText?.text = "1. ${results.first().title} - ${results.first().confidence} %"
        bottomSheetBehavior?.peekHeight = PEEK_HEIGHT
        bottomSheetBehavior?.state = BottomSheetBehavior.STATE_COLLAPSED
    }

    private fun setUpBottomSheet() {
        val bottomSheetView = findViewById<View>(R.id.bottom_sheet)
        bottomSheetBehavior = BottomSheetBehavior.from(bottomSheetView).apply {
            addBottomSheetCallback(
                    object : BottomSheetBehavior.BottomSheetCallback() {
                        override fun onStateChanged(bottomSheet: View, newState: Int) {
                            Log.d(TAG, "Bottom sheet new state: $newState")
                            bottomSheetScrimView?.visibility =
                                    if (newState == BottomSheetBehavior.STATE_HIDDEN) View.GONE else View.VISIBLE
                        }

                        override fun onSlide(bottomSheet: View, slideOffset: Float) {
                            if (slideOffset.isNaN()) {
                                return
                            }

                            val collapsedStateHeight = bottomSheetBehavior!!.peekHeight.coerceAtMost(bottomSheet.height)
                            bottomSheetScrimView?.updateWithThumbnailTranslate(
                                    inputBitmap!!,
                                    collapsedStateHeight,
                                    slideOffset,
                                    bottomSheet)
                        }
                    }
            )
            state = BottomSheetBehavior.STATE_HIDDEN
        }

        bottomSheetScrimView = findViewById<BottomSheetScrimView>(R.id.bottom_sheet_scrim_view).apply {
            setOnClickListener(this@StaticObjectDetectionActivity)
        }

        bottomSheetCaptionText = findViewById(R.id.bottom_sheet_caption)
        bottomSheetBestText = findViewById(R.id.bottom_sheet_best_match)
    }

    private fun detectObjects(imageUri: Uri) {
        inputImageView?.setImageDrawable(null)
        dotViewContainer?.removeAllViews()
        currentSelectedObjectIndex = 0

        try {
            inputBitmap = Utils.loadImage(this, imageUri, MAX_IMAGE_DIMENSION)
        } catch (e: IOException) {
            Log.e(TAG, "Failed to load file: $imageUri", e)
            return
        }

        inputImageView?.setImageBitmap(inputBitmap)
        loadingView?.visibility = View.VISIBLE

        Executors.newSingleThreadExecutor().execute {
            recreateClassifier(
                    Model.QUANTIZED_EFFICIENTNET,
                    Device.CPU,
                    1
            )

            inputBitmap?.let {
                processImage(it)
            } ?: throw NullPointerException("Bitmap is null!")
        }
    }

    private fun recreateClassifier(model: Model, device: Device, numThreads: Int) {
        if (classifier != null) {
            Logger.d("Closing classifier.")
            classifier?.close()
            classifier = null
        }
        if (device === Device.GPU
                && (model === Model.QUANTIZED_MOBILENET || model === Model.QUANTIZED_EFFICIENTNET)) {
            Logger.d("Not creating classifier: GPU doesn't support quantized models.")
            runOnUiThread {
                Toast.makeText(
                        this,
                        "Error regarding GPU support for Quant models[CHAR_LIMIT=60]",
                        Toast.LENGTH_LONG
                ).show()
            }
            return
        }
        try {
            Logger.d("Creating classifier (model=$model, device=$device, numThreads=$numThreads)")
            classifier = create(this, model, device, numThreads)
        } catch (e: IOException) {
            Logger.e("Failed to create classifier: $e")
        }
    }

    private fun processImage(rgbFrameBitmap: Bitmap) {
        val currentClassifier = classifier ?: throw  IllegalStateException("Classifier not ready!")

        measureTimeMillis {
            val results = currentClassifier.recognizeImage(rgbFrameBitmap, 0)
            runOnUiThread {
                showSearchResults(results)
            }
            Logger.d("Result ready: $results")
        }.also {
            Logger.v("Detect: $it ms")
        }
    }

    companion object {
        private const val TAG = "StaticObjectActivity"
        private const val MAX_IMAGE_DIMENSION = 1024
        private const val PEEK_HEIGHT = 200
    }
}
