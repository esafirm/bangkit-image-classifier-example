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

import android.animation.AnimatorInflater
import android.animation.AnimatorSet
import android.annotation.SuppressLint
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.hardware.Camera
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.View.OnClickListener
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Observer
import androidx.lifecycle.ViewModelProviders
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.chip.Chip
import com.google.android.material.floatingactionbutton.ExtendedFloatingActionButton
import com.google.common.base.Objects
import com.google.firebase.ml.md.R
import com.google.firebase.ml.md.kotlin.camera.CameraSource
import com.google.firebase.ml.md.kotlin.camera.CameraSourcePreview
import com.google.firebase.ml.md.kotlin.camera.GraphicOverlay
import com.google.firebase.ml.md.kotlin.camera.WorkflowModel
import com.google.firebase.ml.md.kotlin.camera.WorkflowModel.WorkflowState
import com.google.firebase.ml.md.kotlin.objectdetection.MultiObjectProcessor
import com.google.firebase.ml.md.kotlin.objectdetection.ProminentObjectProcessor
import com.google.firebase.ml.md.kotlin.productsearch.BottomSheetScrimView
import com.google.firebase.ml.md.kotlin.settings.PreferenceUtils
import com.google.firebase.ml.md.kotlin.settings.SettingsActivity
import com.google.firebase.ml.md.kotlin.tflite.Classifier
import java.io.IOException

/** Demonstrates the object detection and visual search workflow using camera preview.  */
class LiveObjectDetectionActivity : AppCompatActivity(), OnClickListener {

    private var cameraSource: CameraSource? = null
    private var workflowModel: WorkflowModel? = null
    private var currentWorkflowState: WorkflowState? = null

    private var objectThumbnailForBottomSheet: Bitmap? = null
    private var slidingSheetUpFromHiddenState: Boolean = false

    private val classifier by lazy {
        ClassifierHelper(this, ClassifierSpec(
            Classifier.Model.QUANTIZED_EFFICIENTNET,
            Classifier.Device.CPU,
            1
        ))
    }

    /* --------------------------------------------------- */
    /* > Views */
    /* --------------------------------------------------- */

    private val preview by lazy { findViewById<CameraSourcePreview>(R.id.camera_preview) }
    private val graphicOverlay by lazy { findViewById<GraphicOverlay>(R.id.camera_preview_graphic_overlay) }

    private var searchButtonAnimator: AnimatorSet? = null
    private val searchButton by lazy { findViewById<ExtendedFloatingActionButton>(R.id.product_search_button) }

    private val progress by lazy { findViewById<ProgressBar>(R.id.search_progress_bar) }

    private var bottomSheetBehavior: BottomSheetBehavior<View>? = null
    private var bottomSheetScrimView: BottomSheetScrimView? = null

    private val settingsButton by lazy { findViewById<View>(R.id.settings_button) }
    private val flashButton by lazy { findViewById<View>(R.id.flash_button) }

    private var promptChipAnimator: AnimatorSet? = null
    private val promptChip by lazy { findViewById<Chip>(R.id.bottom_prompt_chip) }

    private val bottomSheetBestView by lazy { findViewById<TextView>(R.id.bottom_sheet_best_match) }
    private val bottomSheetCaptionView by lazy { findViewById<TextView>(R.id.bottom_sheet_caption) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_live_object_kotlin)

        graphicOverlay.apply {
            setOnClickListener(this@LiveObjectDetectionActivity)
            cameraSource = CameraSource(this)
        }
        promptChipAnimator =
            (AnimatorInflater.loadAnimator(this, R.animator.bottom_prompt_chip_enter) as AnimatorSet).apply {
                setTarget(promptChip)
            }
        searchButton.setOnClickListener(this@LiveObjectDetectionActivity)
        searchButtonAnimator =
            (AnimatorInflater.loadAnimator(this, R.animator.search_button_enter) as AnimatorSet).apply {
                setTarget(searchButton)
            }
        setUpBottomSheet()

        findViewById<View>(R.id.close_button).setOnClickListener(this)
        flashButton.setOnClickListener(this@LiveObjectDetectionActivity)
        settingsButton.setOnClickListener(this@LiveObjectDetectionActivity)

        setUpWorkflowModel()
    }

    override fun onResume() {
        super.onResume()

        workflowModel?.markCameraFrozen()
        settingsButton.isEnabled = true
        bottomSheetBehavior?.state = BottomSheetBehavior.STATE_HIDDEN
        currentWorkflowState = WorkflowState.NOT_STARTED
        cameraSource?.setFrameProcessor(
            if (PreferenceUtils.isMultipleObjectsMode(this)) {
                MultiObjectProcessor(graphicOverlay, workflowModel!!)
            } else {
                ProminentObjectProcessor(graphicOverlay, workflowModel!!)
            }
        )
        workflowModel?.setWorkflowState(WorkflowState.DETECTING)
    }

    override fun onPause() {
        super.onPause()
        currentWorkflowState = WorkflowState.NOT_STARTED
        stopCameraPreview()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraSource?.release()
        cameraSource = null
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
            R.id.product_search_button -> {
                searchButton.isEnabled = false
                workflowModel?.onSearchButtonClicked()
            }
            R.id.bottom_sheet_scrim_view -> bottomSheetBehavior?.setState(BottomSheetBehavior.STATE_HIDDEN)
            R.id.close_button -> onBackPressed()
            R.id.flash_button -> {
                if (flashButton.isSelected) {
                    flashButton.isSelected = false
                    cameraSource?.updateFlashMode(Camera.Parameters.FLASH_MODE_OFF)
                } else {
                    flashButton.isSelected = true
                    cameraSource?.updateFlashMode(Camera.Parameters.FLASH_MODE_TORCH)
                }
            }
            R.id.settings_button -> {
                settingsButton.isEnabled = false
                startActivity(Intent(this, SettingsActivity::class.java))
            }
        }
    }

    private fun startCameraPreview() {
        val cameraSource = this.cameraSource ?: return
        val workflowModel = this.workflowModel ?: return
        if (!workflowModel.isCameraLive) {
            try {
                workflowModel.markCameraLive()
                preview?.start(cameraSource)
            } catch (e: IOException) {
                Log.e(TAG, "Failed to start camera preview!", e)
                cameraSource.release()
                this.cameraSource = null
            }
        }
    }

    private fun stopCameraPreview() {
        if (workflowModel?.isCameraLive == true) {
            workflowModel!!.markCameraFrozen()
            flashButton.isSelected = false
            preview?.stop()
        }
    }

    private fun setUpBottomSheet() {
        bottomSheetBehavior = BottomSheetBehavior.from(findViewById(R.id.bottom_sheet))
        bottomSheetBehavior?.setBottomSheetCallback(
            object : BottomSheetBehavior.BottomSheetCallback() {
                override fun onStateChanged(bottomSheet: View, newState: Int) {
                    Log.d(TAG, "Bottom sheet new state: $newState")
                    bottomSheetScrimView?.visibility =
                        if (newState == BottomSheetBehavior.STATE_HIDDEN) View.GONE else View.VISIBLE
                    graphicOverlay.clear()

                    when (newState) {
                        BottomSheetBehavior.STATE_HIDDEN -> workflowModel?.setWorkflowState(WorkflowState.DETECTING)
                        BottomSheetBehavior.STATE_COLLAPSED,
                        BottomSheetBehavior.STATE_EXPANDED,
                        BottomSheetBehavior.STATE_HALF_EXPANDED -> slidingSheetUpFromHiddenState = false
                        BottomSheetBehavior.STATE_DRAGGING, BottomSheetBehavior.STATE_SETTLING -> {
                        }
                    }
                }

                override fun onSlide(bottomSheet: View, slideOffset: Float) {
                    val searchedObject = workflowModel!!.searchedObject.value
                    if (searchedObject == null || java.lang.Float.isNaN(slideOffset)) {
                        return
                    }

                    val bottomSheetBehavior = bottomSheetBehavior ?: return
                    val collapsedStateHeight = bottomSheetBehavior.peekHeight.coerceAtMost(bottomSheet.height)
                    val bottomBitmap = objectThumbnailForBottomSheet ?: return
                    if (slidingSheetUpFromHiddenState) {
                        val thumbnailSrcRect = graphicOverlay.translateRect(searchedObject.boundingBox)
                        bottomSheetScrimView?.updateWithThumbnailTranslateAndScale(
                            bottomBitmap,
                            collapsedStateHeight,
                            slideOffset,
                            thumbnailSrcRect)
                    } else {
                        bottomSheetScrimView?.updateWithThumbnailTranslate(
                            bottomBitmap, collapsedStateHeight, slideOffset, bottomSheet)
                    }
                }
            })

        bottomSheetScrimView = findViewById<BottomSheetScrimView>(R.id.bottom_sheet_scrim_view).apply {
            setOnClickListener(this@LiveObjectDetectionActivity)
        }
    }

    private fun setUpWorkflowModel() {
        workflowModel = ViewModelProviders.of(this).get(WorkflowModel::class.java).apply {

            // Observes the workflow state changes, if happens, update the overlay view indicators and
            // camera preview state.
            workflowState.observe(this@LiveObjectDetectionActivity, Observer { workflowState ->
                if (workflowState == null || Objects.equal(currentWorkflowState, workflowState)) {
                    return@Observer
                }
                currentWorkflowState = workflowState
                Log.d(TAG, "Current workflow state: ${workflowState.name}")

                if (PreferenceUtils.isAutoSearchEnabled(this@LiveObjectDetectionActivity)) {
                    stateChangeInAutoSearchMode(workflowState)
                } else {
                    stateChangeInManualSearchMode(workflowState)
                }
            })

            // Observes changes on the object to search, if happens, fire product search request.
            objectToSearch.observe(this@LiveObjectDetectionActivity, Observer { detectObject ->
                Logger.d("Detect object: $detectObject")
                val capturedBitmap = detectObject.getBitmap()
                classifier.execute(
                    bitmap = capturedBitmap,
                    onError = {
                        Toast.makeText(
                            this@LiveObjectDetectionActivity,
                            it.message,
                            Toast.LENGTH_SHORT
                        ).show()
                    },
                    onResult = {
                        showReesult(capturedBitmap, it)
                    }
                )
            })

        }
    }

    @SuppressLint("SetTextI18n")
    private fun showReesult(bitmap: Bitmap, results: List<Classifier.Recognition>) {
        progress.visibility = View.GONE
        objectThumbnailForBottomSheet = bitmap
        slidingSheetUpFromHiddenState = true

        // Create caption, the unclean way
        if (results.size > 1) {
            val resultString = results
                .subList(1, results.size)
                .foldIndexed("") { index, acc, recognition ->
                    "${acc}${index + 2}. ${recognition.formattedString()}\n"
                }
            bottomSheetCaptionView.text = resultString
        }

        bottomSheetBestView.text = "1. ${results.first().formattedString()}"
        bottomSheetBehavior?.state = BottomSheetBehavior.STATE_EXPANDED
    }


    private fun stateChangeInAutoSearchMode(workflowState: WorkflowState) {
        val wasPromptChipGone = promptChip.visibility == View.GONE

        searchButton.visibility = View.GONE
        progress.visibility = View.GONE
        when (workflowState) {
            WorkflowState.DETECTING, WorkflowState.DETECTED, WorkflowState.CONFIRMING -> {
                promptChip.visibility = View.VISIBLE
                promptChip.setText(
                    if (workflowState == WorkflowState.CONFIRMING)
                        R.string.prompt_hold_camera_steady
                    else
                        R.string.prompt_point_at_an_object)
                startCameraPreview()
            }
            WorkflowState.CONFIRMED -> {
                promptChip.visibility = View.VISIBLE
                promptChip.setText(R.string.prompt_processing)
                stopCameraPreview()
            }
            WorkflowState.SEARCHING -> {
                progress.visibility = View.VISIBLE
                promptChip.visibility = View.VISIBLE
                promptChip.setText(R.string.prompt_processing)
                stopCameraPreview()
            }
            WorkflowState.SEARCHED -> {
                promptChip.visibility = View.GONE
                stopCameraPreview()
            }
            else -> promptChip.visibility = View.GONE
        }

        val shouldPlayPromptChipEnteringAnimation = wasPromptChipGone && promptChip.visibility == View.VISIBLE
        if (shouldPlayPromptChipEnteringAnimation && promptChipAnimator?.isRunning == false) {
            promptChipAnimator?.start()
        }
    }

    private fun stateChangeInManualSearchMode(workflowState: WorkflowState) {
        val wasPromptChipGone = promptChip.visibility == View.GONE
        val wasSearchButtonGone = searchButton.visibility == View.GONE

        progress.visibility = View.GONE
        when (workflowState) {
            WorkflowState.DETECTING, WorkflowState.DETECTED, WorkflowState.CONFIRMING -> {
                promptChip.visibility = View.VISIBLE
                promptChip.setText(R.string.prompt_point_at_an_object)
                searchButton.visibility = View.GONE
                startCameraPreview()
            }
            WorkflowState.CONFIRMED -> {
                promptChip.visibility = View.GONE
                searchButton.visibility = View.VISIBLE
                searchButton.isEnabled = true
                searchButton.setBackgroundColor(Color.WHITE)
                startCameraPreview()
            }
            WorkflowState.SEARCHING -> {
                promptChip.visibility = View.GONE
                searchButton.visibility = View.VISIBLE
                searchButton.isEnabled = false
                searchButton.setBackgroundColor(Color.GRAY)
                progress.visibility = View.VISIBLE
                stopCameraPreview()
            }
            WorkflowState.SEARCHED -> {
                promptChip.visibility = View.GONE
                searchButton.visibility = View.GONE
                stopCameraPreview()
            }
            else -> {
                promptChip.visibility = View.GONE
                searchButton.visibility = View.GONE
            }
        }

        val shouldPlayPromptChipEnteringAnimation = wasPromptChipGone && promptChip.visibility == View.VISIBLE
        promptChipAnimator?.let {
            if (shouldPlayPromptChipEnteringAnimation && !it.isRunning) it.start()
        }

        val shouldPlaySearchButtonEnteringAnimation = wasSearchButtonGone && searchButton.visibility == View.VISIBLE
        searchButtonAnimator?.let {
            if (shouldPlaySearchButtonEnteringAnimation && !it.isRunning) it.start()
        }
    }

    companion object {
        private const val TAG = "LiveObjectActivity"
    }
}
