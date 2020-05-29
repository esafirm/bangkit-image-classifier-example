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

package com.google.firebase.ml.md.kotlin.settings

import android.content.Context
import android.graphics.RectF
import android.preference.PreferenceManager
import androidx.annotation.StringRes
import com.google.android.gms.common.images.Size
import com.google.firebase.ml.md.R
import com.google.firebase.ml.md.kotlin.camera.CameraSizePair
import com.google.firebase.ml.md.kotlin.camera.GraphicOverlay
import com.google.firebase.ml.vision.barcode.FirebaseVisionBarcode

/** Utility class to retrieve shared preferences.  */
object PreferenceUtils {

    fun isAutoSearchEnabled(context: Context): Boolean =
            getBooleanPref(context, R.string.pref_key_enable_auto_search, true)

    fun isMultipleObjectsMode(context: Context): Boolean =
            getBooleanPref(context, R.string.pref_key_object_detector_enable_multiple_objects, false)

    fun isClassificationEnabled(context: Context): Boolean =
            getBooleanPref(context, R.string.pref_key_object_detector_enable_classification, false)

    fun saveStringPreference(context: Context, @StringRes prefKeyId: Int, value: String?) {
        PreferenceManager.getDefaultSharedPreferences(context)
                .edit()
                .putString(context.getString(prefKeyId), value)
                .apply()
    }

    fun getConfirmationTimeMs(context: Context): Int = when {
        isMultipleObjectsMode(context) -> 300
        isAutoSearchEnabled(context) -> getIntPref(context, R.string.pref_key_confirmation_time_in_auto_search, 1500)
        else -> getIntPref(context, R.string.pref_key_confirmation_time_in_manual_search, 500)
    }

    private fun getIntPref(context: Context, @StringRes prefKeyId: Int, defaultValue: Int): Int {
        val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
        val prefKey = context.getString(prefKeyId)
        return sharedPreferences.getInt(prefKey, defaultValue)
    }

    fun getUserSpecifiedPreviewSize(context: Context): CameraSizePair? {
        return try {
            val previewSizePrefKey = context.getString(R.string.pref_key_rear_camera_preview_size)
            val pictureSizePrefKey = context.getString(R.string.pref_key_rear_camera_picture_size)
            val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
            CameraSizePair(
                    Size.parseSize(sharedPreferences.getString(previewSizePrefKey, null)),
                    Size.parseSize(sharedPreferences.getString(pictureSizePrefKey, null)))
        } catch (e: Exception) {
            null
        }
    }

    private fun getBooleanPref(context: Context, @StringRes prefKeyId: Int, defaultValue: Boolean): Boolean =
            PreferenceManager.getDefaultSharedPreferences(context).getBoolean(context.getString(prefKeyId), defaultValue)
}
