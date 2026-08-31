package com.aosmith.type

import android.content.Context
import android.content.SharedPreferences

class Prefs(context: Context) {
    private val sp: SharedPreferences = context.getSharedPreferences("type", Context.MODE_PRIVATE)

    var autocorrect: Boolean
        get() = sp.getBoolean("autocorrect", true)
        set(v) = sp.edit().putBoolean("autocorrect", v).apply()

    var liveSuggestions: Boolean
        get() = sp.getBoolean("live_suggestions", true)
        set(v) = sp.edit().putBoolean("live_suggestions", v).apply()

    var adaptiveKeys: Boolean
        get() = sp.getBoolean("adaptive_keys", true)
        set(v) = sp.edit().putBoolean("adaptive_keys", v).apply()

    var learnFromTyping: Boolean
        get() = sp.getBoolean("learn_typing", true)
        set(v) = sp.edit().putBoolean("learn_typing", v).apply()

    /** Bumped when the user asks to forget learned typing; the service reacts to the change. */
    var personalCleared: Long
        get() = sp.getLong("personal_cleared", 0)
        set(v) = sp.edit().putLong("personal_cleared", v).apply()

    var haptics: Boolean
        get() = sp.getBoolean("haptics", true)
        set(v) = sp.edit().putBoolean("haptics", v).apply()

    /** Id from [com.aosmith.type.model.ModelCatalog], or a bare file name for imported models. */
    var modelId: String?
        get() = sp.getString("model_id", null)
        set(v) = sp.edit().putString("model_id", v).apply()

    /** Inference threads; 0 means pick automatically. */
    var threads: Int
        get() = sp.getInt("threads", 0)
        set(v) = sp.edit().putInt("threads", v).apply()

    fun registerListener(l: SharedPreferences.OnSharedPreferenceChangeListener) = sp.registerOnSharedPreferenceChangeListener(l)
    fun unregisterListener(l: SharedPreferences.OnSharedPreferenceChangeListener) = sp.unregisterOnSharedPreferenceChangeListener(l)
}
