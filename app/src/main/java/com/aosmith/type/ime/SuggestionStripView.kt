package com.aosmith.type.ime

import android.content.Context
import android.graphics.Typeface
import android.util.AttributeSet
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.aosmith.type.R

/**
 * Suggestion pills floating over the top key row — the strip reserves no height of its
 * own. Up to three chips (model suggestions tinted, the first chip can be an undo
 * affordance) plus a status line; when there is nothing to show, nothing is drawn and
 * touches fall through to the keys beneath. The ✨ sentence fix lives on the keyboard
 * itself (short press of the emoji key).
 */
class SuggestionStripView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : LinearLayout(context, attrs) {

    interface Listener {
        fun onSuggestionPicked(text: String, isTypedWord: Boolean)
        fun onUndo()
    }

    data class Suggestion(val text: String, val fromModel: Boolean = false, val isTypedWord: Boolean = false)

    var listener: Listener? = null

    /**
     * Bubble mode: when set and [caretProvider] yields fresh geometry, suggestions render
     * as floating bubbles beside the word in the app instead of chips in this band (the
     * service collapses the band). Fields that never report a caret keep the chips.
     */
    var bubbles: SuggestionBubbles? = null
    var caretProvider: (() -> android.graphics.Rect?)? = null

    private fun bubbleCaret(): android.graphics.Rect? =
        if (bubbles == null) null else caretProvider?.invoke()

    private val colorText = ContextCompat.getColor(context, R.color.kb_strip_text)
    private val colorMuted = ContextCompat.getColor(context, R.color.kb_strip_text_muted)
    private val colorModel = ContextCompat.getColor(context, R.color.kb_strip_llm)
    private val dp = resources.displayMetrics.density

    private val chips: List<TextView> = List(3) {
        makeChip().apply {
            layoutParams = LayoutParams(0, LayoutParams.MATCH_PARENT, 1f)
            visibility = View.GONE
        }
    }
    private val statusView: TextView = makeChip().apply {
        layoutParams = LayoutParams(0, LayoutParams.MATCH_PARENT, 1f)
        setTextColor(colorMuted)
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
        visibility = View.GONE
    }

    init {
        orientation = HORIZONTAL
        filterTouchesWhenObscured = true
        // Transparent overlay: unclaimed touches pass through to the keyboard below.
        setBackgroundColor(0)
        chips.forEach(::addView)
        addView(statusView)
    }

    private fun makeChip(): TextView = TextView(context).apply {
        gravity = Gravity.CENTER
        maxLines = 1
        isSingleLine = true
        ellipsize = android.text.TextUtils.TruncateAt.END
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 16f)
        setTextColor(colorText)
        setPadding((6 * dp).toInt(), 0, (6 * dp).toInt(), 0)
        isClickable = true
        elevation = 3 * dp
        stylePill(this)
    }

    /** Floating over keys, every visible element needs an opaque pill behind it. */
    private fun stylePill(chip: TextView) {
        val pill = android.graphics.drawable.GradientDrawable().apply {
            cornerRadius = resources.getDimension(R.dimen.kb_key_radius)
            setColor(ContextCompat.getColor(context, R.color.kb_strip_background))
        }
        val bg = android.graphics.drawable.RippleDrawable(
            android.content.res.ColorStateList.valueOf(ContextCompat.getColor(context, R.color.kb_key_pressed)),
            pill,
            null,
        )
        val pad = (4 * dp).toInt()
        chip.background = android.graphics.drawable.InsetDrawable(bg, (3 * dp).toInt(), pad, (3 * dp).toInt(), pad)
    }

    fun showSuggestions(items: List<Suggestion>) {
        bubbleCaret()?.let { caret ->
            bubbles?.show(items, caret)
            hideChips()
            return
        }
        bubbles?.dismiss()
        statusView.visibility = View.GONE
        chips.forEachIndexed { i, chip ->
            val item = items.getOrNull(i)
            if (item == null) {
                chip.visibility = View.GONE
                chip.setOnClickListener(null)
            } else {
                chip.visibility = View.VISIBLE
                chip.text = if (item.isTypedWord) "“${item.text}”" else item.text
                chip.setTextColor(if (item.fromModel) colorModel else colorText)
                chip.typeface = if (item.fromModel) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
                chip.setOnClickListener { listener?.onSuggestionPicked(item.text, item.isTypedWord) }
            }
        }
    }

    /**
     * Word-key mode: the few words that can complete what is being typed. As an overlay
     * these draw the same pills; the takeover keys below carry the emphasis.
     */
    fun showWordKeys(words: List<String>) {
        bubbleCaret()?.let { caret ->
            bubbles?.show(words.map { Suggestion(it) }, caret)
            hideChips()
            return
        }
        bubbles?.dismiss()
        statusView.visibility = View.GONE
        chips.forEachIndexed { i, chip ->
            val word = words.getOrNull(i)
            if (word == null) {
                chip.visibility = View.GONE
                chip.setOnClickListener(null)
            } else {
                chip.visibility = View.VISIBLE
                chip.text = word
                chip.setTextColor(colorText)
                chip.typeface = Typeface.DEFAULT_BOLD
                chip.setOnClickListener { listener?.onSuggestionPicked(word, false) }
            }
        }
    }

    fun showUndo(original: String) {
        bubbleCaret()?.let { caret ->
            bubbles?.showUndo(original, caret)
            hideChips()
            return
        }
        bubbles?.dismiss()
        statusView.visibility = View.GONE
        chips.forEachIndexed { i, chip ->
            if (i == 0) {
                chip.visibility = View.VISIBLE
                chip.text = "↶ $original"
                chip.setTextColor(colorText)
                chip.typeface = Typeface.DEFAULT
                chip.setOnClickListener { listener?.onUndo() }
            } else {
                chip.visibility = View.GONE
                chip.setOnClickListener(null)
            }
        }
    }

    fun showStatus(text: String) {
        bubbles?.dismiss()
        chips.forEach { it.visibility = View.GONE }
        statusView.text = text
        statusView.visibility = View.VISIBLE
    }

    private fun hideChips() {
        statusView.visibility = View.GONE
        chips.forEach {
            it.visibility = View.GONE
            it.setOnClickListener(null)
        }
    }

    fun clear() {
        bubbles?.dismiss()
        hideChips()
    }
}
