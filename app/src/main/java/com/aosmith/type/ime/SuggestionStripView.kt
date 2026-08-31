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
 * The row above the keys: a "fix sentence" button on the left, then up to three suggestion
 * chips. Suggestions from the model are tinted; the first chip can be an undo affordance.
 */
class SuggestionStripView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : LinearLayout(context, attrs) {

    interface Listener {
        fun onSuggestionPicked(text: String, isTypedWord: Boolean)
        fun onFixSentence()
        fun onUndo()
    }

    data class Suggestion(val text: String, val fromModel: Boolean = false, val isTypedWord: Boolean = false)

    var listener: Listener? = null

    private val colorText = ContextCompat.getColor(context, R.color.kb_strip_text)
    private val colorMuted = ContextCompat.getColor(context, R.color.kb_strip_text_muted)
    private val colorModel = ContextCompat.getColor(context, R.color.kb_strip_llm)
    private val colorDivider = ContextCompat.getColor(context, R.color.kb_strip_divider)
    private val dp = resources.displayMetrics.density

    private val fixButton: TextView = makeChip().apply {
        text = "✨"
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 17f)
        layoutParams = LayoutParams((48 * dp).toInt(), LayoutParams.MATCH_PARENT)
        setOnClickListener { listener?.onFixSentence() }
        contentDescription = context.getString(R.string.strip_fix)
    }
    private val chips: List<TextView> = List(3) {
        makeChip().apply {
            layoutParams = LayoutParams(0, LayoutParams.MATCH_PARENT, 1f)
            visibility = View.INVISIBLE
        }
    }
    private val statusView: TextView = makeChip().apply {
        layoutParams = LayoutParams(0, LayoutParams.MATCH_PARENT, 1f)
        setTextColor(colorMuted)
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
        visibility = View.GONE
    }

    var fixEnabled: Boolean = true
        set(value) {
            field = value
            fixButton.alpha = if (value) 1f else 0.35f
            fixButton.isEnabled = value
        }

    private val dividers = ArrayList<View>()

    init {
        orientation = HORIZONTAL
        filterTouchesWhenObscured = true
        setBackgroundColor(ContextCompat.getColor(context, R.color.kb_strip_background))
        minimumHeight = resources.getDimensionPixelSize(R.dimen.kb_strip_height)
        addView(fixButton)
        addView(statusView)
        chips.forEachIndexed { i, chip ->
            if (i > 0) {
                val d = makeDivider()
                dividers += d
                addView(d)
            }
            addView(chip)
        }
        syncDividers()
    }

    /** A divider is only visible between two visible chips. */
    private fun syncDividers() {
        dividers.forEachIndexed { i, d ->
            val visible = chips[i].visibility == View.VISIBLE && chips[i + 1].visibility == View.VISIBLE
            d.visibility = if (visible) View.VISIBLE else View.INVISIBLE
        }
    }

    private fun makeChip(): TextView = TextView(context).apply {
        gravity = Gravity.CENTER
        maxLines = 1
        isSingleLine = true
        ellipsize = android.text.TextUtils.TruncateAt.END
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 16f)
        setTextColor(colorText)
        setPadding((6 * dp).toInt(), 0, (6 * dp).toInt(), 0)
        val out = TypedValue()
        context.theme.resolveAttribute(android.R.attr.selectableItemBackground, out, true)
        setBackgroundResource(out.resourceId)
        isClickable = true
    }

    private fun makeDivider(): View = View(context).apply {
        layoutParams = LayoutParams((1 * dp).toInt(), (22 * dp).toInt()).apply { gravity = Gravity.CENTER_VERTICAL }
        setBackgroundColor(colorDivider)
        tag = "divider"
    }

    fun showSuggestions(items: List<Suggestion>) {
        statusView.visibility = View.GONE
        chips.forEachIndexed { i, chip ->
            val item = items.getOrNull(i)
            if (item == null) {
                chip.visibility = View.INVISIBLE
                chip.setOnClickListener(null)
            } else {
                chip.visibility = View.VISIBLE
                chip.text = if (item.isTypedWord) "“${item.text}”" else item.text
                chip.setTextColor(if (item.fromModel) colorModel else colorText)
                chip.typeface = if (item.fromModel) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
                styleAsChip(chip)
                chip.setOnClickListener { listener?.onSuggestionPicked(item.text, item.isTypedWord) }
            }
        }
        syncDividers()
    }

    /**
     * Word-key mode: the few words that can complete what is being typed, drawn as real keys.
     * Used when the dictionary trie narrows the possibilities down far enough.
     */
    fun showWordKeys(words: List<String>) {
        statusView.visibility = View.GONE
        chips.forEachIndexed { i, chip ->
            val word = words.getOrNull(i)
            if (word == null) {
                chip.visibility = View.INVISIBLE
                chip.setOnClickListener(null)
                styleAsChip(chip)
            } else {
                chip.visibility = View.VISIBLE
                chip.text = word
                chip.setTextColor(ContextCompat.getColor(context, R.color.kb_key_text))
                chip.typeface = Typeface.DEFAULT_BOLD
                styleAsKey(chip)
                chip.setOnClickListener { listener?.onSuggestionPicked(word, false) }
            }
        }
        syncDividers()
    }

    private val keyBackground: android.graphics.drawable.GradientDrawable
        get() = android.graphics.drawable.GradientDrawable().apply {
            cornerRadius = resources.getDimension(R.dimen.kb_key_radius)
            setColor(ContextCompat.getColor(context, R.color.kb_key))
        }

    private fun styleAsKey(chip: TextView) {
        val pad = (4 * dp).toInt()
        val bg = android.graphics.drawable.RippleDrawable(
            android.content.res.ColorStateList.valueOf(ContextCompat.getColor(context, R.color.kb_key_pressed)),
            keyBackground,
            null,
        )
        chip.background = android.graphics.drawable.InsetDrawable(bg, (3 * dp).toInt(), pad, (3 * dp).toInt(), pad)
    }

    private fun styleAsChip(chip: TextView) {
        val out = TypedValue()
        context.theme.resolveAttribute(android.R.attr.selectableItemBackground, out, true)
        chip.setBackgroundResource(out.resourceId)
    }

    fun showUndo(original: String) {
        statusView.visibility = View.GONE
        chips.forEachIndexed { i, chip ->
            if (i == 0) {
                chip.visibility = View.VISIBLE
                chip.text = "↶ $original"
                chip.setTextColor(colorText)
                chip.typeface = Typeface.DEFAULT
                styleAsChip(chip)
                chip.setOnClickListener { listener?.onUndo() }
            } else {
                chip.visibility = View.INVISIBLE
                styleAsChip(chip)
                chip.setOnClickListener(null)
            }
        }
        syncDividers()
    }

    fun showStatus(text: String) {
        chips.forEach { it.visibility = View.GONE }
        statusView.text = text
        statusView.visibility = View.VISIBLE
        syncDividers()
    }

    fun clear() {
        statusView.visibility = View.GONE
        chips.forEach {
            it.visibility = View.INVISIBLE
            it.setOnClickListener(null)
        }
        syncDividers()
    }
}
