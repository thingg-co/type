package com.aosmith.type.ime

import android.content.res.ColorStateList
import android.graphics.Rect
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.RippleDrawable
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.widget.LinearLayout
import android.widget.PopupWindow
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.aosmith.type.R

/**
 * Suggestion bubbles floating beside the word being typed, in the app's own text area:
 * translucent pills positioned just above the caret from CursorAnchorInfo geometry, so
 * the keyboard itself stays pure keys. The popup borrows the IME window's token (the
 * same mechanism as key-preview popups), is touchable only over its own pills, and is
 * dismissed whenever the caret goes stale, the field ends, or the window hides.
 */
class SuggestionBubbles(private val anchor: View) {

    interface Listener {
        fun onBubblePicked(text: String, isTypedWord: Boolean)
        fun onBubbleUndo()
    }

    var listener: Listener? = null

    private val context get() = anchor.context
    private val dp = anchor.resources.displayMetrics.density
    private val row = LinearLayout(anchor.context).apply { orientation = LinearLayout.HORIZONTAL }
    private val popup = PopupWindow(row, LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
        isFocusable = false
        isTouchable = true
        isClippingEnabled = false
        inputMethodMode = PopupWindow.INPUT_METHOD_NOT_NEEDED
        elevation = 6 * dp
    }

    private fun pill(text: String, model: Boolean, undo: Boolean, onTap: () -> Unit): TextView =
        TextView(context).apply {
            gravity = Gravity.CENTER
            maxLines = 1
            isSingleLine = true
            this.text = text
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
            setTextColor(
                ContextCompat.getColor(context, if (model) R.color.kb_strip_llm else R.color.kb_strip_text),
            )
            typeface = if (model) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
            setPadding((10 * dp).toInt(), (5 * dp).toInt(), (10 * dp).toInt(), (5 * dp).toInt())
            val body = GradientDrawable().apply {
                cornerRadius = 12 * dp
                setColor(ContextCompat.getColor(context, R.color.kb_strip_background))
                alpha = 232 // translucent: the text beneath stays readable through the edge
            }
            background = RippleDrawable(
                ColorStateList.valueOf(ContextCompat.getColor(context, R.color.kb_key_pressed)),
                body,
                null,
            )
            isClickable = true
            setOnClickListener { onTap() }
            if (undo) alpha = 0.95f
        }

    /** Shows [items] as bubbles above the caret; [caret] is in screen coordinates. */
    fun show(items: List<SuggestionStripView.Suggestion>, caret: Rect) {
        if (items.isEmpty()) {
            dismiss()
            return
        }
        row.removeAllViews()
        items.take(3).forEach { item ->
            row.addView(
                pill(if (item.isTypedWord) "“${item.text}”" else item.text, item.fromModel, undo = false) {
                    listener?.onBubblePicked(item.text, item.isTypedWord)
                },
                LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                    marginEnd = (4 * dp).toInt()
                },
            )
        }
        place(caret)
    }

    fun showUndo(original: String, caret: Rect) {
        row.removeAllViews()
        row.addView(pill("↶ $original", model = false, undo = true) { listener?.onBubbleUndo() })
        place(caret)
    }

    private fun place(caret: Rect) {
        row.measure(View.MeasureSpec.UNSPECIFIED, View.MeasureSpec.UNSPECIFIED)
        val w = row.measuredWidth
        val h = row.measuredHeight
        // Popup coordinates are relative to the IME window's origin, which sits at the
        // keyboard's top — not the screen's. Convert the caret's screen rect; above the
        // keyboard the y goes negative, which clipping-disabled popups accept (the key
        // preview trick).
        val winLoc = IntArray(2)
        val scrLoc = IntArray(2)
        anchor.getLocationInWindow(winLoc)
        anchor.getLocationOnScreen(scrLoc)
        val originX = scrLoc[0] - winLoc[0]
        val originY = scrLoc[1] - winLoc[1]
        val screenW = anchor.resources.displayMetrics.widthPixels
        val xScreen = (caret.left - w / 4).coerceIn((4 * dp).toInt(), (screenW - w - (4 * dp).toInt()).coerceAtLeast(0))
        val x = xScreen - originX
        val y = caret.top - originY - h - (6 * dp).toInt()
        if (popup.isShowing) {
            popup.update(x, y, -1, -1)
        } else if (anchor.windowToken != null) {
            try {
                popup.showAtLocation(anchor, Gravity.NO_GRAVITY, x, y)
            } catch (e: Exception) {
                android.util.Log.w("SuggestionBubbles", "show failed", e)
            }
        }
    }

    val isShowing: Boolean get() = popup.isShowing

    fun dismiss() {
        if (popup.isShowing) popup.dismiss()
    }
}
