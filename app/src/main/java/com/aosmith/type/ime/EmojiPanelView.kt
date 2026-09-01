package com.aosmith.type.ime

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.VelocityTracker
import android.view.View
import android.view.ViewConfiguration
import android.widget.OverScroller
import androidx.core.content.ContextCompat
import com.aosmith.type.R
import kotlin.math.abs

/**
 * Canvas-drawn emoji picker shown in place of the key rows: category tabs on top, a
 * scrollable grid, and a bottom row with "abc" (back to letters) and backspace. Keeps the
 * exact height of the letters layout so the strip above never jumps.
 */
class EmojiPanelView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    interface Listener {
        fun onEmojiPicked(emoji: String)
        fun onBackToLetters()
        fun onBackspace()
    }

    var listener: Listener? = null
    var hapticsEnabled: Boolean = true

    /** Static categories from [EmojiData]; the recents tab is prepended internally. */
    var categories: List<EmojiData.Category> = emptyList()
        set(value) {
            field = value
            if (selectedTab >= tabCount) selectedTab = 0
            invalidate()
        }

    /** Most recently used emoji, newest first. Shown as the first tab when non-empty. */
    var recents: List<String> = emptyList()
        set(value) {
            field = value
            invalidate()
        }

    private var selectedTab = 1 // 0 is recents; land on the first real category by default
        set(value) {
            field = value
            scrollY2 = 0f
            scroller.forceFinished(true)
            invalidate()
        }

    private val tabCount: Int get() = categories.size + 1

    private fun gridEmoji(): List<String> =
        if (selectedTab == 0) recents else categories.getOrNull(selectedTab - 1)?.emoji ?: emptyList()

    private val dp = resources.displayMetrics.density
    private val keyHeight = resources.getDimension(R.dimen.kb_key_height)
    private val rowGap = resources.getDimension(R.dimen.kb_row_gap)
    private val sidePadding = resources.getDimension(R.dimen.kb_side_padding)
    private val bottomPadding = resources.getDimension(R.dimen.kb_bottom_padding)
    private val radius = resources.getDimension(R.dimen.kb_key_radius)
    private val labelSizeSmall = resources.getDimension(R.dimen.kb_key_text_small)

    private val colorBackground = ContextCompat.getColor(context, R.color.kb_background)
    private val colorKeySpecial = ContextCompat.getColor(context, R.color.kb_key_special)
    private val colorKeyPressed = ContextCompat.getColor(context, R.color.kb_key_pressed)
    private val colorText = ContextCompat.getColor(context, R.color.kb_key_text_special)
    private val colorAccent = ContextCompat.getColor(context, R.color.kb_key_action)

    private val emojiPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { textAlign = Paint.Align.CENTER }
    private val keyPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { textAlign = Paint.Align.CENTER }
    private val tmpRect = RectF()

    private val tabHeight get() = keyHeight * 0.75f
    private val bottomRowTop get() = height - bottomPadding - keyHeight
    private val gridTop get() = rowGap + tabHeight + rowGap
    private val gridBottom get() = bottomRowTop - rowGap
    private val columns = 8

    // ---- scrolling -----------------------------------------------------------------------

    // "scrollY" is taken by View; this is the grid's own scroll offset.
    private var scrollY2 = 0f
    private val scroller = OverScroller(context)
    private var velocity: VelocityTracker? = null
    private val touchSlop = ViewConfiguration.get(context).scaledTouchSlop
    private val maxFling = ViewConfiguration.get(context).scaledMaximumFlingVelocity

    private val cellWidth get() = (width - 2 * sidePadding) / columns
    private val cellHeight get() = cellWidth * 0.85f

    private fun maxScroll(): Float {
        val rows = (gridEmoji().size + columns - 1) / columns
        return (rows * cellHeight - (gridBottom - gridTop)).coerceAtLeast(0f)
    }

    override fun computeScroll() {
        if (scroller.computeScrollOffset()) {
            scrollY2 = scroller.currY.toFloat().coerceIn(0f, maxScroll())
            invalidate()
        }
    }

    // ---- layout & drawing ----------------------------------------------------------------

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val width = MeasureSpec.getSize(widthMeasureSpec)
        // Mirror KeyboardView's letters height exactly (4 rows) so swapping views is seamless.
        val height = (4 * keyHeight + 4 * rowGap + bottomPadding).toInt()
        setMeasuredDimension(width, height)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(colorBackground)
        drawTabs(canvas)
        drawGrid(canvas)
        drawBottomRow(canvas)
    }

    private fun drawTabs(canvas: Canvas) {
        val w = (width - 2 * sidePadding) / tabCount
        emojiPaint.textSize = tabHeight * 0.45f
        for (t in 0 until tabCount) {
            val cx = sidePadding + t * w + w / 2f
            val icon = if (t == 0) "🕘" else categories[t - 1].icon
            emojiPaint.alpha = if (t == selectedTab) 255 else 140
            val cy = rowGap + tabHeight / 2f
            canvas.drawText(icon, cx, cy - (emojiPaint.descent() + emojiPaint.ascent()) / 2f, emojiPaint)
            if (t == selectedTab) {
                keyPaint.color = colorAccent
                canvas.drawRoundRect(cx - w * 0.3f, rowGap + tabHeight - 3 * dp, cx + w * 0.3f, rowGap + tabHeight, 2 * dp, 2 * dp, keyPaint)
            }
        }
        emojiPaint.alpha = 255
    }

    private fun drawGrid(canvas: Canvas) {
        val emoji = gridEmoji()
        if (emoji.isEmpty()) {
            textPaint.color = colorText
            textPaint.textSize = labelSizeSmall
            canvas.drawText(
                context.getString(R.string.emoji_no_recents),
                width / 2f, (gridTop + gridBottom) / 2f, textPaint,
            )
            return
        }
        canvas.save()
        canvas.clipRect(0f, gridTop, width.toFloat(), gridBottom)
        emojiPaint.textSize = cellHeight * 0.55f
        val firstRow = (scrollY2 / cellHeight).toInt()
        val lastRow = ((scrollY2 + gridBottom - gridTop) / cellHeight).toInt()
        for (row in firstRow..lastRow) {
            for (col in 0 until columns) {
                val i = row * columns + col
                if (i >= emoji.size) break
                val cx = sidePadding + col * cellWidth + cellWidth / 2f
                val cy = gridTop + row * cellHeight - scrollY2 + cellHeight / 2f
                if (i == pressedCell) {
                    keyPaint.color = colorKeyPressed
                    canvas.drawRoundRect(
                        cx - cellWidth / 2f + 2 * dp, cy - cellHeight / 2f + 2 * dp,
                        cx + cellWidth / 2f - 2 * dp, cy + cellHeight / 2f - 2 * dp,
                        radius, radius, keyPaint,
                    )
                }
                canvas.drawText(emoji[i], cx, cy - (emojiPaint.descent() + emojiPaint.ascent()) / 2f, emojiPaint)
            }
        }
        canvas.restore()
    }

    private enum class BottomKey { ABC, BACKSPACE }

    private fun bottomKeyRect(key: BottomKey): RectF {
        val w = 1.5f * (width - 2 * sidePadding) / 10f
        val top = bottomRowTop
        return when (key) {
            BottomKey.ABC -> RectF(sidePadding, top, sidePadding + w, top + keyHeight)
            BottomKey.BACKSPACE -> RectF(width - sidePadding - w, top, width - sidePadding, top + keyHeight)
        }
    }

    private fun drawBottomRow(canvas: Canvas) {
        textPaint.textSize = labelSizeSmall
        for (key in BottomKey.entries) {
            val r = bottomKeyRect(key)
            tmpRect.set(r)
            keyPaint.color = if (pressedBottom == key) colorKeyPressed else colorKeySpecial
            canvas.drawRoundRect(tmpRect, radius, radius, keyPaint)
            textPaint.color = colorText
            val label = if (key == BottomKey.ABC) "abc" else "⌫"
            canvas.drawText(label, r.centerX(), r.centerY() - (textPaint.descent() + textPaint.ascent()) / 2f, textPaint)
        }
    }

    // ---- touch ---------------------------------------------------------------------------

    private var downX = 0f
    private var downY = 0f
    private var dragging = false
    private var lastY = 0f
    private var pressedCell = -1
    private var pressedBottom: BottomKey? = null

    private val backspaceRepeat = object : Runnable {
        override fun run() {
            if (pressedBottom != BottomKey.BACKSPACE) return
            repeatFired = true
            listener?.onBackspace()
            postDelayed(this, 45L)
        }
    }
    private var repeatFired = false

    private fun cellAt(x: Float, y: Float): Int {
        if (y < gridTop || y >= gridBottom) return -1
        val col = ((x - sidePadding) / cellWidth).toInt()
        if (col < 0 || col >= columns) return -1
        val row = ((y - gridTop + scrollY2) / cellHeight).toInt()
        val i = row * columns + col
        return if (i in gridEmoji().indices) i else -1
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                scroller.forceFinished(true)
                velocity = VelocityTracker.obtain().also { it.addMovement(event) }
                downX = event.x
                downY = event.y
                lastY = event.y
                dragging = false
                repeatFired = false
                pressedCell = cellAt(event.x, event.y)
                pressedBottom = BottomKey.entries.firstOrNull { bottomKeyRect(it).contains(event.x, event.y) }
                if (pressedCell >= 0 || pressedBottom != null) {
                    if (hapticsEnabled) performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
                }
                if (pressedBottom == BottomKey.BACKSPACE) postDelayed(backspaceRepeat, 400L)
                invalidate()
            }
            MotionEvent.ACTION_MOVE -> {
                velocity?.addMovement(event)
                if (!dragging && pressedBottom == null && downY < bottomRowTop &&
                    abs(event.y - downY) > touchSlop && abs(event.y - downY) > abs(event.x - downX)
                ) {
                    dragging = true
                    pressedCell = -1
                }
                if (dragging) {
                    scrollY2 = (scrollY2 + lastY - event.y).coerceIn(0f, maxScroll())
                    invalidate()
                }
                lastY = event.y
            }
            MotionEvent.ACTION_UP -> {
                velocity?.addMovement(event)
                removeCallbacks(backspaceRepeat)
                when {
                    dragging -> {
                        val v = velocity?.let { it.computeCurrentVelocity(1000, maxFling.toFloat()); it.yVelocity } ?: 0f
                        if (abs(v) > 200f) {
                            scroller.fling(0, scrollY2.toInt(), 0, -v.toInt(), 0, 0, 0, maxScroll().toInt())
                            postInvalidateOnAnimation()
                        }
                    }
                    pressedBottom != null -> {
                        if (bottomKeyRect(pressedBottom!!).contains(event.x, event.y)) {
                            when (pressedBottom) {
                                BottomKey.ABC -> listener?.onBackToLetters()
                                BottomKey.BACKSPACE -> if (!repeatFired) listener?.onBackspace()
                                null -> {}
                            }
                        }
                    }
                    event.y < gridTop -> {
                        val w = (width - 2 * sidePadding) / tabCount
                        val tab = ((event.x - sidePadding) / w).toInt()
                        if (tab in 0 until tabCount) selectedTab = tab
                    }
                    else -> {
                        val cell = cellAt(event.x, event.y)
                        if (cell >= 0 && cell == pressedCell) listener?.onEmojiPicked(gridEmoji()[cell])
                    }
                }
                velocity?.recycle()
                velocity = null
                pressedCell = -1
                pressedBottom = null
                invalidate()
            }
            MotionEvent.ACTION_CANCEL -> {
                removeCallbacks(backspaceRepeat)
                velocity?.recycle()
                velocity = null
                pressedCell = -1
                pressedBottom = null
                dragging = false
                invalidate()
            }
        }
        return true
    }

    init {
        isClickable = true
        isFocusable = false
        filterTouchesWhenObscured = true
        setBackgroundColor(colorBackground)
    }
}
