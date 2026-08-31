package com.aosmith.type.ime

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import android.util.AttributeSet
import android.util.TypedValue
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import androidx.core.content.ContextCompat
import com.aosmith.type.R
import kotlin.math.max
import kotlin.math.min

/**
 * Canvas-drawn keyboard. Keys are laid out from [KeyboardLayouts]; letter keys can be
 * dimmed and shrunk by [keyWeights] (the adaptive-key feature), but their hit areas never
 * change, so an unlikely letter stays as easy to tap as before.
 */
class KeyboardView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    interface Listener {
        fun onText(text: String)
        fun onBackspace()
        fun onEnter()
        fun onSpace()
        fun onWordKey(word: String)
        fun onEscapeWordMode()
    }

    enum class Shift { OFF, ON, LOCKED }

    var listener: Listener? = null

    var shift: Shift = Shift.OFF
        set(value) {
            if (field != value) {
                field = value
                invalidate()
            }
        }

    var layer: Layer = Layer.LETTERS
        set(value) {
            if (field != value) {
                field = value
                rebuildKeys()
                invalidate()
            }
        }

    /**
     * Weight per lowercase letter, in [0, 1], for the adaptive-key effect. null means every
     * key is drawn normally (no word in progress or the prefix is unknown).
     */
    var keyWeights: Map<Char, Float>? = null
        set(value) {
            field = value
            invalidate()
        }

    var adaptiveEnabled: Boolean = true
        set(value) {
            field = value
            invalidate()
        }
    var hapticsEnabled: Boolean = true

    /**
     * Word takeover: when set, the letter rows are replaced by these words as huge buttons,
     * with backspace, space, enter and an "abc" escape key keeping the flow available. The
     * view keeps its exact height so nothing on screen jumps.
     */
    var wordTakeover: List<String>? = null
        set(value) {
            if (field != value) {
                field = value
                rebuildKeys()
                invalidate()
            }
        }

    private class KeyBox(val spec: KeySpec, val rect: RectF, val row: Int)

    private val keys = ArrayList<KeyBox>()
    private val pressed = HashMap<Int, KeyBox>() // pointer id -> key
    private var lastShiftTapAt = 0L
    private var repeatKey: KeyBox? = null
    private var repeatFired = false

    private val dp = resources.displayMetrics.density
    private val keyHeight = resources.getDimension(R.dimen.kb_key_height)
    private val rowGap = resources.getDimension(R.dimen.kb_row_gap)
    private val keyGap = resources.getDimension(R.dimen.kb_key_gap)
    private val sidePadding = resources.getDimension(R.dimen.kb_side_padding)
    private val bottomPadding = resources.getDimension(R.dimen.kb_bottom_padding)
    private val radius = resources.getDimension(R.dimen.kb_key_radius)
    private val labelSize = resources.getDimension(R.dimen.kb_key_text)
    private val labelSizeSmall = resources.getDimension(R.dimen.kb_key_text_small)

    private val colorBackground = ContextCompat.getColor(context, R.color.kb_background)
    private val colorKey = ContextCompat.getColor(context, R.color.kb_key)
    private val colorKeyPressed = ContextCompat.getColor(context, R.color.kb_key_pressed)
    private val colorKeySpecial = ContextCompat.getColor(context, R.color.kb_key_special)
    private val colorKeyAction = ContextCompat.getColor(context, R.color.kb_key_action)
    private val colorText = ContextCompat.getColor(context, R.color.kb_key_text)
    private val colorTextSpecial = ContextCompat.getColor(context, R.color.kb_key_text_special)
    private val colorTextAction = ContextCompat.getColor(context, R.color.kb_key_text_action)
    private val colorShadow = ContextCompat.getColor(context, R.color.kb_key_shadow)

    private val keyPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val shadowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = colorShadow }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
    }
    private val tmpRect = RectF()

    private val repeatRunnable = object : Runnable {
        override fun run() {
            val k = repeatKey ?: return
            repeatFired = true
            fire(k.spec)
            postDelayed(this, REPEAT_INTERVAL_MS)
        }
    }

    init {
        isClickable = true
        isFocusable = false
        // Ignore touches arriving through an overlay drawn on top of the keys (tapjacking).
        filterTouchesWhenObscured = true
        setBackgroundColor(colorBackground)
        rebuildKeys()
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val width = MeasureSpec.getSize(widthMeasureSpec)
        val rows = KeyboardLayouts.forLayer(layer).rows.size
        val height = (rows * keyHeight + (rows - 1) * rowGap + rowGap + bottomPadding).toInt()
        setMeasuredDimension(width, height)
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        rebuildKeys()
    }

    private fun takeoverLayout(words: List<String>): LayoutSpec {
        val escape = KeySpec("abc", KeyAction.EscapeWords, 1.5f, special = true)
        val backspace = KeySpec("⌫", KeyAction.Backspace, 1.5f, special = true)
        val enter = KeySpec("↵", KeyAction.Enter, 1.5f, special = true, accent = true)
        val row3 = if (words.size > 2) {
            RowSpec(listOf(KeySpec(words[2], KeyAction.Word(words[2]), 8.5f), backspace))
        } else {
            RowSpec(listOf(backspace), leadingPad = 8.5f)
        }
        return LayoutSpec(
            listOf(
                RowSpec(listOf(KeySpec(words[0], KeyAction.Word(words[0]), 10f))),
                if (words.size > 1) {
                    RowSpec(listOf(KeySpec(words[1], KeyAction.Word(words[1]), 10f)))
                } else {
                    RowSpec(emptyList())
                },
                row3,
                RowSpec(
                    listOf(
                        escape,
                        KeySpec(",", KeyAction.Text(","), 1f, special = true),
                        KeySpec("", KeyAction.Space, 5f),
                        KeySpec(".", KeyAction.Text("."), 1f, special = true),
                        enter,
                    ),
                ),
            ),
        )
    }

    private fun rebuildKeys() {
        keys.clear()
        val takeover = wordTakeover
        val spec = if (!takeover.isNullOrEmpty()) takeoverLayout(takeover) else KeyboardLayouts.forLayer(layer)
        val usable = width - 2 * sidePadding
        if (usable <= 0) return
        val unit = usable / 10f
        var y = rowGap
        spec.rows.forEachIndexed { rowIndex, row ->
            var x = sidePadding + row.leadingPad * unit
            for (key in row.keys) {
                val w = key.width * unit
                keys += KeyBox(key, RectF(x, y, x + w, y + keyHeight), rowIndex)
                x += w
            }
            y += keyHeight + rowGap
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val weights = if (adaptiveEnabled && layer == Layer.LETTERS) keyWeights else null
        for (box in keys) {
            val spec = box.spec
            var scale = 1f
            var alpha = 255
            if (weights != null && spec.isLetter) {
                val w = weights[(spec.action as KeyAction.Text).text[0].lowercaseChar()] ?: 0f
                if (w <= 0f) {
                    scale = 0.74f
                    alpha = 110
                } else {
                    scale = 0.9f + 0.1f * min(1f, w * 3f)
                }
            }
            val r = box.rect
            val halfGap = keyGap / 2f
            val cx = (r.left + r.right) / 2f
            val cy = (r.top + r.bottom) / 2f
            val hw = (r.width() / 2f - halfGap) * scale
            val hh = (r.height() / 2f) * scale
            tmpRect.set(cx - hw, cy - hh, cx + hw, cy + hh)

            val isPressed = pressed.values.any { it === box }
            val shiftActive = spec.action == KeyAction.Shift && shift != Shift.OFF
            keyPaint.color = when {
                isPressed -> colorKeyPressed
                spec.accent -> colorKeyAction
                shiftActive -> colorKeyPressed
                spec.special -> colorKeySpecial
                else -> colorKey
            }
            keyPaint.alpha = alpha
            shadowPaint.alpha = alpha / 4
            tmpRect.offset(0f, 1.5f * dp)
            canvas.drawRoundRect(tmpRect, radius, radius, shadowPaint)
            tmpRect.offset(0f, -1.5f * dp)
            canvas.drawRoundRect(tmpRect, radius, radius, keyPaint)

            val label = labelFor(spec)
            if (label.isNotEmpty()) {
                textPaint.color = when {
                    spec.accent -> colorTextAction
                    spec.special -> colorTextSpecial
                    else -> colorText
                }
                textPaint.alpha = alpha
                val isWordKey = spec.action is KeyAction.Word
                textPaint.textSize = when {
                    isWordKey -> labelSize * 0.95f
                    spec.isLetter || label.length == 1 -> labelSize * scale
                    else -> labelSizeSmall * scale
                }
                textPaint.isFakeBoldText = isWordKey || (spec.action == KeyAction.Shift && shift == Shift.LOCKED)
                val baseline = cy - (textPaint.descent() + textPaint.ascent()) / 2f
                canvas.drawText(label, cx, baseline, textPaint)
                if (spec.action == KeyAction.Shift && shift == Shift.LOCKED) {
                    canvas.drawRect(cx - 6 * dp, tmpRect.bottom - 7 * dp, cx + 6 * dp, tmpRect.bottom - 5 * dp, textPaint)
                }
            }
        }
    }

    private fun labelFor(spec: KeySpec): String {
        if (spec.action == KeyAction.Space) return ""
        if (spec.isLetter && shift != Shift.OFF) return spec.label.uppercase()
        return spec.label
    }

    private fun keyAt(x: Float, y: Float): KeyBox? {
        var best: KeyBox? = null
        var bestDist = Float.MAX_VALUE
        for (box in keys) {
            val r = box.rect
            if (r.contains(x, y)) return box
            // Fall back to the nearest key in the same row band so touches in gaps still land.
            if (y >= r.top - rowGap && y <= r.bottom + rowGap) {
                val dx = when {
                    x < r.left -> r.left - x
                    x > r.right -> x - r.right
                    else -> 0f
                }
                val dy = when {
                    y < r.top -> r.top - y
                    y > r.bottom -> y - r.bottom
                    else -> 0f
                }
                val d = dx * dx + dy * dy
                if (d < bestDist) {
                    bestDist = d
                    best = box
                }
            }
        }
        return best
    }

    /**
     * Adaptive hit resolution: when a touch lands on a letter that cannot continue the current
     * word but is close to the edge shared with a letter that can, prefer the possible one.
     */
    private fun resolveAdaptive(box: KeyBox, x: Float): KeyBox {
        val weights = keyWeights ?: return box
        if (!adaptiveEnabled || layer != Layer.LETTERS || !box.spec.isLetter) return box
        val own = weights[(box.spec.action as KeyAction.Text).text[0]] ?: 0f
        if (own > 0f) return box
        val r = box.rect
        val edgeZone = r.width() * 0.3f
        val neighbours = keys.filter { it.row == box.row && it.spec.isLetter && it !== box }
        val candidate = when {
            x - r.left < edgeZone -> neighbours.firstOrNull { it.rect.right <= r.left + 1f && it.rect.right >= r.left - keyGap - 1f }
            r.right - x < edgeZone -> neighbours.firstOrNull { it.rect.left >= r.right - 1f && it.rect.left <= r.right + keyGap + 1f }
            else -> null
        } ?: return box
        val w = weights[(candidate.spec.action as KeyAction.Text).text[0]] ?: 0f
        return if (w > 0.02f) candidate else box
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_POINTER_DOWN -> {
                val index = event.actionIndex
                val id = event.getPointerId(index)
                val box = keyAt(event.getX(index), event.getY(index)) ?: return true
                pressed[id] = box
                if (hapticsEnabled) performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
                if (box.spec.action == KeyAction.Backspace) {
                    repeatKey = box
                    repeatFired = false
                    postDelayed(repeatRunnable, REPEAT_DELAY_MS)
                }
                invalidate()
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_POINTER_UP -> {
                val index = event.actionIndex
                val id = event.getPointerId(index)
                val down = pressed.remove(id)
                if (down != null) {
                    if (down.spec.action == KeyAction.Backspace) {
                        removeCallbacks(repeatRunnable)
                        val fired = repeatFired
                        repeatKey = null
                        if (!fired) fire(down.spec)
                    } else {
                        // Release position wins if it is still on a key; otherwise the key pressed.
                        val up = keyAt(event.getX(index), event.getY(index)) ?: down
                        val chosen = if (up.spec.isLetter) resolveAdaptive(up, event.getX(index)) else up
                        fire(chosen.spec)
                    }
                }
                invalidate()
            }
            MotionEvent.ACTION_CANCEL -> {
                pressed.clear()
                removeCallbacks(repeatRunnable)
                repeatKey = null
                invalidate()
            }
        }
        return true
    }

    private fun fire(spec: KeySpec) {
        when (val action = spec.action) {
            is KeyAction.Text -> {
                val text = if (shift != Shift.OFF && spec.isLetter) action.text.uppercase() else action.text
                listener?.onText(text)
                if (shift == Shift.ON) shift = Shift.OFF
            }
            KeyAction.Backspace -> listener?.onBackspace()
            KeyAction.Enter -> listener?.onEnter()
            KeyAction.Space -> listener?.onSpace()
            KeyAction.Shift -> {
                val now = SystemClock.uptimeMillis()
                shift = when (shift) {
                    Shift.OFF -> Shift.ON
                    Shift.ON -> if (now - lastShiftTapAt < DOUBLE_TAP_MS) Shift.LOCKED else Shift.OFF
                    Shift.LOCKED -> Shift.OFF
                }
                lastShiftTapAt = now
            }
            is KeyAction.SwitchLayer -> {
                layer = action.layer
                if (layer != Layer.LETTERS) shift = Shift.OFF
                requestLayout()
            }
            is KeyAction.Word -> listener?.onWordKey(action.word)
            KeyAction.EscapeWords -> listener?.onEscapeWordMode()
        }
    }

    companion object {
        private const val REPEAT_DELAY_MS = 400L
        private const val REPEAT_INTERVAL_MS = 45L
        private const val DOUBLE_TAP_MS = 350L
    }
}
