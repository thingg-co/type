package com.aosmith.board.ime

enum class Layer { LETTERS, SYMBOLS, SYMBOLS2 }

sealed class KeyAction {
    data class Text(val text: String) : KeyAction()
    object Backspace : KeyAction()
    object Enter : KeyAction()
    object Shift : KeyAction()
    object Space : KeyAction()
    data class SwitchLayer(val layer: Layer) : KeyAction()
}

/** [width] is in key units: one unit is a tenth of the usable keyboard width. */
class KeySpec(
    val label: String,
    val action: KeyAction,
    val width: Float = 1f,
    val special: Boolean = false,
    val accent: Boolean = false,
) {
    val isLetter: Boolean get() = action is KeyAction.Text && action.text.length == 1 && action.text[0].isLetter()
}

class RowSpec(val keys: List<KeySpec>, val leadingPad: Float = 0f, val trailingPad: Float = 0f)

class LayoutSpec(val rows: List<RowSpec>)

/**
 * Layouts are plain data so another language (Thai is the next one planned) only needs a new
 * function here plus its own dictionary asset.
 */
object KeyboardLayouts {

    private fun chars(s: String): List<KeySpec> = s.map { KeySpec(it.toString(), KeyAction.Text(it.toString())) }

    private val shift = KeySpec("⇧", KeyAction.Shift, 1.5f, special = true)
    private val backspace = KeySpec("⌫", KeyAction.Backspace, 1.5f, special = true)
    private val enter = KeySpec("↵", KeyAction.Enter, 1.5f, special = true, accent = true)
    private val space = KeySpec("", KeyAction.Space, 5f)
    private val comma = KeySpec(",", KeyAction.Text(","), 1f, special = true)
    private val period = KeySpec(".", KeyAction.Text("."), 1f, special = true)
    private val toSymbols = KeySpec("?123", KeyAction.SwitchLayer(Layer.SYMBOLS), 1.5f, special = true)
    private val toSymbols2 = KeySpec("#+=", KeyAction.SwitchLayer(Layer.SYMBOLS2), 1.5f, special = true)
    private val toLetters = KeySpec("ABC", KeyAction.SwitchLayer(Layer.LETTERS), 1.5f, special = true)

    val letters = LayoutSpec(
        listOf(
            RowSpec(chars("qwertyuiop")),
            RowSpec(chars("asdfghjkl"), leadingPad = 0.5f, trailingPad = 0.5f),
            RowSpec(listOf(shift) + chars("zxcvbnm") + listOf(backspace)),
            RowSpec(listOf(toSymbols, comma, space, period, enter)),
        ),
    )

    val symbols = LayoutSpec(
        listOf(
            RowSpec(chars("1234567890")),
            RowSpec(chars("@#\$_&-+()/")),
            RowSpec(listOf(toSymbols2) + chars("*\"':;!?") + listOf(backspace)),
            RowSpec(listOf(toLetters, comma, space, period, enter)),
        ),
    )

    val symbols2 = LayoutSpec(
        listOf(
            RowSpec(chars("~`|•√π÷×¶∆")),
            RowSpec(chars("£¢€¥^°={}\\")),
            RowSpec(listOf(KeySpec("?123", KeyAction.SwitchLayer(Layer.SYMBOLS), 1.5f, special = true)) + chars("%©®™✓[]") + listOf(backspace)),
            RowSpec(listOf(toLetters, comma, space, period, enter)),
        ),
    )

    fun forLayer(layer: Layer): LayoutSpec = when (layer) {
        Layer.LETTERS -> letters
        Layer.SYMBOLS -> symbols
        Layer.SYMBOLS2 -> symbols2
    }
}
