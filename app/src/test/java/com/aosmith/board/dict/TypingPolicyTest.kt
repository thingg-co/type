package com.aosmith.board.dict

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The regression that motivated this suite: typing "typing" must never surface "type" as a
 * correction mid-word. Mid-word is prediction territory; the model is only consulted once
 * the prefix cannot start any dictionary word.
 */
class TypingPolicyTest {

    private val dict = Dictionary(TestWords.LIST.asSequence())

    @Test fun `mid-word prefix of a real word predicts, never corrects`() {
        val action = TypingPolicy.midWord(dict, "typi")
        assertTrue("expected Predictions, got $action", action is MidWordAction.Predictions)
        assertEquals("typing", (action as MidWordAction.Predictions).words.first())
    }

    @Test fun `few families become word keys`() {
        val action = TypingPolicy.midWord(dict, "restaur")
        assertTrue(action is MidWordAction.WordKeys)
        assertEquals(listOf("restaurant", "restaurateur"), (action as MidWordAction.WordKeys).words)
    }

    @Test fun `two-letter prefixes with few families also morph`() {
        val action = TypingPolicy.midWord(dict, "ca")
        assertTrue(action is MidWordAction.WordKeys)
        assertEquals(listOf("can", "cat"), (action as MidWordAction.WordKeys).words)
    }

    @Test fun `impossible prefix is a typo and may ask the model`() {
        val action = TypingPolicy.midWord(dict, "experiwnce")
        assertTrue(action is MidWordAction.Typo)
        val typo = action as MidWordAction.Typo
        assertTrue(typo.askModel)
        assertEquals("experience", typo.dictSuggestions.first())
    }

    @Test fun `very short impossible prefixes stay quiet about the model`() {
        val action = TypingPolicy.midWord(dict, "xq")
        assertTrue(action is MidWordAction.Typo)
        assertTrue(!(action as MidWordAction.Typo).askModel)
    }

    @Test fun `complete words with nothing to add stay silent`() {
        assertEquals(MidWordAction.None, TypingPolicy.midWord(dict, "typing"))
        assertEquals(MidWordAction.None, TypingPolicy.midWord(dict, "t"))
        assertEquals(MidWordAction.None, TypingPolicy.midWord(dict, ""))
    }
}
