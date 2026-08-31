package com.aosmith.type.dict

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
        val action = TypingPolicy.midWord(dict, null, null, "typi")
        assertTrue("expected Predictions, got $action", action is MidWordAction.Predictions)
        assertEquals("typing", (action as MidWordAction.Predictions).words.first())
    }

    @Test fun `few families become word keys`() {
        val action = TypingPolicy.midWord(dict, null, null, "restaur")
        assertTrue(action is MidWordAction.WordKeys)
        assertEquals(listOf("restaurant", "restaurateur"), (action as MidWordAction.WordKeys).words)
    }

    @Test fun `two-letter prefixes with few families also morph`() {
        val action = TypingPolicy.midWord(dict, null, null, "ca")
        assertTrue(action is MidWordAction.WordKeys)
        assertEquals(listOf("can", "cat"), (action as MidWordAction.WordKeys).words)
    }

    @Test fun `impossible prefix is a typo and may ask the model`() {
        val action = TypingPolicy.midWord(dict, null, null, "experiwnce")
        assertTrue(action is MidWordAction.Typo)
        val typo = action as MidWordAction.Typo
        assertTrue(typo.askModel)
        assertEquals("experience", typo.dictSuggestions.first())
    }

    @Test fun `very short impossible prefixes stay quiet about the model`() {
        val action = TypingPolicy.midWord(dict, null, null, "xq")
        assertTrue(action is MidWordAction.Typo)
        assertTrue(!(action as MidWordAction.Typo).askModel)
    }

    @Test fun `complete words with nothing to add stay silent`() {
        assertEquals(MidWordAction.None, TypingPolicy.midWord(dict, null, null, "typing"))
        assertEquals(MidWordAction.None, TypingPolicy.midWord(dict, null, null, "t"))
        assertEquals(MidWordAction.None, TypingPolicy.midWord(dict, null, null, ""))
    }

    // ---- context via bigrams ---------------------------------------------------------

    private val bigrams = TestBigrams.build(
        dict,
        // "they can" outranks "they cat"; after "the" the word "cat" beats "can".
        Triple("they", "can", 200),
        Triple("the", "cat", 200),
        Triple("the", "experiment", 150),
        Triple("the", "experience", 90),
        Triple("we", "can", 220),
        Triple("can", "type", 120),
        Triple("can", "typist", 250),
    )

    @Test fun `previous token reranks word keys`() {
        val neutral = TypingPolicy.midWord(dict, bigrams, null, "ca")
        assertEquals(listOf("can", "cat"), (neutral as MidWordAction.WordKeys).words)
        val afterThe = TypingPolicy.midWord(dict, bigrams, "the", "ca")
        assertEquals(listOf("cat", "can"), (afterThe as MidWordAction.WordKeys).words)
    }

    @Test fun `previous token reranks predictions`() {
        // "typi" spans four word families, so it lands in the Predictions branch.
        val afterCan = TypingPolicy.midWord(dict, bigrams, "can", "typi")
        assertTrue("expected Predictions, got $afterCan", afterCan is MidWordAction.Predictions)
        assertEquals("typist", (afterCan as MidWordAction.Predictions).words.first())
    }

    @Test fun `previous token reranks completions in word keys`() {
        // Unigram order puts experience first; "the experiment" is the stronger pair.
        val afterThe = TypingPolicy.midWord(dict, bigrams, "the", "exper")
        assertTrue("expected WordKeys, got $afterThe", afterThe is MidWordAction.WordKeys)
        assertEquals(listOf("experiment", "experience"), (afterThe as MidWordAction.WordKeys).words.take(2))
    }

    @Test fun `empty prefix suggests next words from context`() {
        val action = TypingPolicy.midWord(dict, bigrams, "we", "")
        assertTrue(action is MidWordAction.NextWords)
        assertEquals("can", (action as MidWordAction.NextWords).words.first())
    }

    @Test fun `empty prefix without context stays silent`() {
        assertEquals(MidWordAction.None, TypingPolicy.midWord(dict, bigrams, null, ""))
        assertEquals(MidWordAction.None, TypingPolicy.midWord(dict, bigrams, "qzqzqz", ""))
    }

    @Test fun `unknown previous token falls back to frequency order`() {
        val action = TypingPolicy.midWord(dict, bigrams, "borogove", "ca")
        assertEquals(listOf("can", "cat"), (action as MidWordAction.WordKeys).words)
    }

    @Test fun `two-letter predictions appear only when context backs them`() {
        // "ty" spans too many families for word keys; without context it stays quiet.
        assertEquals(MidWordAction.None, TypingPolicy.midWord(dict, null, null, "ty"))
        assertEquals(MidWordAction.None, TypingPolicy.midWord(dict, bigrams, "the", "ty"))
        val afterCan = TypingPolicy.midWord(dict, bigrams, "can", "ty")
        assertTrue("expected Predictions, got $afterCan", afterCan is MidWordAction.Predictions)
        assertEquals("typist", (afterCan as MidWordAction.Predictions).words.first())
    }
}
