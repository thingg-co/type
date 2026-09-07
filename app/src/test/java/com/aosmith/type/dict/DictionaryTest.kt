package com.aosmith.type.dict

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DictionaryTest {

    private val dict = Dictionary(TestWords.LIST.asSequence())

    @Test fun `known words and simple variants`() {
        assertTrue(dict.isKnown("typing"))
        assertTrue(dict.isKnown("Typing"))
        assertTrue(dict.isKnown("cats"))          // listed
        assertTrue(dict.isKnown("experts"))       // listed
        assertTrue(dict.isKnown("restaurant's"))  // possessive of a listed word
        assertFalse(dict.isKnown("experiwnce"))
        assertFalse(dict.isKnown(""))
    }

    @Test fun `slip predictions complete a prefix typed with one slip`() {
        val d = Dictionary(sequenceOf("the", "better", "beyond", "betting", "bet", "verge", "begin", "beta"))
        assertEquals(listOf("better"), d.slipPredictions("betye", 2))     // y sits next to t; "betti" is a second slip away
        assertEquals("beyond", d.slipPredictions("beyom", 2).first())              // m sits next to n
        assertEquals("better", d.slipPredictions("bteter", 2).first())             // a transposition
        assertEquals("Better", d.slipPredictions("Betye", 1).first())              // casing mirrored
        assertTrue(d.slipPredictions("be", 2).isEmpty())                            // too short to judge
        assertTrue(d.slipPredictions("zzzzz", 2).isEmpty())                         // nothing within one slip
    }

    @Test fun `slip predictions rank an exact prefix first and a far word not at all`() {
        val d = Dictionary(sequenceOf("bet", "better", "betting", "belt"))   // most frequent first, as the real list
        assertEquals(listOf("bet", "better", "betting"), d.slipPredictions("bet", 3))
        assertTrue("belt" !in d.slipPredictions("betting", 3))
    }

    @Test fun `short slip fix substitutes one adjacent key into a common word`() {
        val d = Dictionary(sequenceOf("so", "on", "sky", "ask"))
        assertEquals("so", d.shortSlipFix("sk"))     // k sits under o
        assertEquals("So", d.shortSlipFix("Sk"))     // casing mirrored
        assertNull(d.shortSlipFix("qz"))             // no common neighbour substitution
        assertNull(d.shortSlipFix("s"))              // too short to judge
    }

    @Test fun `hasPrefix follows the trie`() {
        assertTrue(dict.hasPrefix("typi"))
        assertTrue(dict.hasPrefix("restau"))
        assertFalse(dict.hasPrefix("experiw"))
    }

    @Test fun `completions collapse families and stay empty when too many`() {
        assertEquals(listOf("restaurant", "restaurateur"), dict.completions("restaur", 3))
        assertTrue(dict.completions("t", 3).isEmpty())
        assertTrue(dict.completions("zzz", 3).isEmpty())
    }

    @Test fun `completions match the typed case`() {
        assertEquals(listOf("Restaurant", "Restaurateur"), dict.completions("Restaur", 3))
    }

    @Test fun `predictions rank by frequency`() {
        assertEquals(listOf("typing", "typical", "typically"), dict.predictions("typi", 3))
        assertTrue(dict.predictions("experiw", 3).isEmpty())
    }

    @Test fun `suggest finds close known words`() {
        assertEquals("weird", dict.suggest("wierd", 1).first())
        assertEquals("experience", dict.suggest("experiwnce", 1).first())
        assertTrue(dict.suggest("qqqqq", 3).isEmpty())
    }

    @Test fun `next letters are weighted and complete`() {
        val next = dict.nextLetters("th")
        assertEquals(setOf('e'), next.keys) // the, they, them
        assertEquals(1f, next.values.sum(), 1e-4f)
        assertTrue(dict.nextLetters("experiw").isEmpty())
    }

    @Test fun `edit distance covers the typo classes`() {
        assertEquals(0, Dictionary.editDistance("same", "same"))
        assertEquals(1, Dictionary.editDistance("teh", "the"))          // transposition
        assertEquals(1, Dictionary.editDistance("resturant", "restaurant")) // insertion
        assertEquals(1, Dictionary.editDistance("morninf", "morning"))  // substitution
        assertEquals(3, Dictionary.editDistance("kitten", "sitting"))
    }

    @Test fun `match case mirrors the original`() {
        assertEquals("Restaurant", Dictionary.matchCase("Resturant", "restaurant"))
        assertEquals("HELLO", Dictionary.matchCase("HELO", "hello"))
        assertEquals("hello", Dictionary.matchCase("helo", "Hello"))
    }

    @Test fun `slip candidates flag rare neighbour variants of common words`() {
        // Common words up front, filler to push the rare entries past SLIP_MIN_RANK, then
        // the real-word slips the 64k list actually contains ("nake", "mot", "cam").
        val words = listOf("the", "make", "not", "can", "hate", "night", "might") +
            (0 until Dictionary.SLIP_MIN_RANK + 100).map { "zz${it}x" } +
            listOf("nake", "mot", "cam", "jate")
        val d = Dictionary(words.asSequence())
        assertEquals(listOf("make"), d.slipCandidates("nake"))
        assertEquals(listOf("not"), d.slipCandidates("mot"))
        assertEquals(listOf("can"), d.slipCandidates("cam"))
        assertEquals(listOf("make"), d.slipCandidates("Nake"))   // case-insensitive
        assertTrue(d.slipCandidates("might").isEmpty())          // common words stay trusted
        assertTrue(d.slipCandidates("make").isEmpty())
        assertTrue(d.slipCandidates("zz2050x").isEmpty())        // rare but no common variant
    }

    @Test fun `slip candidates demand a wide frequency gap`() {
        val words = listOf("the") +
            (0 until Dictionary.SLIP_MIN_RANK + 100).map { "zz${it}x" } +
            listOf("hate", "jate") // j->h is adjacent, but "hate" is barely more common here
        val d = Dictionary(words.asSequence())
        assertTrue(d.slipCandidates("jate").isEmpty())
    }

    @Test fun `stem folds plural and possessive`() {
        assertEquals("restaurant", Dictionary.stem("restaurants"))
        assertEquals("restaurant", Dictionary.stem("restaurant's"))
        assertEquals("cat", Dictionary.stem("cats"))
        assertEquals("was", Dictionary.stem("was")) // short words keep their s
    }
}

object TestWords {
    val LIST = listOf(
        "the", "to", "and", "they", "them", "we", "can", "how", "was",
        "type", "types", "typing", "typical", "typically", "typist",
        "improve", "experience", "experiment", "expert", "experts",
        "restaurant", "restaurants", "restaurant's", "restaurateur",
        "weird", "believe", "friend", "cat", "cats",
    )
}
