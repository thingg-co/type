package com.aosmith.board.dict

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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
