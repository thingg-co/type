package com.aosmith.type.dict

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class LexerTest {

    @Test fun `finds the previous word across spaces`() {
        assertEquals("should", Lexer.previousWord("Typing should "))
        assertEquals("should", Lexer.previousWord("Typing should"))
        assertEquals("we", Lexer.previousWord("how can we  "))
    }

    @Test fun `lowercases and keeps contractions together`() {
        assertEquals("don't", Lexer.previousWord("I Don't "))
    }

    @Test fun `sentence boundaries clear the context`() {
        assertNull(Lexer.previousWord("See you tomorrow. "))
        assertNull(Lexer.previousWord("Really? "))
        assertNull(Lexer.previousWord("Stop! "))
        assertNull(Lexer.previousWord("line one\n"))
    }

    @Test fun `numbers and punctuation give no context`() {
        assertNull(Lexer.previousWord("meet at 3 "))
        assertNull(Lexer.previousWord("a, "))
        assertNull(Lexer.previousWord(""))
        assertNull(Lexer.previousWord("   "))
    }
}
