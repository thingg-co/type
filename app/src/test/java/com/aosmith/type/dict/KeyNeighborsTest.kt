package com.aosmith.type.dict

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class KeyNeighborsTest {

    @Test fun `adjacency follows the physical layout`() {
        assertTrue(KeyNeighbors.adjacent('e', 'r'))   // same row
        assertTrue(KeyNeighbors.adjacent('a', 's'))
        assertTrue(KeyNeighbors.adjacent('s', 'w'))   // across the stagger
        assertTrue(KeyNeighbors.adjacent('f', 'v'))
        assertTrue(KeyNeighbors.adjacent('G', 'h'))   // case-insensitive
        assertFalse(KeyNeighbors.adjacent('q', 'p'))
        assertFalse(KeyNeighbors.adjacent('a', 'l'))
        assertFalse(KeyNeighbors.adjacent('z', 'm'))
        assertFalse(KeyNeighbors.adjacent('e', 'e'))  // same key is not a slip
    }

    @Test fun `neighbour lists cover the cross-row slips`() {
        assertTrue('k' in KeyNeighbors.neighbors('i'))
        assertTrue('i' in KeyNeighbors.neighbors('k'))
        assertTrue('m' in KeyNeighbors.neighbors('n'))
        assertTrue('m' in KeyNeighbors.neighbors('k'))
        assertFalse('p' in KeyNeighbors.neighbors('q'))
        assertTrue(KeyNeighbors.neighbors('\'').isEmpty())
    }

    @Test fun `weighted distance makes neighbour slips cheap`() {
        val dict = Dictionary(TestWords.LIST.asSequence())
        // "ststion" -> "station": a->s is one adjacent slip, well under a full edit.
        assertTrue(dict.weightedDistance("ststion", "station") < 0.5f)
        // distant substitution costs a full edit
        assertEquals(1f, dict.weightedDistance("cat", "cap"), 1e-4f) // t->p not adjacent
        // transpositions stay cheaper than two edits
        assertEquals(KeyNeighbors.TRANSPOSE_COST, dict.weightedDistance("teh", "the"), 1e-4f)
    }

    @Test fun `suggestions prefer the fat-finger explanation`() {
        val dict = Dictionary(
            (TestWords.LIST + listOf("station", "startion", "cut", "cap")).asSequence(),
        )
        assertEquals("station", dict.suggest("ststion", 1).first())
        // "cst": s is adjacent to a (cat) but not to u (cut); "cat" must win even though
        // both are one substitution away.
        assertEquals("cat", dict.suggest("cst", 2).first())
    }
}
