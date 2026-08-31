package com.aosmith.type.llm

/** Thin JNI surface over llama.cpp. See app/src/main/cpp/llm_jni.cpp for the semantics. */
object LlamaNative {
    init {
        System.loadLibrary("type_llm")
    }

    external fun load(modelPath: String, nThreads: Int, nCtx: Int): Boolean
    external fun free()
    external fun isLoaded(): Boolean
    external fun modelInfo(): ByteArray?
    external fun formatChat(roles: Array<String>, contents: Array<String>, addAssistant: Boolean): ByteArray?
    external fun setPrefix(prefix: String): Int
    external fun tokenCount(text: String): Int
    external fun complete(suffix: String, maxTokens: Int, grammar: String?, stopAtNewline: Boolean): ByteArray?
    external fun cancel()

    // Next-word network (asset copied to a file; see dict/NeuralLm.kt)
    external fun nnLoad(path: String): Boolean
    external fun nnTopK(hidden: ByteArray, hiddenScale: Float, k: Int): IntArray?
    external fun nnBenchMs(iters: Int): Float
    external fun nnFree()
}
