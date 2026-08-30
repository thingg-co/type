// JNI bridge between com.aosmith.board.llm.LlamaNative and llama.cpp.
//
// Design notes
//  * One model, one context, one sequence. All calls are serialised with a mutex; the
//    Kotlin side additionally funnels everything through a single-thread dispatcher.
//  * The fixed part of the prompt (system prompt + few-shot examples + the opening of the
//    user turn) is decoded once by setPrefix() and kept in the KV cache. complete() rolls
//    the cache back to that point and only decodes the short per-request suffix, which is
//    what makes per-word correction fast enough for a keyboard.
//  * Text crosses the JNI boundary as UTF-8 byte arrays so partial code points produced
//    by byte-level tokenizers can never crash NewStringUTF.

#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"

#define TAG "BoardLLM"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

std::mutex g_mutex;
bool g_backend_ready = false;

llama_model *        g_model = nullptr;
llama_context *      g_ctx   = nullptr;
const llama_vocab *  g_vocab = nullptr;
llama_batch          g_batch = {};
int                  g_n_batch = 0;
int                  g_n_ctx   = 0;

std::vector<llama_token> g_prefix_tokens;
bool                     g_prefix_in_cache = false;

std::atomic<bool> g_cancel{false};

void log_callback(ggml_log_level level, const char * text, void *) {
    int prio;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: prio = ANDROID_LOG_ERROR; break;
        case GGML_LOG_LEVEL_WARN:  prio = ANDROID_LOG_WARN;  break;
        case GGML_LOG_LEVEL_INFO:
        case GGML_LOG_LEVEL_CONT:  prio = ANDROID_LOG_INFO;  break;
        default:                   return; // drop debug spam
    }
    __android_log_write(prio, "llama.cpp", text);
}

std::string jstring_to_utf8(JNIEnv * env, jstring s) {
    if (s == nullptr) return {};
    const char * chars = env->GetStringUTFChars(s, nullptr);
    std::string out = chars ? chars : "";
    if (chars) env->ReleaseStringUTFChars(s, chars);
    return out;
}

jbyteArray utf8_to_jbytes(JNIEnv * env, const std::string & s) {
    jbyteArray arr = env->NewByteArray((jsize) s.size());
    if (arr != nullptr && !s.empty()) {
        env->SetByteArrayRegion(arr, 0, (jsize) s.size(), reinterpret_cast<const jbyte *>(s.data()));
    }
    return arr;
}

std::vector<llama_token> tokenize(const std::string & text, bool add_special, bool parse_special) {
    std::vector<llama_token> out;
    if (text.empty()) return out;
    int n = llama_tokenize(g_vocab, text.c_str(), (int) text.size(), nullptr, 0, add_special, parse_special);
    if (n < 0) n = -n;
    if (n == 0) return out;
    out.resize(n);
    int m = llama_tokenize(g_vocab, text.c_str(), (int) text.size(), out.data(), n, add_special, parse_special);
    if (m < 0) { out.clear(); return out; }
    out.resize(m);
    return out;
}

// Feed `n` tokens starting at absolute position `pos0`. Logits are only requested for the
// final token, and only when the caller is about to sample.
bool decode_tokens(const llama_token * toks, int n, int pos0, bool want_last_logits) {
    for (int i = 0; i < n; i += g_n_batch) {
        const int cnt = std::min(g_n_batch, n - i);
        g_batch.n_tokens = cnt;
        for (int j = 0; j < cnt; j++) {
            g_batch.token[j]     = toks[i + j];
            g_batch.pos[j]       = pos0 + i + j;
            g_batch.n_seq_id[j]  = 1;
            g_batch.seq_id[j][0] = 0;
            g_batch.logits[j]    = 0;
        }
        const bool last_chunk = (i + cnt == n);
        if (last_chunk && want_last_logits) g_batch.logits[cnt - 1] = 1;
        const int rc = llama_decode(g_ctx, g_batch);
        if (rc != 0) {
            LOGE("llama_decode failed with %d (pos0=%d, n=%d)", rc, pos0, n);
            return false;
        }
        if (g_cancel.load()) return false;
    }
    return true;
}

void free_all_locked() {
    if (g_batch.token != nullptr) { llama_batch_free(g_batch); g_batch = {}; }
    if (g_ctx)   { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    g_vocab = nullptr;
    g_prefix_tokens.clear();
    g_prefix_in_cache = false;
}

} // namespace

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_aosmith_board_llm_LlamaNative_load(JNIEnv * env, jobject, jstring jpath, jint n_threads, jint n_ctx) {
    std::lock_guard<std::mutex> lock(g_mutex);
    free_all_locked();

    if (!g_backend_ready) {
        llama_log_set(log_callback, nullptr);
        llama_backend_init();
        g_backend_ready = true;
    }

    const std::string path = jstring_to_utf8(env, jpath);
    LOGI("loading %s (threads=%d, n_ctx=%d)", path.c_str(), (int) n_threads, (int) n_ctx);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    g_model = llama_model_load_from_file(path.c_str(), mparams);
    if (g_model == nullptr) {
        LOGE("llama_model_load_from_file failed");
        return JNI_FALSE;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx          = (uint32_t) std::max(256, (int) n_ctx);
    cparams.n_batch        = 256;
    cparams.n_ubatch       = 256;
    cparams.n_threads      = std::max(1, (int) n_threads);
    cparams.n_threads_batch = std::max(1, (int) n_threads);
    cparams.no_perf        = true;
    g_ctx = llama_init_from_model(g_model, cparams);
    if (g_ctx == nullptr) {
        LOGE("llama_init_from_model failed");
        free_all_locked();
        return JNI_FALSE;
    }

    g_vocab   = llama_model_get_vocab(g_model);
    g_n_ctx   = (int) llama_n_ctx(g_ctx);
    g_n_batch = (int) llama_n_batch(g_ctx);
    g_batch   = llama_batch_init(g_n_batch, 0, 1);

    char desc[256] = {0};
    llama_model_desc(g_model, desc, sizeof(desc));
    LOGI("loaded: %s, %.1f MB, n_ctx=%d, n_batch=%d", desc, llama_model_size(g_model) / 1e6, g_n_ctx, g_n_batch);
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_aosmith_board_llm_LlamaNative_free(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    free_all_locked();
}

JNIEXPORT jboolean JNICALL
Java_com_aosmith_board_llm_LlamaNative_isLoaded(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_ctx != nullptr ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jbyteArray JNICALL
Java_com_aosmith_board_llm_LlamaNative_modelInfo(JNIEnv * env, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_model == nullptr) return nullptr;
    char desc[256] = {0};
    llama_model_desc(g_model, desc, sizeof(desc));
    std::string info = std::string(desc) + " | " + std::to_string(llama_model_size(g_model) / 1000000) + " MB | n_ctx " + std::to_string(g_n_ctx);
    return utf8_to_jbytes(env, info);
}

// Renders a chat through the model's own template. roles/contents are parallel arrays.
// Returns null when the model ships no template (the Kotlin side then falls back to ChatML).
JNIEXPORT jbyteArray JNICALL
Java_com_aosmith_board_llm_LlamaNative_formatChat(JNIEnv * env, jobject, jobjectArray roles, jobjectArray contents, jboolean add_assistant) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_model == nullptr) return nullptr;
    const char * tmpl = llama_model_chat_template(g_model, nullptr);
    if (tmpl == nullptr) return nullptr;

    const jsize n = env->GetArrayLength(roles);
    if (n != env->GetArrayLength(contents)) return nullptr;
    std::vector<std::string> role_str(n), content_str(n);
    std::vector<llama_chat_message> msgs(n);
    for (jsize i = 0; i < n; i++) {
        auto r = (jstring) env->GetObjectArrayElement(roles, i);
        auto c = (jstring) env->GetObjectArrayElement(contents, i);
        role_str[i]    = jstring_to_utf8(env, r);
        content_str[i] = jstring_to_utf8(env, c);
        env->DeleteLocalRef(r);
        env->DeleteLocalRef(c);
        msgs[i] = { role_str[i].c_str(), content_str[i].c_str() };
    }
    int need = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), add_assistant, nullptr, 0);
    if (need < 0) {
        LOGW("chat template not supported by llama_chat_apply_template; falling back");
        return nullptr;
    }
    std::string buf((size_t) need + 1, '\0');
    int written = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), add_assistant, buf.data(), (int) buf.size());
    if (written < 0) return nullptr;
    buf.resize((size_t) written);
    return utf8_to_jbytes(env, buf);
}

JNIEXPORT jint JNICALL
Java_com_aosmith_board_llm_LlamaNative_setPrefix(JNIEnv * env, jobject, jstring jprefix) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_ctx == nullptr) return -1;
    g_cancel.store(false);

    const std::string prefix = jstring_to_utf8(env, jprefix);
    std::vector<llama_token> toks = tokenize(prefix, /*add_special=*/true, /*parse_special=*/true);
    if (toks.empty() || (int) toks.size() > g_n_ctx - 96) {
        LOGE("prefix has %zu tokens; does not fit n_ctx=%d", toks.size(), g_n_ctx);
        return -1;
    }
    llama_memory_clear(llama_get_memory(g_ctx), true);
    g_prefix_in_cache = false;
    if (!decode_tokens(toks.data(), (int) toks.size(), 0, false)) return -1;
    g_prefix_tokens   = std::move(toks);
    g_prefix_in_cache = true;
    LOGI("prefix cached: %zu tokens", g_prefix_tokens.size());
    return (jint) g_prefix_tokens.size();
}

JNIEXPORT jint JNICALL
Java_com_aosmith_board_llm_LlamaNative_tokenCount(JNIEnv * env, jobject, jstring jtext) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_ctx == nullptr) return -1;
    return (jint) tokenize(jstring_to_utf8(env, jtext), false, true).size();
}

// Greedy completion of `suffix` appended to the cached prefix. `grammar` is an optional
// GBNF string constraining the output; `stop_at_newline` truncates at the first '\n'.
JNIEXPORT jbyteArray JNICALL
Java_com_aosmith_board_llm_LlamaNative_complete(JNIEnv * env, jobject, jstring jsuffix, jint max_tokens, jstring jgrammar, jboolean stop_at_newline) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_ctx == nullptr) return nullptr;
    g_cancel.store(false);

    const std::string suffix = jstring_to_utf8(env, jsuffix);
    llama_memory_t mem = llama_get_memory(g_ctx);

    int pos0 = 0;
    if (g_prefix_in_cache) {
        pos0 = (int) g_prefix_tokens.size();
        llama_memory_seq_rm(mem, 0, pos0, -1);
    } else {
        llama_memory_clear(mem, true);
    }
    std::vector<llama_token> toks = tokenize(suffix, /*add_special=*/!g_prefix_in_cache, /*parse_special=*/true);
    if (toks.empty()) return nullptr;
    if (pos0 + (int) toks.size() + (int) max_tokens + 1 > g_n_ctx) {
        LOGE("request too long: prefix=%d suffix=%zu max=%d n_ctx=%d", pos0, toks.size(), (int) max_tokens, g_n_ctx);
        return nullptr;
    }
    const int64_t t0 = llama_time_us();
    if (!decode_tokens(toks.data(), (int) toks.size(), pos0, true)) return nullptr;
    int cur_pos = pos0 + (int) toks.size();

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler * chain = llama_sampler_chain_init(sparams);
    const std::string grammar = jstring_to_utf8(env, jgrammar);
    if (!grammar.empty()) {
        llama_sampler * g = llama_sampler_init_grammar(g_vocab, grammar.c_str(), "root");
        if (g != nullptr) llama_sampler_chain_add(chain, g);
        else LOGE("grammar failed to parse; sampling unconstrained");
    }
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());

    std::string out;
    int generated = 0;
    for (int i = 0; i < (int) max_tokens; i++) {
        const llama_token tok = llama_sampler_sample(chain, g_ctx, -1);
        generated++;
        if (llama_vocab_is_eog(g_vocab, tok)) break;

        char buf[512];
        const int n = llama_token_to_piece(g_vocab, tok, buf, sizeof(buf), 0, false);
        if (n < 0) break;
        std::string piece(buf, (size_t) n);
        if (stop_at_newline) {
            const size_t nl = piece.find('\n');
            if (nl != std::string::npos) { out += piece.substr(0, nl); break; }
        }
        out += piece;
        if (g_cancel.load()) break;
        if (i + 1 < (int) max_tokens) {
            if (!decode_tokens(&tok, 1, cur_pos, true)) break;
            cur_pos++;
        }
    }
    llama_sampler_free(chain);
    const double ms = (llama_time_us() - t0) / 1000.0;
    LOGI("complete: %zu prompt tok, %d gen tok, %.0f ms -> \"%s\"", toks.size(), generated, ms, out.c_str());
    return utf8_to_jbytes(env, out);
}

JNIEXPORT void JNICALL
Java_com_aosmith_board_llm_LlamaNative_cancel(JNIEnv *, jobject) {
    g_cancel.store(true);
}

} // extern "C"
