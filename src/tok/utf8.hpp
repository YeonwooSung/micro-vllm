#pragma once

#include <cstdint>
#include <string>

namespace mvllm {

// One UTF-8 sequence at s[i]. Invalid/truncated lead is a single byte.
// Returns bytes consumed, or 0 if s/cp is null or i is out of range (*cp = 0).
int utf8_next(const unsigned char *s, int len, int i, uint32_t *cp);

// Write UTF-8 for cp into o (1-4 bytes). Returns bytes written, or 0 if o is null.
int utf8_put(char *o, uint32_t cp);

// Decode every codepoint in s[0,len) into out (at most cap). Returns count written.
int utf8_decode_all(const unsigned char *s, int len, uint32_t *out, int cap);

// Encode one codepoint to a UTF-8 string.
std::string utf8_encode_cp(uint32_t cp);

// Unicode letter / number / space from official generated tok_unicode.h.
bool uni_is_L(uint32_t cp);
bool uni_is_N(uint32_t cp);
bool uni_is_S(uint32_t cp);
bool uni_is_Lu(uint32_t cp); // uppercase letter
bool uni_is_Ll(uint32_t cp); // lowercase letter
bool uni_is_M(uint32_t cp);  // Unicode Mark (Mn/Mc/Me)
bool uni_is_P(uint32_t cp); // Unicode Punctuation (P*)
uint32_t uni_to_lower(uint32_t cp); // simple case fold; identity if none
uint32_t uni_to_upper(uint32_t cp); // simple case fold; identity if none

} // namespace mvllm
