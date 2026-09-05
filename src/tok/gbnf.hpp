#pragma once

#include "../core/types.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mvllm {

// Byte-level GBNF subset (colibri grammar-draft.md):
//   root ::= ...
//   literals "..." with \" \\ \n \r \t \xHH
//   classes [...] / [^...], rule refs, groups, ? * +, |, # comments
// Root rule must be named `root`.
class Gbnf {
public:
    Status compile(const std::string &src, std::string &err);
    void reset();
    bool ready() const { return ready_; }
    // Advance the PDA by one UTF-8 byte. Returns false on reject (walker dies;
    // reset() to re-arm).
    bool accept_byte(unsigned char b);
    bool accept_bytes(const std::string &s);
    // ok[i] = 1 iff token i (decoded via decode) is a legal next token.
    // decode(id) must return the token's UTF-8 text; empty tokens are never allowed.
    void allow_mask(const std::function<std::string(int)> &decode, uint8_t *ok, int vocab) const;

private:
    bool ready_ = false;
};

} // namespace mvllm
