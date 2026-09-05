#include "h3_reuse.hpp"

#include <algorithm>
#include <cstdlib>

namespace mvllm {
namespace {

constexpr int kMaxReuseInterval = 32;

void clear_mask(uint8_t *selected, int steps) {
    std::fill(selected, selected + steps, static_cast<uint8_t>(0));
}

int fail_mask(uint8_t *selected, int steps) {
    clear_mask(selected, steps);
    return -1;
}

bool should_evaluate(int step, int steps, int reuse_interval) {
    return reuse_interval == 1 || step == 0 || step == steps - 1 || step % reuse_interval == 0;
}

} // namespace

int h3_dit_reuse_schedule(int steps, int reuse_interval, uint8_t *selected, int selected_count) {
    if (steps < 1 || reuse_interval < 1 || reuse_interval > kMaxReuseInterval || selected == nullptr ||
        selected_count < steps)
        return -1;

    clear_mask(selected, steps);
    int count = 0;
    for (int step = 0; step < steps; ++step) {
        if (!should_evaluate(step, steps, reuse_interval))
            continue;
        selected[step] = 1;
        ++count;
    }
    return count;
}

int h3_parse_reuse_steps(int steps, const char *text, uint8_t *selected) {
    if (text == nullptr || text[0] == '\0')
        return 0;
    if (steps < 1 || selected == nullptr)
        return -1;

    clear_mask(selected, steps);

    int count = 0;
    int previous = -1;
    const char *cursor = text;
    for (;;) {
        char *parsed_end = nullptr;
        const long value = std::strtol(cursor, &parsed_end, 10);
        if (parsed_end == cursor || value < 0L || value >= static_cast<long>(steps) ||
            value <= static_cast<long>(previous))
            return fail_mask(selected, steps);

        const int step = static_cast<int>(value);
        selected[step] = 1;
        previous = step;
        ++count;

        if (parsed_end[0] == '\0')
            break;
        if (parsed_end[0] != ',')
            return fail_mask(selected, steps);
        cursor = parsed_end + 1;
        if (cursor[0] == '\0')
            return fail_mask(selected, steps);
    }

    if (selected[0] == 0 || selected[steps - 1] == 0)
        return fail_mask(selected, steps);
    return count;
}

} // namespace mvllm
