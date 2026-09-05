#include "dump_env.hpp"

#include <cstdlib>
#include <sys/stat.h>

namespace mvllm {
namespace {

const char *nonempty_env(const char *name) {
    const char *v = std::getenv(name);
    if (!v || !v[0])
        return nullptr;
    return v;
}

const char *pair(const char *mvllm, const char *coli, const char **which) {
    if (const char *v = nonempty_env(mvllm)) {
        if (which)
            *which = mvllm;
        return v;
    }
    if (const char *v = nonempty_env(coli)) {
        if (which)
            *which = coli;
        return v;
    }
    return nullptr;
}

bool is_dir(const char *p) {
    if (!p || !p[0])
        return false;
    struct stat st {};
    return ::stat(p, &st) == 0 && S_ISDIR(st.st_mode);
}

} // namespace

const char *dump_env_k3() { return pair("MVLLM_DUMP_K3", "COLI_DUMP_K3", nullptr); }

const char *dump_env_glm53() { return pair("MVLLM_DUMP_GLM53", "COLI_DUMP_GLM53", nullptr); }

const char *dump_env_h3() { return pair("MVLLM_DUMP_H3", "COLI_DUMP_H3", nullptr); }

const char *dump_env_dsv4() { return pair("MVLLM_DUMP_DSV4", "COLI_DUMP_DSV4", nullptr); }

const char *dump_env_for(Family f) {
    switch (f) {
    case Family::KimiK3:
        return dump_env_k3();
    case Family::Glm53:
        return dump_env_glm53();
    case Family::H3:
        return dump_env_h3();
    case Family::Dsv4:
        return dump_env_dsv4();
    default:
        return nullptr;
    }
}

const char *dump_env_first_dir(const char **which) {
    const char *name = nullptr;
    const char *p = pair("MVLLM_DUMP_K3", "COLI_DUMP_K3", &name);
    if (is_dir(p)) {
        if (which)
            *which = name;
        return p;
    }
    p = pair("MVLLM_DUMP_GLM53", "COLI_DUMP_GLM53", &name);
    if (is_dir(p)) {
        if (which)
            *which = name;
        return p;
    }
    p = pair("MVLLM_DUMP_H3", "COLI_DUMP_H3", &name);
    if (is_dir(p)) {
        if (which)
            *which = name;
        return p;
    }
    p = pair("MVLLM_DUMP_DSV4", "COLI_DUMP_DSV4", &name);
    if (is_dir(p)) {
        if (which)
            *which = name;
        return p;
    }
    if (which)
        *which = nullptr;
    return nullptr;
}

} // namespace mvllm
