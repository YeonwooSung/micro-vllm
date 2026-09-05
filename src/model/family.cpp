#include "family.hpp"

#include <memory>

namespace mvllm {

std::unique_ptr<FamilyEngine> make_kimi_k3();
std::unique_ptr<FamilyEngine> make_glm53();
std::unique_ptr<FamilyEngine> make_h3();
std::unique_ptr<FamilyEngine> make_llama();
std::unique_ptr<FamilyEngine> make_dsv4();

std::unique_ptr<FamilyEngine> make_engine(Family family) {
    switch (family) {
    case Family::KimiK3:
        return make_kimi_k3();
    case Family::Glm53:
        return make_glm53();
    case Family::H3:
        return make_h3();
    case Family::Llama:
        return make_llama();
    case Family::Dsv4:
        return make_dsv4();
    default:
        return nullptr;
    }
}

} // namespace mvllm
