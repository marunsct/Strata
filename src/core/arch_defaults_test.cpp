// src/core/arch_defaults_test.cpp - the per-architecture default table (no GPU needed).
#include "strata/core/arch_defaults.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static void unset(const char* k) {
#if defined(_WIN32)
    _putenv_s(k, "");
#else
    unsetenv(k);
#endif
}
static void put(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

int main() {
    using namespace strata::core;
    unset("STRATA_GFX1151_DEFAULTS");
    unset("STRATA_GFX1100_DEFAULTS");
    // other architectures (and CUDA, which passes an empty name): nothing
    CHECK(arch_default_env("gfx1100").empty());
    CHECK(arch_default_env("gfx1100:sramecc-:xnack-").empty());   // the default start on the RX 7900 is unchanged
    CHECK(arch_default_env("gfx1101").empty());
    CHECK(arch_default_env("gfx1102").empty());
    CHECK(arch_default_env("gfx1150").empty());
    CHECK(arch_default_env("gfx1152").empty());
    CHECK(arch_default_env("gfx11510").empty());   // not a prefix match
    CHECK(arch_default_env("").empty());
    CHECK(arch_default_env(nullptr).empty());
    // gfx1151, plain and with its feature suffix: the table, and no bit-changing switch in it
    for (const char* a : {"gfx1151", "gfx1151:sramecc-:xnack-"}) {
        const auto t = arch_default_env(a);
        CHECK(t.size() >= 10);
        for (const auto& kv : t) {
            CHECK(kv.first.rfind("STRATA_", 0) == 0);
            for (const char* bits : {"STRATA_PF_GEMM", "STRATA_PF_FUSED", "STRATA_HC_UPMIX", "STRATA_PA_FAST", "STRATA_HIP_WMMA",
                                     "STRATA_SELECT_WMMA", "STRATA_HC_Q8", "STRATA_PF_HCDOWN", "STRATA_WMMA_GEMM"})
                CHECK(kv.first != bits);
        }
    }
    // gfx12: the five prompt switches only, no decode switch, no bit-changing one
    unset("STRATA_GFX12_DEFAULTS");
    for (const char* a : {"gfx1200", "gfx1201", "gfx1201:sramecc-:xnack-"}) {
        const auto t = arch_default_env(a);
        CHECK(t.size() == 5);
        for (const auto& kv : t) CHECK(kv.first != "STRATA_HCD_EXACT");
        for (const auto& kv : t) {
            CHECK(kv.first != "STRATA_Q8_PACKED" && kv.first != "STRATA_SH_STREAM" && kv.first != "STRATA_PF_GEMM" &&
                  kv.first != "STRATA_PF_FUSED");
        }
    }
    CHECK(arch_default_env("gfx12010").empty());
    put("STRATA_GFX12_DEFAULTS", "0");
    CHECK(arch_default_env("gfx1201").empty());
    unset("STRATA_GFX12_DEFAULTS");
    // gfx1100 (RX 7900 XTX / XT): opt-in only.  0 and an unset variable give nothing; 1 is the gfx12 prompt half plus
    // STRATA_SPLIT_OVERLAP; 2 adds the gfx1151 decode half; no bit-changing switch in either; exactly gfx1100, never 7800 / 7600
    put("STRATA_GFX1100_DEFAULTS", "0");
    CHECK(arch_default_env("gfx1100").empty());
    put("STRATA_GFX1100_DEFAULTS", "1");
    for (const char* a : {"gfx1100", "gfx1100:sramecc-:xnack-"}) {
        const auto t = arch_default_env(a);
        CHECK(t.size() == 6);
        bool overlap = false, gdn = false;
        for (const auto& kv : t) {
            if (kv.first == "STRATA_SPLIT_OVERLAP") overlap = kv.second == "1";
            if (kv.first == "STRATA_GDN_HEAD") gdn = kv.second == "1";
            CHECK(kv.first != "STRATA_Q8_PACKED" && kv.first != "STRATA_HCD_EXACT" && kv.first != "STRATA_SH_STREAM");
            for (const char* bits : {"STRATA_PF_GEMM", "STRATA_PF_FUSED", "STRATA_HC_UPMIX", "STRATA_PA_FAST", "STRATA_HIP_WMMA",
                                     "STRATA_SELECT_WMMA", "STRATA_HC_Q8", "STRATA_PF_HCDOWN", "STRATA_WMMA_GEMM"})
                CHECK(kv.first != bits);
        }
        CHECK(overlap && gdn);
    }
    CHECK(arch_default_env("gfx1101").empty());
    CHECK(arch_default_env("gfx1102").empty());
    CHECK(arch_default_env("gfx11000").empty());
    put("STRATA_GFX1100_DEFAULTS", "2");
    {
        const auto t = arch_default_env("gfx1100");
        CHECK(t.size() == 16);
        bool q8 = false, sh = false;
        for (const auto& kv : t) { if (kv.first == "STRATA_Q8_PACKED") q8 = true; if (kv.first == "STRATA_SH_STREAM") sh = true; }
        CHECK(q8 && !sh);   // SH_STREAM is setup's HIP calibration's to decide
    }
    // a user's setting wins there too, and the switch is reported as not set
    put("STRATA_SPLIT_OVERLAP", "0");
    {
        const auto set = apply_arch_defaults("gfx1100");
        CHECK(std::string(std::getenv("STRATA_SPLIT_OVERLAP")) == "0");
        for (const auto& n : set) CHECK(n != "STRATA_SPLIT_OVERLAP");
        CHECK(std::getenv("STRATA_GDN_PP") != nullptr && std::string(std::getenv("STRATA_GDN_PP")) == "2");
    }
    unset("STRATA_SPLIT_OVERLAP");
    unset("STRATA_GFX1100_DEFAULTS");
    CHECK(arch_default_env("gfx1100").empty());
    // a user's setting wins, whatever it is; the rest are set
    put("STRATA_TSUM", "0");
    unset("STRATA_QFUSE");
    const auto set = apply_arch_defaults("gfx1151");
    CHECK(std::getenv("STRATA_TSUM") != nullptr && std::string(std::getenv("STRATA_TSUM")) == "0");
    CHECK(std::getenv("STRATA_QFUSE") != nullptr && std::string(std::getenv("STRATA_QFUSE")) == "1");
    for (const auto& n : set) CHECK(n != "STRATA_TSUM");
    // the opt-out
    put("STRATA_GFX1151_DEFAULTS", "0");
    CHECK(arch_default_env("gfx1151").empty());
    CHECK(apply_arch_defaults("gfx1151").empty());
    std::printf(fails ? "arch_defaults_test: %d FAILED\n" : "arch_defaults_test: OK\n", fails);
    return fails ? 1 : 0;
}
