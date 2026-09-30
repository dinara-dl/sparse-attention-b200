"""Dense14 main p02/p03; archived p01, s01..s10 and v1..v15 retain their meaning."""

MAIN_SOURCES = {
    "p02": "01_block_sparse.cu",
    "p03": "02_gqa_block_sparse.cu",
}
MAIN_VARIANTS = {"p02": 1, "p03": 2}
MAIN_ORIGINS = {
    "p02": "codex_1/dense14_sparse_progression/01_sparse_two_q.cu",
    "p03": "codex_1/dense14_sparse_progression/02_sparse_gqa.cu",
}
# Preserve old CLI/API IDs and operator variant numbers; do not rename old results.
ARCHIVED_PROGRESSION_SOURCES = {
    "p01": "archive/dense13_progression/01_block_sparse.cu",
}
PROGRESSION_SOURCES = {**ARCHIVED_PROGRESSION_SOURCES, **MAIN_SOURCES}
PROGRESSION_VARIANTS = {"p01": 0, **MAIN_VARIANTS}
PROGRESSION_ORIGINS = {
    "p01": "codex_1/dense13_sparse_comparison/01_sparse_two_q.cu", **MAIN_ORIGINS,
}

COMPACT_FILENAMES = {
    "s01": "01_baseline_block_sparse.cu",
    "s02": "02_kv_load_pipeline_block_sparse.cu",
    "s03": "03_compute_pipeline_block_sparse.cu",
    "s04": "04_log2_hardware_exp2_block_sparse.cu",
    "s05": "05_cache96_reread32_block_sparse.cu",
    "s06": "06_early_pv_96_32_block_sparse.cu",
    "s07": "07_persistent_cta_fma_block_sparse.cu",
    "s08": "08_lazy_rescale_threshold8_block_sparse.cu",
    "s09": "09_true_qstage2_block_sparse.cu",
    "s10": "10_native_gqa_block_sparse.cu",
}
COMPACT_SOURCES = {key: f"archive/s01_s10/{name}" for key, name in COMPACT_FILENAMES.items()}

LEGACY_FILENAMES = (
    "1_baseline_block_sparse.cu",
    "2_p_to_tmem_block_sparse.cu",
    "3_swizzle_block_sparse.cu",
    "4_warp_specialization_block_sparse.cu",
    "5_kv_load_pipeline_block_sparse.cu",
    "6_compute_pipeline_block_sparse.cu",
    "7_log2_hardware_exp2_block_sparse.cu",
    "8_cache96_reread32_block_sparse.cu",
    "9_kv4_load_pipeline_block_sparse.cu",
    "10_early_pv_96_32_block_sparse.cu",
    "11_selective_fma_exp2_block_sparse.cu",
    "12_persistent_cta_shell_block_sparse.cu",
    "13_lazy_rescale_threshold8_block_sparse.cu",
    "14_true_qstage2_block_sparse.cu",
    "15_native_gqa_block_sparse.cu",
)
LEGACY_SOURCES = {
    f"v{index}": f"archive/v1_v15/{filename}"
    for index, filename in enumerate(LEGACY_FILENAMES, 1)
}
KERNEL_SOURCES = {**PROGRESSION_SOURCES, **COMPACT_SOURCES, **LEGACY_SOURCES}
PARENTS = dict(zip(COMPACT_SOURCES, ("v4", "v5", "v6", "v7", "v9", "v10",
                                "v12", "v13", "v14", "v15")))
GQA_VERSIONS = frozenset({"p01", "p02", "p03", "s10", "v15"})
PERSISTENT_VERSIONS = frozenset({"p02", "p03", "s07", "s08", "s09", "s10", "v12", "v13", "v14", "v15"})
TWO_Q_VERSIONS = frozenset({"p01", "p02", "s09", "v14"})
