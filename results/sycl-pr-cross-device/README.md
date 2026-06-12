# SYCL PR Cross-Device Results

This directory stores outputs from `eval/nx2/run_sycl_pr_cross_device.sh`.

Small-fixture local runs can sanity-check the runner, but they do **not** prove
the Q6_K path and should not be cited as full non-B70 validation for the
llama.cpp PR.

For useful PR evidence, run the harness on a non-B70 Intel SYCL GPU with a
mixed K-quant MoE model whose down-projection experts are Q6_K.
