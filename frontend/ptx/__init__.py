# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The shared PTX core of Layer 1 (design: PTX as the only IR): pure Python stdlib, no
torch / triton / cutlass, so the host adapters (frontend/ttgir, frontend/cute) and the
tests import it as lean_kernel_verify.frontend.ptx; its modules also run as scripts
(`python3 -I`). Each module's docstring states its contract.
"""
