#!/bin/sh
# Download a target model + its drafter, pinned to the exact revisions used for the published results.
# usage: scripts/get-model.sh qwen3-4b|qwen3.5-4b|lfm2.5-2.6b|gemma-4-e2b|gemma-4-e4b [DEST=models]
# needs: pip install -U huggingface_hub   (provides the `hf` command)
set -e
DEST=${2:-models}
get() { hf download "$1" --revision "$2" --local-dir "$DEST/$3"; }
case "$1" in
  qwen3-4b)    get mlx-community/Qwen3-4B-4bit        4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25 qwen3-4b-4bit
               get z-lab/Qwen3-4B-DFlash-b16          b74e3a329c4d963783143b1e970d95b002be72bd qwen3-4b-dflash ;;
  qwen3.5-4b)  get mlx-community/Qwen3.5-4B-4bit      0e7ffd5c629ef7719d4cbc04069232580bfa9d9c qwen3.5-4b-4bit
               get z-lab/Qwen3.5-4B-DFlash            9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf qwen3.5-4b-dflash ;;
  lfm2.5-2.6b) get mlx-community/LFM2.5-2.6B-4bit     72a08228cf703cd7dc0187abf4edbbd8e935f864 lfm2.5-2.6b-4bit
               get LiquidAI/LFM2.5-2.6B-DSpark        458cedab07d0f7b2b05700c77e1aa463d43d6f04 lfm2.5-2.6b-dspark ;;
  gemma-4-e2b) get mlx-community/gemma-4-e2b-it-4bit  238767527555cb75a05732a84dff5d6ba0dd6809 gemma-4-e2b-4bit
               get google/gemma-4-E2B-it-assistant    2d874ef7d29f9a30599a1e4b3c1cbc9595f005df gemma-4-e2b-assistant ;;
  gemma-4-e4b) get mlx-community/gemma-4-e4b-it-4bit  475b9088d29754a3379866cf5aeb6b41acd313c2 gemma-4-e4b-4bit
               get google/gemma-4-E4B-it-assistant    8d0031ea8c2109e2b1e86bb9368a4539b537f80a gemma-4-e4b-assistant ;;
  *) echo "usage: $0 qwen3-4b|qwen3.5-4b|lfm2.5-2.6b|gemma-4-e2b|gemma-4-e4b [DEST]"; exit 1 ;;
esac
