if [ -x ./build/micro-vllm-cuda ]; then
  exec ./build/micro-vllm-cuda "$@"
fi
echo "Llama CUDA demo not built. cmake -B build -DMVLLM_HOST_ONLY=OFF -DMVLLM_CUDA=ON" >&2
exit 1