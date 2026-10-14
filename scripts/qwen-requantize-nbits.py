# Quantizes a decoder from qwen-dequantize-decoder.py with MatMulNBits (weights 4 or 8 bit,
# activations quantized per row), so results do not depend on how many tokens a call reads.
# python3 qwen-requantize-nbits.py decoder.onnx out/decoder.onnx 4
import sys, onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig
src, dst, bits = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = onnx.load(src)
cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, accuracy_level=4, bits=bits)
q = MatMulNBitsQuantizer(m, bits=bits, block_size=32, is_symmetric=True, accuracy_level=4, algo_config=cfg)
q.process()
q.model.save_model_to_file(dst, use_external_data_format=True)
print("ok")
