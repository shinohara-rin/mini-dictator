# Rewrites the sherpa-onnx int8 Qwen3-ASR decoder with dequantized fp32 weights, so activations stay
# fp32 and a call of many tokens computes what single-token calls do.
# python3 qwen-dequantize-decoder.py decoder.int8.onnx out/decoder.onnx
import onnx, numpy as np, sys
from onnx import numpy_helper, helper
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
init = {t.name: t for t in g.initializer}
prod = {o: n for n in g.node for o in n.output}
cons = {}
for n in g.node:
    for i in n.input: cons.setdefault(i, []).append(n)
remove, add, new_inits = set(), [], []
count = 0
for n in list(g.node):
    if n.op_type != 'MatMulInteger': continue
    aq, wq, azp, wzp = n.input
    dql = prod[aq]; assert dql.op_type == 'DynamicQuantizeLinear'
    x = dql.input[0]
    cast = cons[n.output[0]]; assert len(cast) == 1 and cast[0].op_type == 'Cast'
    cast = cast[0]
    mul = cons[cast.output[0]]; assert len(mul) == 1 and mul[0].op_type == 'Mul'
    mul = mul[0]
    # scale product node feeding mul
    sname = [i for i in mul.input if i != cast.output[0]][0]
    smul = prod[sname]; assert smul.op_type == 'Mul'
    wscale = [i for i in smul.input if i in init][0]
    W = numpy_helper.to_array(init[wq]).astype(np.float32)
    Z = numpy_helper.to_array(init[wzp]).astype(np.float32)
    S = numpy_helper.to_array(init[wscale]).astype(np.float32)
    Wf = (W - Z) * S
    name = wq + '_fp32'
    new_inits.append(numpy_helper.from_array(Wf.astype(np.float32), name))
    add.append((mul, helper.make_node('MatMul', [x, name], [mul.output[0]], name=n.name + '_fp')))
    remove.update([n.name, cast.name, mul.name, smul.name])
    count += 1
# DynamicQuantizeLinear nodes whose outputs are now unused
nodes = []
for n in g.node:
    if n.name in remove: 
        for m_, new in add:
            if m_.name == n.name: nodes.append(new)
        continue
    nodes.append(n)
used = {i for n in nodes for i in n.input} | {o.name for o in g.output}
nodes = [n for n in nodes if not (n.op_type == 'DynamicQuantizeLinear' and not any(o in used for o in n.output))]
del g.node[:]; g.node.extend(nodes)
used = {i for n in nodes for i in n.input}
keep = [t for t in g.initializer if t.name in used]
del g.initializer[:]; g.initializer.extend(keep + new_inits)
print('replaced', count)
onnx.save(m, dst, save_as_external_data=True, location=dst.split('/')[-1] + '.data')
