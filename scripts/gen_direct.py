#!/usr/bin/env python3
"""Experiment generator for mul_mat_direct.wgsl's quantized loop: RV row groups (4*RV rows) x NV column groups (4*NV cols).
usage: gen_direct.py RV NV > wgsl   (prints the replacement text for the DIRECT_QUANT block and the store loop)"""
import sys
RV, NV = int(sys.argv[1]), int(sys.argv[2])
C = int(sys.argv[3]) if len(sys.argv) > 3 else 4  # columns per group: 4 (mat4x4), 2 (mat2x4) or 1 (dot)
VT = {4: 'vec4<f32>', 2: 'vec2<f32>', 1: 'f32'}[C]
Z = {4: 'vec4<f32>(0.0)', 2: 'vec2<f32>(0.0)', 1: '0.0'}[C]
MT = {4: 'mat4x4<f32>', 2: 'mat2x4<f32>', 1: ''}[C]
R = 4 * RV
o = []
o.append("#ifdef DIRECT_QUANT")
o.append("#ifdef DIRECT_Q8_0\n    const PAIR_WORDS = 17u;\n    const HALF = 8u;\n#else\n    const PAIR_WORDS = 9u;\n    const HALF = 4u;\n#endif")
for g in range(RV):
    o.append(f"    let mrow{g} = min(vec4<u32>(row_base + {4*g}u) + vec4<u32>(0u, 1u, 2u, 3u), vec4<u32>(last_m));")
    o.append(f"    let rw{g} = (vec4<u32>(src0_batch_offset) + mrow{g} * params.stride_01) * (PAIR_WORDS / 2u * 4u + 2u) / 4u;")
for t in range(NV * C):
    o.append(f"    let xc{t} = (src1_batch_offset + min(col_base + {t}u, last_n) * params.stride_11) >> 2u;")
for r in range(R):
    for v in range(NV):
        o.append(f"    var acc{r}_{v} = {Z};")
o.append("    let n_pairs = params.k / 64u;\n    for (var pr = 0u; pr < n_pairs; pr++) {\n        let xq = pr * 16u;")
for g in range(RV):
    o.append(f"        let pw{g} = rw{g} + vec4<u32>(pr * PAIR_WORDS);")
    o.append(f"        var c{g} = vec4<u32>(src0[pw{g}.x], src0[pw{g}.y], src0[pw{g}.z], src0[pw{g}.w]);")
o.append("        for (var half = 0u; half < 2u; half++) {")
for g in range(RV):
    o.append(f"            let db{g} = select(c{g} >> vec4<u32>(16u), c{g} & vec4<u32>(0xFFFFu), half == 0u);")
    o.append(f"            let d{g} = vec4<f32>(unpack2x16float(db{g}.x)[0], unpack2x16float(db{g}.y)[0], unpack2x16float(db{g}.z)[0], unpack2x16float(db{g}.w)[0]);")
    o.append(f"            let wbase{g} = pw{g} + vec4<u32>(1u + half * HALF);")
for r in range(R):
    for v in range(NV): o.append(f"            var s{r}_{v} = {Z};")
o.append("            let xb = xq + half * 8u;\n            for (var j = 0u; j < HALF; j++) {")
for g in range(RV):
    o.append(f"                let wi{g} = wbase{g} + vec4<u32>(j);")
    o.append(f"                let nx{g} = vec4<u32>(src0[wi{g}.x], src0[wi{g}.y], src0[wi{g}.z], src0[wi{g}.w]);")
    o.append(f"                let w{g} = select(nx{g}, (c{g} >> vec4<u32>(16u)) | (nx{g} << vec4<u32>(16u)), half == 0u);")
    o.append(f"                c{g} = nx{g};")
xs = lambda v, off: MT + "(" + ", ".join(f"src1[xc{v*C+t} + xb + j{off}]" for t in range(C)) + ")"
mul = (lambda a, b: f"dot({a}, {b})") if C == 1 else (lambda a, b: f"{a} * {b}")
o.append("#ifdef DIRECT_Q8_0")
for v in range(NV): o.append(f"                let xm{v} = {xs(v, '')};")
for r in range(R):
    o.append(f"                let q{r} = unpack_i8(w{r//4}[{r%4}]);")
    for v in range(NV): o.append(f"                s{r}_{v} += {mul(f'q{r}', f'xm{v}')};")
o.append("#else")
for v in range(NV):
    o.append(f"                let xl{v} = {xs(v, '')};")
    o.append(f"                let xh{v} = {xs(v, ' + 4u')};")
for r in range(R):
    o.append(f"                let b{r} = unpack_u8(w{r//4}[{r%4}]);")
    o.append(f"                let lo{r} = vec4<f32>(b{r} & vec4<u32>(15u)) - 8.0;")
    o.append(f"                let hi{r} = vec4<f32>(b{r} >> vec4<u32>(4u)) - 8.0;")
    for v in range(NV): o.append(f"                s{r}_{v} += {mul(f'lo{r}', f'xl{v}')} + {mul(f'hi{r}', f'xh{v}')};")
o.append("#endif\n            }")
for r in range(R):
    for v in range(NV): o.append(f"            acc{r}_{v} += d{r//4}[{r%4}] * s{r}_{v};")
o.append("        }\n    }")
print("\n".join(o))
print("//STORE")
comps = [''] if C == 1 else ['.x', '.y', '.z', '.w'][:C]
print(f"    let acc = array<f32, {R*NV*C}>(" + ", ".join(f"acc{r}_{v}{c}" for r in range(R) for v in range(NV) for c in comps) + ");")
