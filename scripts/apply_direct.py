#!/usr/bin/env python3
"""Rewrite mul_mat_direct.wgsl's quantized block for a (RV, NV) tile. usage: apply_direct.py RV NV"""
import subprocess, sys, os
R = os.path.expanduser('~/devfs/cache/parakeet-ggml-webgpu')
p = R + '/transcribe.cpp/ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_direct.wgsl'
base = subprocess.check_output(['git', '-C', R + '/transcribe.cpp', 'show', 'c5a43d4f:ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_direct.wgsl'], text=True)
RV, NV = sys.argv[1], sys.argv[2]
C = sys.argv[3] if len(sys.argv) > 3 else '4'
gen = subprocess.check_output([sys.executable, R + '/scripts/gen_direct.py', RV, NV, C], text=True)
quant, store = gen.split('//STORE\n')
s = base
i = s.index('#ifdef DIRECT_QUANT\n    // Blocks are walked in pairs'); j = s.index('#else\n    let r0 = (src0_batch_offset + m0')
comment = s[i:s.index('#ifdef DIRECT_Q8_0\n    const PAIR_WORDS')].replace('#ifdef DIRECT_QUANT\n', '')
s = s[:i] + quant.replace('#ifdef DIRECT_QUANT\n', '#ifdef DIRECT_QUANT\n' + comment, 1) + s[j:]
# float path builds its own acc array
s = s.replace('#endif\n\n    let dst_batch_offset', '    let acc = array<f32, 16>(acc0.x, acc0.y, acc0.z, acc0.w, acc1.x, acc1.y, acc1.z, acc1.w, acc2.x, acc2.y, acc2.z, acc2.w, acc3.x, acc3.y, acc3.z, acc3.w);\n#endif\n\n    let dst_batch_offset')
s = s.replace(store.strip() if False else '', '')
k = s.index('    let dst_batch_offset'); 
s = s[:k] + '''    let dst_batch_offset = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride;
    let rows = min(TILE_ROWS, params.m - row_base);
    let cols = min(TILE_COLS, params.n - col_base);
    for (var tn = 0u; tn < cols; tn++) {
        let col = dst_batch_offset + (col_base + tn) * params.m + row_base;
        for (var tm = 0u; tm < rows; tm++) {
            dst[col + tm] = acc[tm * TILE_COLS + tn];
        }
    }
}
'''
s = s.replace('    var acc0 = vec4<f32>(0.0);\n    var acc1 = vec4<f32>(0.0);\n    var acc2 = vec4<f32>(0.0);\n    var acc3 = vec4<f32>(0.0);\n\n#ifdef DIRECT_QUANT', '#ifdef DIRECT_QUANT')
s = s.replace('#else\n    let r0 = (src0_batch_offset + m0', '''%s#else
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let r0 = (src0_batch_offset + m0''' % store)
s = s.replace('const TOTAL_WORKGROUP_SIZE = WORKGROUP_SIZE_M * WORKGROUP_SIZE_N;', '''const TOTAL_WORKGROUP_SIZE = WORKGROUP_SIZE_M * WORKGROUP_SIZE_N;
// Output tile per invocation: TILE_ROWS weight rows x TILE_COLS activation rows.
#ifdef DIRECT_QUANT
const TILE_ROWS = %du;
const TILE_COLS = %du;
#else
const TILE_ROWS = 4u;
const TILE_COLS = 4u;
#endif''' % (4 * int(RV), int(C) * int(NV)))
s = s.replace('(params.n + WORKGROUP_SIZE_N * 4u - 1u) / (WORKGROUP_SIZE_N * 4u)', '(params.n + WORKGROUP_SIZE_N * TILE_COLS - 1u) / (WORKGROUP_SIZE_N * TILE_COLS)')
s = s.replace('(params.m + WORKGROUP_SIZE_M * 4u - 1u) / (WORKGROUP_SIZE_M * 4u)', '(params.m + WORKGROUP_SIZE_M * TILE_ROWS - 1u) / (WORKGROUP_SIZE_M * TILE_ROWS)')
s = s.replace('* WORKGROUP_SIZE_M + local_m) * 4u;', '* WORKGROUP_SIZE_M + local_m) * TILE_ROWS;')
s = s.replace('* WORKGROUP_SIZE_N + local_n) * 4u;', '* WORKGROUP_SIZE_N + local_n) * TILE_COLS;')
open(p, 'w').write(s)
