//! The Parakeet TDT encoder imported from ONNX by burn-onnx (onnx2burn), on Burn's wgpu backend.
//! Encoder only: mel features and the reference encoder output come from `ort-bench --dump`.
//!
//! burn-bench [--bpk FILE] [--backend wgpu|vulkan] [--runs 10] DUMP_PREFIX ...
//!   DUMP_PREFIX e.g. cache/dump/a07 (expects PREFIX.feat-128xT.f32 and PREFIX.enc-1024xT.f32)
//!
//! Every timing ends with into_data(), which reads the encoder output back from the GPU.

#[allow(clippy::all, unused, non_snake_case)]
mod encoder {
    include!("../../cache/burn-gen/encoder-model.rs");
}

use std::{fs, process::Command, time::Instant};

use burn::tensor::{Device, DeviceKind, Int, Tensor, TensorData};
use serde_json::json;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn status_mib(key: &str) -> f64 {
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    status.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok()).map(|kb| (kb / 1024.0).round()).unwrap_or(0.0)
}

fn gpu_mib() -> u64 {
    let pid = std::process::id() as u64;
    let Ok(out) = Command::new("nvidia-smi").args(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).output() else { return 0 };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let mut p = l.split(',').map(|s| s.trim().parse::<u64>().ok());
            match (p.next().flatten(), p.next().flatten()) {
                (Some(id), Some(mem)) if id == pid => Some(mem),
                _ => None,
            }
        })
        .sum()
}

fn floats(path: &str) -> Vec<f32> {
    fs::read(path).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

/// PREFIX.<kind>-<A>x<B>.f32 -> (path, A, B)
fn find(prefix: &str, kind: &str) -> (String, usize, usize) {
    let (dir, stem) = prefix.rsplit_once('/').unwrap_or((".", prefix));
    let start = format!("{stem}.{kind}-");
    let name = fs::read_dir(dir).unwrap().filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().to_string()).find(|n| n.starts_with(&start)).expect("dump file");
    let dims = name[start.len()..name.len() - 4].split_once('x').unwrap();
    (format!("{dir}/{name}"), dims.0.parse().unwrap(), dims.1.parse().unwrap())
}

fn stats(v: &[f64]) -> serde_json::Value {
    if v.is_empty() {
        return json!(null);
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let r = |x: f64| (x * 10.0).round() / 10.0;
    json!({ "median": r(s[s.len() / 2]), "worst": r(s[s.len() - 1]), "best": r(s[0]) })
}

fn main() {
    let start = Instant::now();
    let mut args = std::env::args().skip(1);
    let (mut bpk, mut backend, mut runs, mut clips) = ("cache/burn-gen/encoder-model.bpk".to_string(), "wgpu".to_string(), 10usize, vec![]);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bpk" => bpk = args.next().unwrap(),
            "--backend" => backend = args.next().unwrap(),
            "--runs" => runs = args.next().unwrap().parse().unwrap(),
            _ => clips.push(a),
        }
    }
    let device = match backend.as_str() {
        #[cfg(feature = "vulkan")]
        "vulkan" => Device::vulkan(DeviceKind::DiscreteGpu(0)),
        _ => Device::wgpu(DeviceKind::DiscreteGpu(0)),
    };
    let t = Instant::now();
    let model = encoder::Model::from_file(&bpk, &device);
    device.sync().unwrap();
    let load = ms(t);
    std::thread::sleep(std::time::Duration::from_millis(250));
    println!(
        "{}",
        json!({ "record": "load", "runtime": "burn 0.22", "backend": backend, "identity": format!("{:?}", device.identity()),
            "load_ms": { "encoder": load.round(), "process_start_to_loaded": ms(start).round() },
            "gpu_mib_after_load": gpu_mib(), "rss_mib_after_load": status_mib("VmRSS:") })
    );

    let mut peak = 0;
    for (i, prefix) in clips.iter().enumerate() {
        let (feat_path, mels, frames) = find(prefix, "feat");
        let (ref_path, dim, out_frames) = find(prefix, "enc");
        let (feats, reference) = (floats(&feat_path), floats(&ref_path));
        let run = || {
            let t = Instant::now();
            let input = Tensor::<3>::from_data(TensorData::new(feats.clone(), [1, mels, frames]), &device);
            let length = Tensor::<1, Int>::from_data(TensorData::new(vec![frames as i64], [1]), &device);
            let (out, _lengths) = model.forward(input, length);
            let data = out.into_data(); // reads back from the GPU
            (ms(t), data)
        };
        let (first_ms, data) = run();
        let since_start = ms(start);
        let shape = data.shape().clone();
        let out: Vec<f32> = data.try_to_vec::<f32>().unwrap();
        let max_diff = out.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let max_ref = reference.iter().fold(0f32, |m, v| m.max(v.abs()));
        let mut times = vec![];
        if runs > 0 {
            run();
            for _ in 0..runs {
                times.push(run().0);
            }
        }
        peak = peak.max(gpu_mib());
        println!(
            "{}",
            json!({ "record": "audio", "file": prefix, "seconds": (frames as f64 / 100.0 * 100.0).round() / 100.0, "first_in_process": i == 0,
                "first_run_ms": { "enc": first_ms.round() },
                "process_start_to_first_encoder_output_ms": if i == 0 { json!(since_start.round()) } else { json!(null) },
                "warm_runs": times.len(), "enc_ms": stats(&times),
                "output_shape": format!("{shape:?}"), "expected_shape": [1, dim, out_frames],
                "max_abs_diff_vs_ort_cpu": max_diff, "max_abs_reference": max_ref,
                "gpu_mib_peak": peak, "rss_mib_peak": status_mib("VmHWM:") })
        );
    }
}
