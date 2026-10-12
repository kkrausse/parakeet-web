//! Parakeet TDT (istupakov / onnx-asr ONNX layout) on ONNX Runtime, encoder on a chosen execution provider.
//! Works for the 0.6b v2 export (128 mels, 2x640 LSTM state) and the 110m one (80 mels, 1x640): the mel
//! front end is whichever nemo*.onnx sits in the model directory and the state shape is read from the decoder.
//!
//! ort-bench --model DIR [--encoder FILE] [--enc-ep webgpu|cpu|cuda] [--dec-ep cpu|webgpu|cuda]
//!           [--runs 10] [--threads N] [--dec-threads N] [--profile PREFIX] [--dump DIR] [--verbose] AUDIO.f32 ...
//!
//! AUDIO is raw mono 16 kHz f32le. Prints one JSON object per line: a `load` record, then one record
//! per audio file. Every session.run() returns CPU tensors, so each timing includes the GPU finishing
//! and the output being read back.

use std::{
    collections::HashMap,
    fs,
    process::Command,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use ort::{
    ep::{self, ExecutionProviderDispatch},
    logging::LogLevel,
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use serde_json::json;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const MAX_SYMBOLS_PER_FRAME: usize = 10;
const TDT_DURATIONS: usize = 5; // durations 0..=4 follow the vocabulary logits

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn proc_status_mib(key: &str) -> f64 {
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    status
        .lines()
        .find(|l| l.starts_with(key))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<f64>().ok())
        .map(|kb| (kb / 1024.0).round())
        .unwrap_or(0.0)
}

fn gpu_mib(pid: u32) -> u64 {
    let out = Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
        .output();
    let Ok(out) = out else { return 0 };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let mut p = l.split(',').map(|s| s.trim().parse::<u64>().ok());
            match (p.next().flatten(), p.next().flatten()) {
                (Some(id), Some(mem)) if id == pid as u64 => Some(mem),
                _ => None,
            }
        })
        .sum()
}

fn provider(name: &str) -> Res<Vec<ExecutionProviderDispatch>> {
    Ok(match name {
        "cpu" => vec![],
        "webgpu" => vec![ep::WebGPU::default().build().error_on_failure()],
        #[cfg(feature = "cuda")]
        "cuda" => vec![ep::CUDA::default()
            .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Heuristic)
            .build()
            .error_on_failure()],
        other => return Err(format!("unknown execution provider {other}").into()),
    })
}

struct Options {
    verbose: bool,
    threads: Option<usize>,
    dec_threads: Option<usize>,
    profile: Option<String>,
}

fn session(path: &str, ep_name: &str, o: &Options, threads: Option<usize>, profile: bool) -> Res<Session> {
    let mut b = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|e| e.to_string())?
        .with_log_level(if o.verbose { LogLevel::Verbose } else { LogLevel::Warning })
        .map_err(|e| e.to_string())?
        .with_execution_providers(provider(ep_name)?)
        .map_err(|e| e.to_string())?;
    if let Some(n) = threads {
        b = b.with_intra_threads(n).map_err(|e| e.to_string())?;
    }
    if let (true, Some(prefix)) = (profile, &o.profile) {
        b = b.with_profiling(prefix).map_err(|e| e.to_string())?;
    }
    Ok(b.commit_from_file(path)?)
}

struct Model {
    pre: Session,
    enc: Session,
    dec: Session,
    vocab: Vec<String>,
    blank: usize,
    /// LSTM state shape of the prediction network: (layers, hidden).
    state: (usize, usize),
    /// Path prefix: write the mel features and the encoder output of each clip there (for burn-bench).
    dump: Option<String>,
}

struct Timing {
    pre: f64,
    enc: f64,
    dec: f64,
    total: f64,
    frames: usize,
    steps: usize,
    text: String,
}

fn transcribe(m: &mut Model, audio: &[f32]) -> Res<Timing> {
    let start = Instant::now();
    let t = Instant::now();
    let out = m.pre.run(ort::inputs![
        "waveforms" => Tensor::from_array(([1, audio.len()], audio.to_vec()))?,
        "waveforms_lens" => Tensor::from_array(([1], vec![audio.len() as i64]))?,
    ])?;
    let (shape, feats) = out["features"].try_extract_tensor::<f32>()?;
    let (shape, feats) = (shape.to_vec(), feats.to_vec());
    if let Some(prefix) = &m.dump {
        // [1, 128, T] row-major
        fs::write(format!("{prefix}.feat-{}x{}.f32", shape[1], shape[2]), feats.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    }
    let n_feat = out["features_lens"].try_extract_tensor::<i64>()?.1[0];
    drop(out);
    let pre = ms(t);

    let t = Instant::now();
    let out = m.enc.run(ort::inputs![
        "audio_signal" => Tensor::from_array((shape, feats))?,
        "length" => Tensor::from_array(([1], vec![n_feat]))?,
    ])?;
    let (shape, enc_out) = out["outputs"].try_extract_tensor::<f32>()?;
    let (dim, max_frames) = (shape[1] as usize, shape[2] as usize);
    if let Some(prefix) = &m.dump {
        // [1, D, T'] row-major
        fs::write(format!("{prefix}.enc-{dim}x{max_frames}.f32"), enc_out.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    }
    let frames = (out["encoded_lengths"].try_extract_tensor::<i64>()?.1[0] as usize).min(max_frames);
    // [1, D, T] -> T rows of D
    let mut rows = vec![0f32; frames * dim];
    for d in 0..dim {
        for f in 0..frames {
            rows[f * dim + d] = enc_out[d * max_frames + f];
        }
    }
    drop(out);
    let enc = ms(t);

    let t = Instant::now();
    let (layers, hidden) = m.state;
    let (mut s1, mut s2) = (vec![0f32; layers * hidden], vec![0f32; layers * hidden]);
    let mut tokens: Vec<usize> = vec![];
    let (mut frame, mut emitted, mut steps) = (0usize, 0usize, 0usize);
    while frame < frames {
        let last = *tokens.last().unwrap_or(&m.blank) as i32;
        let out = m.dec.run(ort::inputs![
            "encoder_outputs" => Tensor::from_array(([1, dim, 1], rows[frame * dim..(frame + 1) * dim].to_vec()))?,
            "targets" => Tensor::from_array(([1, 1], vec![last]))?,
            "target_length" => Tensor::from_array(([1], vec![1i32]))?,
            "input_states_1" => Tensor::from_array(([layers, 1, hidden], s1.clone()))?,
            "input_states_2" => Tensor::from_array(([layers, 1, hidden], s2.clone()))?,
        ])?;
        steps += 1;
        let logits = out["outputs"].try_extract_tensor::<f32>()?.1;
        let n_vocab = logits.len() - TDT_DURATIONS;
        let argmax = |s: &[f32]| s.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        let (token, skip) = (argmax(&logits[..n_vocab]), argmax(&logits[n_vocab..]));
        if token != m.blank {
            s1.copy_from_slice(out["output_states_1"].try_extract_tensor::<f32>()?.1);
            s2.copy_from_slice(out["output_states_2"].try_extract_tensor::<f32>()?.1);
            tokens.push(token);
            emitted += 1;
        }
        if skip > 0 {
            frame += skip;
            emitted = 0;
        } else if token == m.blank || emitted == MAX_SYMBOLS_PER_FRAME {
            frame += 1;
            emitted = 0;
        }
    }
    let dec = ms(t);
    let text: String = tokens.iter().map(|&t| m.vocab[t].as_str()).collect::<String>().replace('\u{2581}', " ");
    Ok(Timing { pre, enc, dec, total: ms(start), frames, steps, text: text.trim().to_string() })
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

fn main() -> Res<()> {
    let process_start = Instant::now();
    let mut args = std::env::args().skip(1);
    let mut named: HashMap<String, String> = HashMap::new();
    let mut audio_paths = vec![];
    let mut verbose = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--verbose" => verbose = true,
            _ if a.starts_with("--") => {
                named.insert(a[2..].to_string(), args.next().ok_or("missing value")?);
            }
            _ => audio_paths.push(a),
        }
    }
    let get = |k: &str, d: &str| named.get(k).cloned().unwrap_or(d.to_string());
    let dir = get("model", "cache/onnx-tdt-v2");
    let (enc_ep, dec_ep) = (get("enc-ep", "webgpu"), get("dec-ep", "cpu"));
    let runs: usize = get("runs", "10").parse()?;
    let o = Options {
        verbose,
        threads: named.get("threads").map(|t| t.parse()).transpose()?,
        dec_threads: named.get("dec-threads").map(|t| t.parse()).transpose()?,
        profile: named.get("profile").cloned() };

    // Peak GPU memory of this process, sampled in the background.
    let (peak, stop) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicBool::new(false)));
    let sampler = {
        let (peak, stop, pid) = (peak.clone(), stop.clone(), std::process::id());
        thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                peak.fetch_max(gpu_mib(pid), Ordering::Relaxed);
                thread::sleep(Duration::from_millis(100));
            }
        })
    };

    let vocab_text = fs::read_to_string(format!("{dir}/vocab.txt"))?;
    let vocab: Vec<String> = vocab_text.lines().map(|l| l.rsplit_once(' ').map_or(l, |p| p.0).to_string()).collect();
    let blank = vocab.iter().position(|t| t == "<blk>").ok_or("no <blk> in vocab")?;

    let t = Instant::now();
    let mut pre_names: Vec<String> = fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().to_string()))
        .filter(|n| n.starts_with("nemo") && n.ends_with(".onnx"))
        .collect();
    pre_names.sort();
    let pre_name = pre_names.first().ok_or("no nemo*.onnx front end in the model directory")?;
    let pre = session(&format!("{dir}/{pre_name}"), "cpu", &o, None, false)?;
    let pre_load = ms(t);
    let t = Instant::now();
    let enc = session(&format!("{dir}/{}", get("encoder", "encoder-model.onnx")), &enc_ep, &o, o.threads, true)?;
    let enc_load = ms(t);
    let t = Instant::now();
    let dec = session(&format!("{dir}/{}", get("decoder", "decoder_joint-model.onnx")), &dec_ep, &o, o.dec_threads, false)?;
    let dec_load = ms(t);
    let state_shape = dec.inputs().iter().find(|i| i.name() == "input_states_1").and_then(|i| i.dtype().tensor_shape()).ok_or("decoder has no input_states_1")?;
    let state = (state_shape[0] as usize, state_shape[2] as usize);
    let mut m = Model { pre, enc, dec, vocab, blank, state, dump: None };
    thread::sleep(Duration::from_millis(250));
    println!(
        "{}",
        json!({ "record": "load", "enc_ep": enc_ep, "dec_ep": dec_ep, "encoder": get("encoder", "encoder-model.onnx"),
            "load_ms": { "preprocessor": pre_load.round(), "encoder": enc_load.round(), "decoder": dec_load.round(),
                         "process_start_to_loaded": ms(process_start).round() },
            "gpu_mib_after_load": gpu_mib(std::process::id()), "rss_mib_after_load": proc_status_mib("VmRSS:") })
    );

    for (i, path) in audio_paths.iter().enumerate() {
        let bytes = fs::read(path)?;
        let audio: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let seconds = audio.len() as f64 / 16000.0;
        m.dump = named.get("dump").map(|dir| format!("{dir}/{}", std::path::Path::new(path).file_stem().unwrap().to_string_lossy()));
        let first = transcribe(&mut m, &audio)?;
        m.dump = None;
        let since_start = ms(process_start);
        let mut all: Vec<Timing> = vec![];
        if runs > 0 {
            transcribe(&mut m, &audio)?; // discarded warm-up
            for _ in 0..runs {
                all.push(transcribe(&mut m, &audio)?);
            }
        }
        let col = |f: fn(&Timing) -> f64| stats(&all.iter().map(f).collect::<Vec<_>>());
        let median_total = col(|t| t.total)["median"].as_f64();
        println!(
            "{}",
            json!({ "record": "audio", "file": path, "seconds": (seconds * 100.0).round() / 100.0,
                "first_in_process": i == 0,
                "first_run_ms": { "pre": first.pre.round(), "enc": first.enc.round(), "dec": first.dec.round(), "total": first.total.round() },
                "process_start_to_first_transcript_ms": if i == 0 { json!(since_start.round()) } else { json!(null) },
                "warm_runs": all.len(),
                "pre_ms": col(|t| t.pre), "enc_ms": col(|t| t.enc), "dec_ms": col(|t| t.dec), "total_ms": col(|t| t.total),
                "rtf_x_realtime": median_total.map(|t| (seconds * 1e3 / t * 10.0).round() / 10.0),
                "enc_frames": first.frames, "dec_steps": first.steps,
                "gpu_mib_peak": peak.load(Ordering::Relaxed), "rss_mib_peak": proc_status_mib("VmHWM:"),
                "text": first.text, "text_stable": all.iter().all(|t| t.text == first.text) })
        );
    }
    stop.store(true, Ordering::Relaxed);
    sampler.join().ok();
    if o.profile.is_some() {
        eprintln!("profile: {}", m.enc.end_profiling()?);
    }
    // A normal exit segfaults in teardown when a WebGPU session exists (ORT 1.28 + Dawn, after all
    // results are printed), so leave without running destructors.
    use std::io::Write;
    std::io::stdout().flush()?;
    extern "C" {
        fn _exit(code: i32) -> !;
    }
    unsafe { _exit(0) }
}
