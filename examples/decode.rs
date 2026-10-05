// Copyright 2025 Raph Levien
// SPDX-License-Identifier: Apache-2.0 OR MIT

use clap::Parser;
use mp3dec::{Decoder, FrameInfo};

#[derive(Parser)]
struct Args {
    mp3_file: String,
    out_file: Option<String>,
}

fn main() {
    let args = Args::parse();
    let bytes = std::fs::read(args.mp3_file).unwrap();
    println!("len = {}", bytes.len());
    let mut offset = 0;
    let mut decoder = Decoder::new();
    let mut info = FrameInfo::default();
    let mut pcm = [0; 2304];
    let mut n_channels = 0;
    let mut out = None;
    while offset < bytes.len() {
        let n_samples = decoder.decode_frame(&bytes[offset..], Some(&mut pcm), &mut info);
        if n_samples != 0 && out.is_none() {
            if let Some(out_file) = &args.out_file {
                n_channels = info.channels;
                let spec = hound::WavSpec {
                    channels: n_channels as u16,
                    sample_rate: info.hz as u32,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                };
                out = Some(hound::WavWriter::create(out_file, spec).unwrap());
            }
        }
        println!(
            "n_samples = {n_samples}, frame_bytes = {}",
            info.frame_bytes
        );
        if let Some(writer) = &mut out {
            for sample in &pcm[0..n_samples * n_channels] {
                _ = writer.write_sample(*sample);
            }
        }
        offset += info.frame_bytes;
    }
    if let Some(writer) = out {
        writer.finalize().unwrap()
    }
}
