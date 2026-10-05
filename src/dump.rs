// Copyright 2025 Raph Levien
// SPDX-License-Identifier: Apache-2.0 OR MIT

pub fn dump(name: &str, grbuf: &[f32]) {
    print!("{name}");
    for x in grbuf {
        print!(" {x}");
    }
    println!()
}
