# mp3dec

An MP3 (MPEG-1/2/2.5 Layer III) decoder in Rust, ported by hand from [minimp3](https://github.com/lieff/minimp3).
It builds fine as `no-std` and is suitable for microcontrollers, though it does rely on floating point.
There's a fair amount of validation and testing that's been done (it fixes a bug or two present in minimp3), though none of the test mechanism is checked in.

Layers I and II (MP1/MP2) are not supported; frames are skipped. Adding them wouldn't be too hard, and some of the groundwork is in place.

The `dump` feature is for internal development only, and produces intermediate vectors that can be compared against minimp3.

## License

Licensed under either of

- Apache License, Version 2.0
  ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license
  ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

minimp3 is released under CC0 1.0 (public domain dedication).
