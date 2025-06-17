#![no_std]

const MAX_SAMPLES_PER_FRAME: usize = 1152 * 2;

#[derive(Default)]
pub struct FrameInfo {
    pub frame_bytes: usize,
    pub frame_offset: usize,
    pub channels: usize,
    pub hz: usize,
    pub layer: usize,
    pub bitrate_kbps: usize,
}

pub struct Decoder {
    mdct_overlap: [[f32; 9 * 32]; 2],
    qmf_state: [f32; 15 * 2 * 32],
    reserv: usize,
    free_format_bytes: usize,
    header: Header,
    reserv_buf: [u8; 511],
}

// consider float output also
type Sample = i16;

const MAX_FREE_FORMAT_FRAME_SIZE: usize = 2304;
// Note: this is configurable
const MAX_FRAME_SYNC_MATCHES: usize = 10;
const MAX_L3_FRAME_PAYLOAD_BYTES: usize = MAX_FREE_FORMAT_FRAME_SIZE;
const MAX_BITRESERVOIR_BYTES: usize = 511;
const SHORT_BLOCK_TYPE: u8 = 2;
const STOP_BLOCK_TYPE: u8 = 3;
const HDR_SIZE: usize = 4;
const BITS_DEQUANTIZER_OUT: i32 = -1;
const MAX_SCF: i32 = 255 + BITS_DEQUANTIZER_OUT * 4 - 210;
const MAX_SCFI: i32 = (MAX_SCF + 3) & !3;

struct ScaleInfo {
    scf: [f32; 3 * 64],
    total_bands: u8,
    stereo_bands: u8,
    bitalloc: [u8; 64],
    scfcod: [u8; 64],
}

struct GrInfo {
    sfbtab: &'static [u8],
    part_23_length: u16,
    big_values: u16,
    scalefac_compress: u16,
    global_gain: u8,
    block_type: u8,
    mixed_block_flag: u8, // maybe should be bool?
    n_long_sfb: u8,
    n_short_sfb: u8,
    table_select: [u8; 3],
    region_count: [u8; 3],
    subblock_gain: [u8; 3],
    preflag: u8,
    scalefac_scale: u8,
    count1_table: u8,
    scfsi: u8,
}

struct SubbandAlloc {
    tab_offset: u8,
    code_tab_width: u8,
    band_count: u8,
}

struct Bs<'a> {
    buf: &'a [u8],
    core: &'a mut BsCore,
}

#[derive(Default)]
struct BsCore {
    pos: usize,
    limit: usize,
}

struct Scratch {
    bs: BsCore,
    maindata: [u8; MAX_BITRESERVOIR_BYTES + MAX_L3_FRAME_PAYLOAD_BYTES],
    gr_info: [GrInfo; 4],
    grbuf: [[f32; 576]; 2],
    scf: [f32; 40],
    syn: [[f32; 2 * 32]; 18 + 15],
    ist_pos: [[u8; 39]; 2],
}

impl<'a> Bs<'a> {
    fn new(buf: &'a [u8], core: &'a mut BsCore) -> Self {
        Self { buf, core }
    }

    fn get_bits(&mut self, n: usize) -> u32 {
        self.core.get_bits(self.buf, n)
    }
}

impl BsCore {
    fn new(len: usize) -> Self {
        Self {
            pos: 0,
            limit: len * 8,
        }
    }

    fn get_bits(&mut self, buf: &[u8], n: usize) -> u32 {
        let s = self.pos % 8;
        let mut shl = n + s;
        self.pos += n;
        if self.pos > self.limit {
            return 0;
        }
        let mut cache = 0;
        let mut ix = self.pos / 8;
        let mut next = buf[ix] as u32 & (0xff >> s);
        while shl > 8 {
            shl -= 8;
            cache |= next << shl;
            ix += 1;
            next = buf[ix] as u32;
        }
        cache | (next >> (8 - shl))
    }
}

#[derive(Clone, Copy, Default)]
struct Header([u8; 4]);

impl Header {
    /// Create new header.
    ///
    /// Input must be at least the size of a header, or panic.
    fn new(src: &[u8]) -> Self {
        let mut buf = [0; 4];
        buf.copy_from_slice(&src[..HDR_SIZE]);
        Self(buf)
    }

    fn is_mono(self) -> bool {
        self.0[3] & 0xc0 == 0xc0
    }

    fn is_ms_stereo(self) -> bool {
        self.0[3] & 0xe0 == 0x60
    }

    fn is_free_format(self) -> bool {
        self.0[2] & 0xf0 == 0
    }

    fn is_crc(self) -> bool {
        self.0[1] & 1 == 0
    }

    fn test_padding(self) -> bool {
        self.0[2] & 2 != 0
    }

    fn test_mpeg1(self) -> bool {
        self.0[1] & 0x8 != 0
    }

    fn test_not_mpeg25(self) -> bool {
        self.0[1] & 0x10 != 0
    }

    fn test_i_stereo(self) -> bool {
        self.0[3] & 0x10 != 0
    }

    fn test_ms_stereo(self) -> bool {
        self.0[3] & 0x20 != 0
    }

    fn get_stereo_mode(self) -> u8 {
        (self.0[3] >> 6) & 3
    }

    fn get_stereo_mode_ext(self) -> u8 {
        (self.0[3] >> 4) & 3
    }

    fn get_layer(self) -> u8 {
        (self.0[1] >> 1) & 3
    }

    fn get_bitrate(self) -> u8 {
        self.0[2] >> 4
    }

    fn get_sample_rate(self) -> u8 {
        (self.0[2] >> 2) & 3
    }

    fn get_my_sample_rate(self) -> u8 {
        self.get_sample_rate() + (((self.0[1] >> 3) & 1) + ((self.0[1] >> 4) & 1)) * 3
    }

    fn is_frame_576(self) -> bool {
        self.0[1] & 14 == 2
    }

    fn is_layer_1(self) -> bool {
        self.0[1] & 6 == 6
    }

    fn is_valid(self) -> bool {
        self.0[0] == 0xff
            && ((self.0[1] & 0xf0) == 0xf0 || (self.0[1] & 0xfe) == 0xe2)
            && self.get_layer() != 0
            && self.get_bitrate() != 15
            && self.get_sample_rate() != 3
    }

    fn compare(self, other: Self) -> bool {
        other.is_valid()
            && (self.0[1] ^ other.0[1]) & 0xfe == 0
            && (self.0[2] ^ other.0[2]) & 0x0c == 0
            && self.is_free_format() == other.is_free_format()
    }

    fn bitrate_kbps(self) -> usize {
        const HALFRATE: [[[u8; 15]; 3]; 2] = [
            [
                [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80],
                [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80],
                [0, 16, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96, 112, 128],
            ],
            [
                [0, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160],
                [
                    0, 16, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192,
                ],
                [
                    0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224,
                ],
            ],
        ];
        2 * HALFRATE[self.test_mpeg1() as usize][self.get_layer() as usize - 1]
            [self.get_bitrate() as usize] as usize
    }

    fn sample_rate_hz(self) -> usize {
        const G_HZ: [usize; 3] = [44100, 48000, 32000];
        G_HZ[self.get_sample_rate() as usize]
            >> !self.test_mpeg1() as usize
            >> !self.test_not_mpeg25() as usize
    }

    fn frame_samples(self) -> usize {
        if self.is_layer_1() {
            384
        } else {
            1152 >> self.is_frame_576() as usize
        }
    }

    fn frame_bytes(self, free_format_size: usize) -> usize {
        let mut frame_bytes =
            self.frame_samples() * self.bitrate_kbps() * 124 / self.sample_rate_hz();
        if self.is_layer_1() {
            frame_bytes &= !3
        }
        if frame_bytes > 0 {
            frame_bytes
        } else {
            free_format_size
        }
    }

    fn padding(self) -> usize {
        if self.test_padding() {
            if self.is_layer_1() { 4 } else { 1 }
        } else {
            0
        }
    }
    // more, see line ~300
}

fn l3_midside_stereo(left_right: &mut [[f32; 576]; 2], n: usize) {
    for i in 0..n {
        let a = left_right[0][i];
        let b = left_right[1][i];
        left_right[0][i] = a + b;
        left_right[1][i] = a - b;
    }
}

fn l3_intensity_stereo_band(left_right: &mut [[f32; 576]; 2], n: usize, kl: f32, kr: f32) {
    for i in 0..n {
        let a = left_right[0][i];
        left_right[0][i] = a * kl;
        left_right[1][i] = a * kr;
    }
}

fn l3_dct_9(y: &mut [f32; 9]) {
    let mut s0 = y[0];
    let mut s2 = y[2];
    let mut s4 = y[4];
    let mut s6 = y[6];
    let mut s8 = y[8];
    let t0 = s0 + s6 * 0.5;
    s0 -= s6;
    let t4 = (s4 + s2) * 0.93969262;
    let t2 = (s8 + s2) * 0.76604444;
    s6 = (s4 - s8) * 0.17364818;
    s4 += s8 - s2;

    s2 = s0 - s4 * 0.5;
    y[4] = s4 + s0;
    s8 = t0 - t2 + s6;
    s0 = t0 - t4 + s2;
    s4 = t0 + t4 - s6;

    let mut s1 = y[1];
    let mut s3 = y[3];
    let mut s5 = y[5];
    let mut s7 = y[7];
    s3 *= 0.86602540;
    let t0 = (s5 + s1) * 0.98480775;
    let t4 = (s5 - s7) * 0.34202014;
    let t2 = (s1 + s7) * 0.64278761;
    s1 = (s1 - s5 - s7) * 0.86602540;

    s5 = t0 - s3 - t2;
    s7 = t4 - s3 - t0;
    s3 = t4 + s3 - t2;

    y[0] = s4 - s7;
    y[1] = s2 + s1;
    y[2] = s0 - s3;
    y[3] = s8 + s5;
    y[5] = s8 - s5;
    y[6] = s0 + s3;
    y[7] = s2 - s1;
    y[8] = s4 + s7;
}

fn l3_imdct36(grbuf: &mut [f32], overlap: &mut [f32], window: &[f32; 18], nbands: usize) {
    for j in 0..nbands {
        let gr_slice = &mut grbuf[j * 18..];
        let overlap_slice = &mut overlap[j * 9..];
        let mut co = [0.0; 9];
        let mut si = [0.0; 9];
        co[0] = -gr_slice[0];
        si[0] = gr_slice[17];
        for i in 0..4 {
            si[8 - 2 * i] = gr_slice[4 * i + 1] - gr_slice[4 * i + 2];
            co[1 + 2 * i] = gr_slice[4 * i + 1] + gr_slice[4 * i + 2];
            si[7 - 2 * i] = gr_slice[4 * i + 4] - gr_slice[4 * i + 3];
            co[2 + 2 * i] = -(gr_slice[4 * i + 3] + gr_slice[4 * i + 4]);
        }
        l3_dct_9(&mut co);
        l3_dct_9(&mut si);

        si[1] = -si[1];
        si[3] = -si[3];
        si[5] = -si[5];
        si[7] = -si[7];
        const G_TWID9: [f32; 18] = [
            0.73727734, 0.79335334, 0.84339145, 0.88701083, 0.92387953, 0.95371695, 0.97629601,
            0.99144486, 0.99904822, 0.67559021, 0.60876143, 0.53729961, 0.46174861, 0.38268343,
            0.30070580, 0.21643961, 0.13052619, 0.04361938,
        ];
        for i in 0..9 {
            let ovl = overlap_slice[i];
            let sum = co[i] * G_TWID9[9 + i] + si[i] * G_TWID9[i];
            overlap_slice[i] = co[i] * G_TWID9[i] - si[i] * G_TWID9[9 + i];
            gr_slice[i] = ovl * window[i] - sum * window[9 + i];
            gr_slice[17 - i] = ovl * window[9 + i] + sum * window[i];
        }
    }
}

fn l3_idct3(x0: f32, x1: f32, x2: f32) -> [f32; 3] {
    let m1 = x1 * 0.86602540;
    let a1 = x0 - x2 * 0.5;
    [a1 + m1, x0 + x2, a1 - m1]
}

fn l3_imdct12(x: &[f32], dst: &mut [f32], overlap: &mut [f32]) {
    let co = l3_idct3(-x[0], x[6] + x[3], x[12] + x[0]);
    let mut si = l3_idct3(x[15], x[12] - x[9], x[6] - x[3]);
    si[1] = -si[1];
    const G_TWID3: [f32; 6] = [
        0.79335334, 0.92387953, 0.99144486, 0.60876143, 0.38268343, 0.13052619,
    ];
    for i in 0..3 {
        let ovl = overlap[i];
        let sum = co[i] * G_TWID3[3 + i] + si[i] * G_TWID3[i];
        overlap[i] = co[i] * G_TWID3[i] - si[i] * G_TWID3[3 + i];
        dst[i] = ovl * G_TWID3[2 - i] - sum * G_TWID3[5 - i];
        dst[5 - i] = ovl * G_TWID3[5 - i] + sum * G_TWID3[2 - i];
    }
}

fn l3_imdct_short(grbuf: &mut [f32], overlap: &mut [f32], nbands: usize) {
    for j in 0..nbands {
        let gr_slice = &mut grbuf[j * 18..];
        let overlap_slice = &mut overlap[j * 9..];
        let mut tmp = [0.0; 18];
        tmp.copy_from_slice(&gr_slice[..18]);
        gr_slice[..6].copy_from_slice(&overlap_slice[..6]);
        l3_imdct12(&tmp, &mut gr_slice[6..], &mut overlap_slice[6..]);
        l3_imdct12(&tmp[1..], &mut gr_slice[12..], &mut overlap_slice[6..]);
        let (over1, over2) = overlap_slice.split_at_mut(6);
        l3_imdct12(&tmp[2..], over1, over2);
    }
}

impl Decoder {
    pub fn new() -> Self {
        Decoder {
            mdct_overlap: [[0.0; 9 * 32]; 2],
            qmf_state: [0.0; 15 * 2 * 32],
            reserv: 0,
            free_format_bytes: 0,
            header: Header::default(),
            reserv_buf: [0; 511],
        }
    }

    fn save_reservoir(&mut self, scratch: &Scratch) {
        let mut pos = scratch.bs.pos.div_ceil(8);
        let mut remains = scratch.bs.limit / 8 - pos;
        if remains > MAX_BITRESERVOIR_BYTES {
            pos += remains - MAX_BITRESERVOIR_BYTES;
            remains = MAX_BITRESERVOIR_BYTES;
        }
        if remains > 0 {
            self.reserv_buf[..remains].copy_from_slice(&scratch.maindata[pos..][..remains]);
        }
        self.reserv = remains;
    }

    fn l3_restore_reservoir(
        &mut self,
        bs: &mut Bs,
        s: &mut Scratch,
        main_data_begin: usize,
    ) -> bool {
        let frame_bytes = (bs.core.limit - bs.core.pos) / 8;
        let bytes_have = self.reserv.min(main_data_begin);
        s.maindata[..bytes_have]
            .copy_from_slice(&self.reserv_buf[self.reserv - bytes_have..self.reserv]);
        s.maindata[bytes_have..][..frame_bytes]
            .copy_from_slice(&bs.buf[bs.core.pos / 8..][..frame_bytes]);
        s.bs = BsCore::new(bytes_have + frame_bytes);
        self.reserv >= main_data_begin
    }

    pub fn decode_frame(
        &mut self,
        mp3: &[u8],
        pcm: Option<&mut [Sample; MAX_SAMPLES_PER_FRAME]>,
        info: &mut FrameInfo,
    ) -> usize {
        let mut frame_size = 0;
        if mp3.len() > 4 && self.header.0[0] == 0xff {
            let header = Header::new(mp3);
            if self.header.compare(header) {
                frame_size = header.frame_bytes(self.free_format_bytes) + header.padding();
                if frame_size != mp3.len()
                    && (frame_size + HDR_SIZE > mp3.len()
                        || !header.compare(Header::new(&mp3[frame_size..])))
                {
                    frame_size = 0;
                }
            }
        }
        let i = if frame_size == 0 {
            *self = Self::new();
            let (i, new_frame_size) = mp3d_find_frame(mp3, &mut self.free_format_bytes);
            if new_frame_size == 0 || i + new_frame_size > mp3.len() {
                info.frame_bytes = i;
                return 0;
            }
            frame_size = new_frame_size;
            i
        } else {
            0
        };
        let header = Header::new(&mp3[i..]);
        self.header = header;
        info.frame_bytes = i + frame_size;
        info.frame_offset = i;
        info.channels = if header.is_mono() { 1 } else { 2 };
        info.hz = header.sample_rate_hz();
        info.layer = 4 - header.get_layer() as usize;
        info.bitrate_kbps = header.bitrate_kbps();
        if pcm.is_none() {
            return header.frame_samples();
        }
        let bs_buf = &mp3[i..][..frame_size][HDR_SIZE..];
        let mut bs_core = BsCore::new(bs_buf.len());
        let mut bs_frame = Bs::new(bs_buf, &mut bs_core);
        if header.is_crc() {
            _ = bs_frame.get_bits(16);
        }
        let mut scratch = Scratch::default(); // TODO: think about bs
        if info.layer == 3 {
            let main_data_begin = l3_read_side_info(&mut bs_frame, &mut scratch.gr_info, header);
            if main_data_begin == L3_ERROR || bs_frame.core.pos > bs_frame.core.limit {
                self.header.0[0] = 0;
                return 0;
            }
            let success = self.l3_restore_reservoir(&mut bs_frame, &mut scratch, main_data_begin);
            if success {
                let n_gr = if header.test_mpeg1() { 2 } else { 1 };
                for igr in 0..n_gr {
                    scratch.grbuf = [[0.0; 576]; 2];
                    self.l3_decode(&mut scratch, igr * info.channels, info.channels);
                    // TODO: mp3d_synth_granule
                }
            }
            self.save_reservoir(&scratch);
            success as usize * self.header.frame_samples()
        } else {
            // optional TODO: implement level 1 & 2
            return 0;
        }
    }

    fn l3_decode(&mut self, scratch: &mut Scratch, igr: usize, nch: usize) {
        for ch in 0..nch {
            let layer3gr_limit = scratch.bs.pos + scratch.gr_info[igr + ch].part_23_length as usize;
            let mut bs = Bs::new(&scratch.maindata, &mut scratch.bs);
            l3_decode_scalefactors(
                self.header,
                &mut scratch.ist_pos[ch],
                &mut bs,
                &scratch.gr_info[igr + ch],
                &mut scratch.scf,
                ch,
            );
            // TODO: l3_huffman
        }
        todo!()
    }
}

impl Default for Scratch {
    fn default() -> Self {
        todo!()
    }
}

fn mp3d_match_frame(mp3: &[u8], frame_bytes: usize) -> bool {
    let header = Header::new(mp3);
    let mut i = 0;
    for nmatch in 0..MAX_FRAME_SYNC_MATCHES {
        let this = Header::new(&mp3[i..]);
        i += this.frame_bytes(frame_bytes) + this.padding();
        if i + HDR_SIZE > mp3.len() {
            return nmatch > 0;
        }
        if !header.compare(this) {
            return false;
        }
    }
    true
}

/// Returns offset of frame start and frame_bytes
fn mp3d_find_frame(mp3: &[u8], free_format_bytes: &mut usize) -> (usize, usize) {
    for i in 0..mp3.len() - HDR_SIZE {
        let header = Header::new(&mp3[i..]);
        if header.is_valid() {
            let mut frame_bytes = header.frame_bytes(*free_format_bytes);
            let mut frame_and_padding = frame_bytes + header.padding();
            if frame_bytes == 0 {
                for k in HDR_SIZE..MAX_FREE_FORMAT_FRAME_SIZE {
                    if i + 2 * k < mp3.len() - HDR_SIZE {
                        break;
                    }
                    let other = Header::new(&mp3[i + k..]);
                    if header.compare(other) {
                        let fb = k - header.padding();
                        let nextfb = fb + other.padding();
                        if i + k + nextfb + HDR_SIZE > mp3.len()
                            || !header.compare(Header::new(&mp3[i + k + nextfb..]))
                        {
                            continue;
                        }
                        frame_and_padding = k;
                        frame_bytes = fb;
                        *free_format_bytes = fb;
                        if frame_bytes != 0 {
                            break;
                        }
                    }
                }
            }
            if (frame_bytes != 0
                && i + frame_and_padding <= mp3.len()
                && mp3d_match_frame(&mp3[i..], frame_bytes))
                || (i == 0 && frame_and_padding == mp3.len())
            {
                return (i, frame_and_padding);
            }
            *free_format_bytes = 0;
        }
    }
    (mp3.len(), 0)
}

const G_SCF_LONG: [[u8; 23]; 8] = [
    [
        6, 6, 6, 6, 6, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 38, 46, 52, 60, 68, 58, 54, 0,
    ],
    [
        12, 12, 12, 12, 12, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 76, 90, 2, 2, 2, 2, 2, 0,
    ],
    [
        6, 6, 6, 6, 6, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 38, 46, 52, 60, 68, 58, 54, 0,
    ],
    [
        6, 6, 6, 6, 6, 6, 8, 10, 12, 14, 16, 18, 22, 26, 32, 38, 46, 54, 62, 70, 76, 36, 0,
    ],
    [
        6, 6, 6, 6, 6, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 38, 46, 52, 60, 68, 58, 54, 0,
    ],
    [
        4, 4, 4, 4, 4, 4, 6, 6, 8, 8, 10, 12, 16, 20, 24, 28, 34, 42, 50, 54, 76, 158, 0,
    ],
    [
        4, 4, 4, 4, 4, 4, 6, 6, 6, 8, 10, 12, 16, 18, 22, 28, 34, 40, 46, 54, 54, 192, 0,
    ],
    [
        4, 4, 4, 4, 4, 4, 6, 6, 8, 10, 12, 16, 20, 24, 30, 38, 46, 56, 68, 84, 102, 26, 0,
    ],
];
const G_SCF_SHORT: [[u8; 40]; 8] = [
    [
        4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14, 14, 18, 18,
        18, 24, 24, 24, 30, 30, 30, 40, 40, 40, 18, 18, 18, 0,
    ],
    [
        8, 8, 8, 8, 8, 8, 8, 8, 8, 12, 12, 12, 16, 16, 16, 20, 20, 20, 24, 24, 24, 28, 28, 28, 36,
        36, 36, 2, 2, 2, 2, 2, 2, 2, 2, 2, 26, 26, 26, 0,
    ],
    [
        4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 8, 8, 8, 10, 10, 10, 14, 14, 14, 18, 18, 18,
        26, 26, 26, 32, 32, 32, 42, 42, 42, 18, 18, 18, 0,
    ],
    [
        4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14, 14, 18, 18,
        18, 24, 24, 24, 32, 32, 32, 44, 44, 44, 12, 12, 12, 0,
    ],
    [
        4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14, 14, 18, 18,
        18, 24, 24, 24, 30, 30, 30, 40, 40, 40, 18, 18, 18, 0,
    ],
    [
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14, 14,
        18, 18, 18, 22, 22, 22, 30, 30, 30, 56, 56, 56, 0,
    ],
    [
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 10, 10, 10, 12, 12, 12, 14, 14, 14,
        16, 16, 16, 20, 20, 20, 26, 26, 26, 66, 66, 66, 0,
    ],
    [
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 8, 8, 8, 12, 12, 12, 16, 16, 16, 20, 20, 20,
        26, 26, 26, 34, 34, 34, 42, 42, 42, 12, 12, 12, 0,
    ],
];
const G_SCF_MIXED: [[u8; 40]; 8] = [
    [
        6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14, 14, 18, 18, 18, 24, 24,
        24, 30, 30, 30, 40, 40, 40, 18, 18, 18, 0, 0, 0, 0,
    ],
    [
        12, 12, 12, 4, 4, 4, 8, 8, 8, 12, 12, 12, 16, 16, 16, 20, 20, 20, 24, 24, 24, 28, 28, 28,
        36, 36, 36, 2, 2, 2, 2, 2, 2, 2, 2, 2, 26, 26, 26, 0,
    ],
    [
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 10, 10, 10, 14, 14, 14, 18, 18, 18, 26, 26,
        26, 32, 32, 32, 42, 42, 42, 18, 18, 18, 0, 0, 0, 0,
    ],
    [
        6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14, 14, 18, 18, 18, 24, 24,
        24, 32, 32, 32, 44, 44, 44, 12, 12, 12, 0, 0, 0, 0,
    ],
    [
        6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14, 14, 18, 18, 18, 24, 24,
        24, 30, 30, 30, 40, 40, 40, 18, 18, 18, 0, 0, 0, 0,
    ],
    [
        4, 4, 4, 4, 4, 4, 6, 6, 4, 4, 4, 6, 6, 6, 8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14, 14, 18,
        18, 18, 22, 22, 22, 30, 30, 30, 56, 56, 56, 0, 0,
    ],
    [
        4, 4, 4, 4, 4, 4, 6, 6, 4, 4, 4, 6, 6, 6, 6, 6, 6, 10, 10, 10, 12, 12, 12, 14, 14, 14, 16,
        16, 16, 20, 20, 20, 26, 26, 26, 66, 66, 66, 0, 0,
    ],
    [
        4, 4, 4, 4, 4, 4, 6, 6, 4, 4, 4, 6, 6, 6, 8, 8, 8, 12, 12, 12, 16, 16, 16, 20, 20, 20, 26,
        26, 26, 34, 34, 34, 42, 42, 42, 12, 12, 12, 0, 0,
    ],
];

const L3_ERROR: usize = !0;

// Maybe returning Option would be more idiomatic than having `L3_ERROR` return.
fn l3_read_side_info(bs: &mut Bs, grs: &mut [GrInfo; 4], header: Header) -> usize {
    let sr_idx = header.get_my_sample_rate().saturating_sub(1);
    let mut gr_count = if header.is_mono() { 1 } else { 2 };

    let mut scfsi = 0;
    let main_data_begin = if header.test_mpeg1() {
        gr_count *= 2;
        let main_data_begin = bs.get_bits(9);
        scfsi = bs.get_bits(7 + gr_count);
        main_data_begin
    } else {
        bs.get_bits(8 + gr_count) >> gr_count
    };

    let mut part_23_sum = 0;
    for gr in &mut grs[..gr_count] {
        if header.is_mono() {
            scfsi <<= 4;
        }
        gr.part_23_length = bs.get_bits(12) as u16;
        part_23_sum += gr.part_23_length as usize;
        gr.big_values = bs.get_bits(9) as u16;
        if gr.big_values > 288 {
            return L3_ERROR;
        }
        gr.global_gain = bs.get_bits(8) as u8;
        gr.scalefac_compress = bs.get_bits(if header.test_mpeg1() { 4 } else { 9 }) as u16;
        gr.sfbtab = &G_SCF_LONG[sr_idx as usize];
        gr.n_long_sfb = 22;
        gr.n_short_sfb = 0;
        let tables;
        if bs.get_bits(1) != 0 {
            gr.block_type = bs.get_bits(2) as u8;
            if gr.block_type == 0 {
                return L3_ERROR;
            }
            gr.mixed_block_flag = bs.get_bits(1) as u8;
            gr.region_count[0] = 7;
            gr.region_count[1] = 255;
            if gr.block_type == SHORT_BLOCK_TYPE {
                scfsi &= 0xf0f;
                if gr.mixed_block_flag == 0 {
                    gr.region_count[0] = 8;
                    gr.sfbtab = &G_SCF_SHORT[sr_idx as usize];
                    gr.n_long_sfb = 0;
                    gr.n_short_sfb = 39;
                } else {
                    gr.sfbtab = &G_SCF_MIXED[sr_idx as usize];
                    gr.n_long_sfb = if header.test_mpeg1() { 8 } else { 6 };
                    gr.n_short_sfb = 30;
                }
            }
            tables = bs.get_bits(10) << 5;
            gr.subblock_gain[0] = bs.get_bits(3) as u8;
            gr.subblock_gain[1] = bs.get_bits(3) as u8;
            gr.subblock_gain[2] = bs.get_bits(3) as u8;
        } else {
            gr.block_type = 0;
            gr.mixed_block_flag = 0;
            tables = bs.get_bits(15);
            gr.region_count[0] = bs.get_bits(4) as u8;
            gr.region_count[1] = bs.get_bits(3) as u8;
            gr.region_count[2] = 255;
        }
        gr.table_select[0] = (tables >> 10) as u8;
        gr.table_select[1] = (tables >> 5) as u8 & 31;
        gr.table_select[2] = tables as u8 & 31;
        gr.preflag = if header.test_mpeg1() {
            bs.get_bits(1) as u8
        } else {
            (gr.scalefac_compress >= 500) as u8
        };
        gr.scalefac_scale = bs.get_bits(1) as u8;
        gr.count1_table = bs.get_bits(1) as u8;
        gr.scfsi = (scfsi >> 12) as u8 & 15;
        scfsi <<= 4;
    }
    if part_23_sum + bs.core.pos > bs.core.limit + main_data_begin as usize * 8 {
        return L3_ERROR;
    }
    main_data_begin as usize
}

fn l3_read_scalefactors(
    scf: &mut [u8],
    ist_pos: &mut [u8],
    scf_size: &mut [u8],
    scf_count: &[u8],
    bitbuf: &mut Bs,
    scfsi: i32,
) {
    let mut ix = 0;
    for i in 0..4 {
        let cnt = scf_count[i] as usize;
        if cnt == 0 {
            break;
        }
        let this_scfsi = scfsi << i;
        if (this_scfsi & 8) != 0 {
            scf[ix..][..cnt].copy_from_slice(&ist_pos[ix..][..cnt]);
        } else {
            let bits = scf_size[i];
            if bits != 0 {
                scf[ix..][..cnt].fill(0);
                ist_pos[ix..][..cnt].fill(0);
            } else {
                let max_scf = if this_scfsi < 0 { (1 << bits) - 1 } else { !0 };
                for k in 0..cnt {
                    let s = bitbuf.get_bits(bits as usize) as u8;
                    ist_pos[ix + k] = if s == max_scf { !0 } else { s };
                    scf[ix + k] = s;
                }
            }
        }
        ix += cnt;
    }
    scf[ix..][..3].fill(0);
}

fn l3_ldexp_q2(mut y: f32, mut exp_q2: i32) -> f32 {
    const G_EXPFRAC: [f32; 4] = [
        9.31322575e-10,
        7.83145814e-10,
        6.58544508e-10,
        5.53767716e-10,
    ];
    loop {
        let e = exp_q2.min(30 * 4);
        y *= G_EXPFRAC[(e & 3) as usize] * (1 << 30 >> (e >> 2)) as f32;
        exp_q2 -= e;
        if exp_q2 <= 0 {
            return y;
        }
    }
}

fn l3_decode_scalefactors(
    header: Header,
    ist_pos: &mut [u8],
    bs: &mut Bs,
    gr: &GrInfo,
    scf: &mut [f32],
    ch: usize,
) {
    const G_SCF_PARTITIONS: [[u8; 28]; 3] = [
        [
            6, 5, 5, 5, 6, 5, 5, 5, 6, 5, 7, 3, 11, 10, 0, 0, 7, 7, 7, 0, 6, 6, 6, 3, 8, 8, 5, 0,
        ],
        [
            8, 9, 6, 12, 6, 9, 9, 9, 6, 9, 12, 6, 15, 18, 0, 0, 6, 15, 12, 0, 6, 12, 9, 6, 6, 18,
            9, 0,
        ],
        [
            9, 9, 6, 12, 9, 9, 9, 9, 9, 9, 12, 6, 18, 18, 0, 0, 12, 12, 12, 0, 12, 9, 9, 6, 15, 12,
            9, 0,
        ],
    ];
    let scf_ix = (gr.n_short_sfb != 0) as usize + (gr.n_long_sfb == 0) as usize;
    let scf_partition = &G_SCF_PARTITIONS[scf_ix];
    let scf_shift = gr.scalefac_scale + 1;
    let mut scfsi = gr.scfsi as i32;

    let mut scf_size = [0u8; 4];
    let mut iscf = [0u8; 40];

    let mut scf_partition_ix = 0;
    if header.test_mpeg1() {
        const G_SCFC_DECODE: [u8; 16] = [0, 1, 2, 3, 12, 5, 6, 7, 9, 10, 11, 13, 14, 15, 18, 19];
        let part = G_SCFC_DECODE[gr.scalefac_compress as usize];
        scf_size[0] = part >> 2;
        scf_size[1] = part >> 2;
        scf_size[2] = part & 3;
        scf_size[3] = part & 3;
    } else {
        const G_MOD: [u8; 24] = [
            5, 5, 4, 4, 5, 5, 4, 1, 4, 3, 1, 1, 5, 6, 6, 1, 4, 4, 4, 1, 4, 3, 1, 1,
        ];
        let ist = header.test_i_stereo() && ch != 0;
        let mut sfc = (gr.scalefac_compress >> ist as u32) as i32;
        let mut k = ist as usize * 3 * 4;
        while sfc >= 0 {
            let mut modprod = 1;
            for i in (0..4).rev() {
                let g_mod = G_MOD[k + i] as i32;
                scf_size[i] = (sfc / modprod % g_mod) as u8;
                modprod *= g_mod;
            }
            sfc -= modprod;
            k += 4;
        }
        scf_partition_ix = k;
        scfsi = -16;
    }
    l3_read_scalefactors(
        &mut iscf,
        ist_pos,
        &mut scf_size,
        &scf_partition[scf_partition_ix..],
        bs,
        scfsi,
    );

    if gr.n_short_sfb != 0 {
        let sh = 3 - scf_shift;
        for i in (0..gr.n_short_sfb as usize).step_by(3) {
            iscf[gr.n_long_sfb as usize + i] += gr.subblock_gain[0] << sh;
            iscf[gr.n_long_sfb as usize + i + 1] += gr.subblock_gain[1] << sh;
            iscf[gr.n_long_sfb as usize + i + 2] += gr.subblock_gain[2] << sh;
        }
    } else if gr.preflag != 0 {
        const G_PREAMP: [u8; 10] = [1, 1, 1, 1, 2, 2, 3, 3, 3, 2];
        for i in 0..10 {
            iscf[11 + i] = G_PREAMP[i];
        }
    }

    let gain_exp = gr.global_gain as i32 + BITS_DEQUANTIZER_OUT * 4
        - 210
        - if header.is_ms_stereo() { 2 } else { 0 };
    let gain = l3_ldexp_q2((1 << (MAX_SCFI / 4)) as f32, MAX_SCFI - gain_exp);
    for i in 0..(gr.n_long_sfb + gr.n_short_sfb) as usize {
        scf[i] = l3_ldexp_q2(gain, (iscf[i] as i32) << scf_shift);
    }
}

fn mp3d_scale_pcm(sample: f32) -> Sample {
    // Should be correct for [-1.5..-0.5], where minimp3 produces 0
    let y = sample.clamp(-32767.5, 32766.5) + 0.5;
    (y as i16) - (y < 0.) as i16
}

fn mp3d_synth_pair(pcm: &mut [Sample], nch: usize, z: &[f32]) {
    let mut a = (z[14 * 64] - z[0]) * 29.;
    a += (z[1 * 64] + z[13 * 64]) * 213.;
    a += (z[12 * 64] - z[2 * 64]) * 459.;
    a += (z[3 * 64] + z[11 * 64]) * 2037.;
    a += (z[10 * 64] - z[4 * 64]) * 5153.;
    a += (z[5 * 64] + z[9 * 64]) * 6574.;
    a += (z[8 * 64] - z[6 * 64]) * 37489.;
    a += z[7 * 64] * 75038.;
    pcm[0] = mp3d_scale_pcm(a);

    a = z[14 * 64 + 2] * 104.;
    a += z[12 * 64 + 2] * 1567.;
    a += z[10 * 64 + 2] * 9727.;
    a += z[8 * 64 + 2] * 64019.;
    a += z[6 * 64 + 2] * -9975.;
    a += z[4 * 64 + 2] * -45.;
    a += z[2 * 64 + 2] * 146.;
    a += z[0 * 64 + 2] * -5.;
    pcm[16 * nch] = mp3d_scale_pcm(a);
}

fn mp3d_synth(grbuf: &[[f32; 576]], dstl: &mut [Sample], nch: usize, lins: &mut [f32]) {
    const G_WIN: &[f32] = &[
        -1., 26., -31., 208., 218., 401., -519., 2063., 2000., 4788., -5517., 7134., 5959., 35640.,
        -39336., 74992., -1., 24., -35., 202., 222., 347., -581., 2080., 1952., 4425., -5879.,
        7640., 5288., 33791., -41176., 74856., -1., 21., -38., 196., 225., 294., -645., 2087.,
        1893., 4063., -6237., 8092., 4561., 31947., -43006., 74630., -1., 19., -41., 190., 227.,
        244., -711., 2085., 1822., 3705., -6589., 8492., 3776., 30112., -44821., 74313., -1., 17.,
        -45., 183., 228., 197., -779., 2075., 1739., 3351., -6935., 8840., 2935., 28289., -46617.,
        73908., -1., 16., -49., 176., 228., 153., -848., 2057., 1644., 3004., -7271., 9139., 2037.,
        26482., -48390., 73415., -2., 14., -53., 169., 227., 111., -919., 2032., 1535., 2663.,
        -7597., 9389., 1082., 24694., -50137., 72835., -2., 13., -58., 161., 224., 72., -991.,
        2001., 1414., 2330., -7910., 9592., 70., 22929., -51853., 72169., -2., 11., -63., 154.,
        221., 36., -1064., 1962., 1280., 2006., -8209., 9750., -998., 21189., -53534., 71420., -2.,
        10., -68., 147., 215., 2., -1137., 1919., 1131., 1692., -8491., 9863., -2122., 19478.,
        -55178., 70590., -3., 9., -73., 139., 208., -29., -1210., 1870., 970., 1388., -8755.,
        9935., -3300., 17799., -56778., 69679., -3., 8., -79., 132., 200., -57., -1283., 1817.,
        794., 1095., -8998., 9966., -4533., 16155., -58333., 68692., -4., 7., -85., 125., 189.,
        -83., -1356., 1759., 605., 814., -9219., 9959., -5818., 14548., -59838., 67629., -4., 7.,
        -91., 117., 177., -106., -1428., 1698., 402., 545., -9416., 9916., -7154., 12980., -61289.,
        66494., -5., 6., -97., 111., 163., -127., -1498., 1634., 185., 288., -9585., 9838., -8540.,
        11455., -62684., 65290.,
    ];
    let right_ix = nch - 1;
    const Z_OFF: usize = 15 * 64;
    lins[Z_OFF + 4 * 15] = grbuf[0][18 * 16];
    lins[Z_OFF + 4 * 15 + 1] = grbuf[right_ix][18 * 16];
    lins[Z_OFF + 4 * 15 + 2] = grbuf[0][0];
    lins[Z_OFF + 4 * 15 + 3] = grbuf[right_ix][0];

    lins[Z_OFF + 4 * 31] = grbuf[0][1 + 18 * 16];
    lins[Z_OFF + 4 * 31 + 1] = grbuf[right_ix][1 + 18 * 16];
    lins[Z_OFF + 4 * 31 + 2] = grbuf[0][1];
    lins[Z_OFF + 4 * 31 + 3] = grbuf[right_ix][1];

    mp3d_synth_pair(&mut dstl[right_ix..], nch, &lins[4 * 15 + 1..]);
    mp3d_synth_pair(
        &mut dstl[right_ix + 32 * nch..],
        nch,
        &lins[4 * 15 + 64 + 1..],
    );
    mp3d_synth_pair(dstl, nch, &lins[4 * 15..]);
    mp3d_synth_pair(&mut dstl[32 * nch..], nch, &lins[4 * 15 * 64..]);

    for i in (0..15).rev() {
        let ix = Z_OFF + 4 * i;
        lins[ix] = grbuf[0][18 * (31 - i)];
        lins[ix + 1] = grbuf[right_ix][18 * (31 - i)];
        lins[ix + 2] = grbuf[0][1 + 18 * (31 - i)];
        lins[ix + 3] = grbuf[right_ix][1 + 18 * (31 - i)];
        lins[ix + 64] = grbuf[0][1 + 18 * (1 + i)];
        lins[ix + 65] = grbuf[right_ix][1 + 18 * (1 + i)];
        lins[ix - 62] = grbuf[0][18 * (1 + i)];
        lins[ix - 61] = grbuf[right_ix][18 * (1 + i)];
        let mut a = [0.0; 4];
        let mut b = [0.0; 4];
        for k in 0..4 {
            let w0 = G_WIN[(14 - i) * 16 + k * 4];
            let w1 = G_WIN[(14 - i) * 16 + k * 4 + 1];
            let z_ix = ix - k * 2 * 64;
            let y_ix = z_ix - (15 - k * 2) * 64;
            for j in 0..4 {
                b[j] += lins[z_ix + j] * w1 + lins[y_ix + j] * w0;
                a[j] += lins[z_ix + j] * w0 - lins[y_ix + j] * w1;
            }
            let w2 = G_WIN[(14 - i) * 16 + k * 4 + 2];
            let w3 = G_WIN[(14 - i) * 16 + k * 4 + 3];
            for j in 0..4 {
                b[j] += lins[z_ix - 64 + j] * w3 + lins[y_ix + 64 + j] * w2;
                a[j] += lins[z_ix - 64 + j] * w3 - lins[y_ix + 64 + j] * w2;
            }
        }

        dstl[right_ix + (15 - i) * nch] = mp3d_scale_pcm(a[1]);
        dstl[right_ix + (17 + i) * nch] = mp3d_scale_pcm(b[1]);
        dstl[(15 - i) * nch] = mp3d_scale_pcm(a[0]);
        dstl[(17 + i) * nch] = mp3d_scale_pcm(b[0]);
        dstl[right_ix + (47 - i) * nch] = mp3d_scale_pcm(a[3]);
        dstl[right_ix + (49 + i) * nch] = mp3d_scale_pcm(b[3]);
        dstl[(47 - i) * nch] = mp3d_scale_pcm(a[2]);
        dstl[(49 + i) * nch] = mp3d_scale_pcm(b[2]);
    }
}
