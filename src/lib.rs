const MAX_SAMPLES_PER_FRAME: usize = 1152 * 2;

#[derive(Default)]
struct FrameInfo {
    frame_bytes: usize,
    frame_offset: usize,
    channels: usize,
    hz: usize,
    layer: usize,
    bitrate_kbps: usize,
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
const HDR_SIZE: usize = 4;

struct ScaleInfo {
    scf: [f32; 3 * 64],
    total_bands: u8,
    stereo_bands: u8,
    bitalloc: [u8; 64],
    scfcod: [u8; 64],
}

struct GrInfo {
    sfbtab: &'static [u8; 40],
    part_23_length: u16,
    big_values: u16,
    scalefac_compress: u16,
    global_gain: u8,
    block_type: u8,
    mixed_blog_flag: u8,
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

// May rework this so caller always passes down buffer
struct Bs<'a> {
    buf: &'a [u8],
    pos: usize,
    limit: usize,
}

struct Scratch<'a> {
    bs: Bs<'a>,
    maindata: [u8; MAX_BITRESERVOIR_BYTES + MAX_L3_FRAME_PAYLOAD_BYTES],
    gr_info: GrInfo,
    grbuf: [[f32; 576]; 2],
    scr: [f32; 40],
    syn: [[f32; 2 * 32]; 18 + 15],
    ist_pos: [[u8; 39]; 2],
}

impl<'a> Bs<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            pos: 0,
            limit: buf.len() * 8,
        }
    }

    fn get_bits(&mut self, n: usize) -> u32 {
        let s = self.pos % 8;
        let mut shl = n + s;
        self.pos += n;
        if self.pos > self.limit {
            return 0;
        }
        let mut cache = 0;
        let mut ix = self.pos / 8;
        let mut next = self.buf[ix] as u32 & (0xff >> s);
        while shl > 8 {
            shl -= 8;
            cache |= next << shl;
            ix += 1;
            next = self.buf[ix] as u32;
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

    // more (see line 66-72)
    fn get_layer(self) -> u8 {
        (self.0[1] >> 1) & 3
    }

    fn get_bitrate(self) -> u8 {
        self.0[2] >> 4
    }

    fn get_sample_rate(self) -> u8 {
        (self.0[2] >> 2) & 3
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

fn l3_midside_stereo(left_right: &mut [f32; 576 * 2], n: usize) {
    for i in 0..n {
        let a = left_right[i];
        let b = left_right[i + 576];
        left_right[i] = a + b;
        left_right[i + 576] = a - b;
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

// Returns offset of frame start and frame_bytes
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
