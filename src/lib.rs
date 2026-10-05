// Copyright 2025 Raph Levien
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![cfg_attr(all(not(feature = "dump"), not(test)), no_std)]

mod bitstream;
#[cfg(feature = "dump")]
mod dump;

#[cfg(feature = "dump")]
macro_rules! dump {
    ($name: expr, $grbuf: expr) => {
        crate::dump::dump($name, $grbuf);
    };
}

#[cfg(not(feature = "dump"))]
macro_rules! dump {
    ($name: expr, $grbuf: expr) => {
    };
}

use bitstream::{Bs, BsCache, BsCore};

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

// This is only used for Layer 1/2
#[expect(unused)]
struct ScaleInfo {
    scf: [f32; 3 * 64],
    total_bands: u8,
    stereo_bands: u8,
    bitalloc: [u8; 64],
    scfcod: [u8; 64],
}

#[derive(Default, Debug)]
struct GrInfo {
    sfbtab: &'static [u8],
    part_23_length: u16,
    big_values: u16,
    scalefac_compress: u16,
    global_gain: u8,
    block_type: u8,
    mixed_block_flag: bool,
    n_long_sfb: u8,
    n_short_sfb: u8,
    table_select: [u8; 3],
    region_count: [u8; 3],
    subblock_gain: [u8; 3],
    preflag: bool,
    scalefac_scale: u8,
    count1_table: bool,
    scfsi: u8,
}

// This is only used for Layer 1/2
#[expect(unused)]
struct SubbandAlloc {
    tab_offset: u8,
    code_tab_width: u8,
    band_count: u8,
}

struct Scratch {
    bs: BsCore,
    maindata: [u8; MAX_BITRESERVOIR_BYTES + MAX_L3_FRAME_PAYLOAD_BYTES],
    gr_info: [GrInfo; 4],
    grbuf: [[f32; 576]; 2],
    scf: [f32; 40],
    syn: [f32; 2 * 32 * (18 + 15)],
    ist_pos: [[u8; 39]; 2],
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
    // This is only used for Layer 1/2
    #[expect(unused)]
    fn get_stereo_mode(self) -> u8 {
        (self.0[3] >> 6) & 3
    }

    // This is only used for Layer 1/2
    #[expect(unused)]
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
            self.frame_samples() * self.bitrate_kbps() * 125 / self.sample_rate_hz();
        if self.is_layer_1() {
            frame_bytes &= !3;
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

// TODO: maybe get rid of weird 16 negative values.
const G_POW43: [f32; 129 + 16] = [
    0., -1., -2.519842, -4.326749, -6.349604, -8.549880, -10.902724, -13.390518, -16.000000,
    -18.720754, -21.544347, -24.463781, -27.473142, -30.567351, -33.741992, -36.993181, 0., 1.,
    2.519842, 4.326749, 6.349604, 8.549880, 10.902724, 13.390518, 16.000000, 18.720754, 21.544347,
    24.463781, 27.473142, 30.567351, 33.741992, 36.993181, 40.317474, 43.711787, 47.173345,
    50.699631, 54.288352, 57.937408, 61.644865, 65.408941, 69.227979, 73.100443, 77.024898,
    81.000000, 85.024491, 89.097188, 93.216975, 97.382800, 101.593667, 105.848633, 110.146801,
    114.487321, 118.869381, 123.292209, 127.755065, 132.257246, 136.798076, 141.376907, 145.993119,
    150.646117, 155.335327, 160.060199, 164.820202, 169.614826, 174.443577, 179.305980, 184.201575,
    189.129918, 194.090580, 199.083145, 204.107210, 209.162385, 214.248292, 219.364564, 224.510845,
    229.686789, 234.892058, 240.126328, 245.389280, 250.680604, 256.000000, 261.347174, 266.721841,
    272.123723, 277.552547, 283.008049, 288.489971, 293.998060, 299.532071, 305.091761, 310.676898,
    316.287249, 321.922592, 327.582707, 333.267377, 338.976394, 344.709550, 350.466646, 356.247482,
    362.051866, 367.879608, 373.730522, 379.604427, 385.501143, 391.420496, 397.362314, 403.326427,
    409.312672, 415.320884, 421.350905, 427.402579, 433.475750, 439.570269, 445.685987, 451.822757,
    457.980436, 464.158883, 470.357960, 476.577530, 482.817459, 489.077615, 495.357868, 501.658090,
    507.978156, 514.317941, 520.677324, 527.056184, 533.454404, 539.871867, 546.308458, 552.764065,
    559.238575, 565.731879, 572.243870, 578.774440, 585.323483, 591.890898, 598.476581, 605.080431,
    611.702349, 618.342238, 625.000000, 631.675540, 638.368763, 645.079578,
];

/// Compute approximation to (x as f32).powf(4. / 3.)
///
/// Works for values in -16..8224
fn l3_pow_43(x: i32) -> f32 {
    if x < 129 {
        return G_POW43[(16 + x) as usize];
    }
    let (mult, xs) = if x < 1024 { (16., x << 3) } else { (256., x) };
    let sign = (2 * xs) & 64;
    let frac = ((xs & 63) - sign) as f32 / ((xs & !63) + sign) as f32;
    let scale = (1. + frac * (4. / 3. + frac * (2. / 9.))) * mult;
    G_POW43[16 + ((xs + sign) >> 6) as usize] * scale
}

fn l3_huffman(dst: &mut [f32], bs: &mut Bs, gr: &GrInfo, scf: &[f32], layer3gr_limit: usize) {
    const TABS: [i16; 2164] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 785, 785, 785, 785, 784, 784, 784, 784, 513, 513, 513, 513, 513, 513, 513, 513, 256,
        256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, -255, 1313,
        1298, 1282, 785, 785, 785, 785, 784, 784, 784, 784, 769, 769, 769, 769, 256, 256, 256, 256,
        256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 290, 288, -255, 1313, 1298,
        1282, 769, 769, 769, 769, 529, 529, 529, 529, 529, 529, 529, 529, 528, 528, 528, 528, 528,
        528, 528, 528, 512, 512, 512, 512, 512, 512, 512, 512, 290, 288, -253, -318, -351, -367,
        785, 785, 785, 785, 784, 784, 784, 784, 769, 769, 769, 769, 256, 256, 256, 256, 256, 256,
        256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 819, 818, 547, 547, 275, 275, 275, 275,
        561, 560, 515, 546, 289, 274, 288, 258, -254, -287, 1329, 1299, 1314, 1312, 1057, 1057,
        1042, 1042, 1026, 1026, 784, 784, 784, 784, 529, 529, 529, 529, 529, 529, 529, 529, 769,
        769, 769, 769, 768, 768, 768, 768, 563, 560, 306, 306, 291, 259, -252, -413, -477, -542,
        1298, -575, 1041, 1041, 784, 784, 784, 784, 769, 769, 769, 769, 256, 256, 256, 256, 256,
        256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, -383, -399, 1107, 1092, 1106, 1061,
        849, 849, 789, 789, 1104, 1091, 773, 773, 1076, 1075, 341, 340, 325, 309, 834, 804, 577,
        577, 532, 532, 516, 516, 832, 818, 803, 816, 561, 561, 531, 531, 515, 546, 289, 289, 288,
        258, -252, -429, -493, -559, 1057, 1057, 1042, 1042, 529, 529, 529, 529, 529, 529, 529,
        529, 784, 784, 784, 784, 769, 769, 769, 769, 512, 512, 512, 512, 512, 512, 512, 512, -382,
        1077, -415, 1106, 1061, 1104, 849, 849, 789, 789, 1091, 1076, 1029, 1075, 834, 834, 597,
        581, 340, 340, 339, 324, 804, 833, 532, 532, 832, 772, 818, 803, 817, 787, 816, 771, 290,
        290, 290, 290, 288, 258, -253, -349, -414, -447, -463, 1329, 1299, -479, 1314, 1312, 1057,
        1057, 1042, 1042, 1026, 1026, 785, 785, 785, 785, 784, 784, 784, 784, 769, 769, 769, 769,
        768, 768, 768, 768, -319, 851, 821, -335, 836, 850, 805, 849, 341, 340, 325, 336, 533, 533,
        579, 579, 564, 564, 773, 832, 578, 548, 563, 516, 321, 276, 306, 291, 304, 259, -251, -572,
        -733, -830, -863, -879, 1041, 1041, 784, 784, 784, 784, 769, 769, 769, 769, 256, 256, 256,
        256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, -511, -527, -543, 1396,
        1351, 1381, 1366, 1395, 1335, 1380, -559, 1334, 1138, 1138, 1063, 1063, 1350, 1392, 1031,
        1031, 1062, 1062, 1364, 1363, 1120, 1120, 1333, 1348, 881, 881, 881, 881, 375, 374, 359,
        373, 343, 358, 341, 325, 791, 791, 1123, 1122, -703, 1105, 1045, -719, 865, 865, 790, 790,
        774, 774, 1104, 1029, 338, 293, 323, 308, -799, -815, 833, 788, 772, 818, 803, 816, 322,
        292, 307, 320, 561, 531, 515, 546, 289, 274, 288, 258, -251, -525, -605, -685, -765, -831,
        -846, 1298, 1057, 1057, 1312, 1282, 785, 785, 785, 785, 784, 784, 784, 784, 769, 769, 769,
        769, 512, 512, 512, 512, 512, 512, 512, 512, 1399, 1398, 1383, 1367, 1382, 1396, 1351,
        -511, 1381, 1366, 1139, 1139, 1079, 1079, 1124, 1124, 1364, 1349, 1363, 1333, 882, 882,
        882, 882, 807, 807, 807, 807, 1094, 1094, 1136, 1136, 373, 341, 535, 535, 881, 775, 867,
        822, 774, -591, 324, 338, -671, 849, 550, 550, 866, 864, 609, 609, 293, 336, 534, 534, 789,
        835, 773, -751, 834, 804, 308, 307, 833, 788, 832, 772, 562, 562, 547, 547, 305, 275, 560,
        515, 290, 290, -252, -397, -477, -557, -622, -653, -719, -735, -750, 1329, 1299, 1314,
        1057, 1057, 1042, 1042, 1312, 1282, 1024, 1024, 785, 785, 785, 785, 784, 784, 784, 784,
        769, 769, 769, 769, -383, 1127, 1141, 1111, 1126, 1140, 1095, 1110, 869, 869, 883, 883,
        1079, 1109, 882, 882, 375, 374, 807, 868, 838, 881, 791, -463, 867, 822, 368, 263, 852,
        837, 836, -543, 610, 610, 550, 550, 352, 336, 534, 534, 865, 774, 851, 821, 850, 805, 593,
        533, 579, 564, 773, 832, 578, 578, 548, 548, 577, 577, 307, 276, 306, 291, 516, 560, 259,
        259, -250, -2107, -2507, -2764, -2909, -2974, -3007, -3023, 1041, 1041, 1040, 1040, 769,
        769, 769, 769, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
        256, -767, -1052, -1213, -1277, -1358, -1405, -1469, -1535, -1550, -1582, -1614, -1647,
        -1662, -1694, -1726, -1759, -1774, -1807, -1822, -1854, -1886, 1565, -1919, -1935, -1951,
        -1967, 1731, 1730, 1580, 1717, -1983, 1729, 1564, -1999, 1548, -2015, -2031, 1715, 1595,
        -2047, 1714, -2063, 1610, -2079, 1609, -2095, 1323, 1323, 1457, 1457, 1307, 1307, 1712,
        1547, 1641, 1700, 1699, 1594, 1685, 1625, 1442, 1442, 1322, 1322, -780, -973, -910, 1279,
        1278, 1277, 1262, 1276, 1261, 1275, 1215, 1260, 1229, -959, 974, 974, 989, 989, -943, 735,
        478, 478, 495, 463, 506, 414, -1039, 1003, 958, 1017, 927, 942, 987, 957, 431, 476, 1272,
        1167, 1228, -1183, 1256, -1199, 895, 895, 941, 941, 1242, 1227, 1212, 1135, 1014, 1014,
        490, 489, 503, 487, 910, 1013, 985, 925, 863, 894, 970, 955, 1012, 847, -1343, 831, 755,
        755, 984, 909, 428, 366, 754, 559, -1391, 752, 486, 457, 924, 997, 698, 698, 983, 893, 740,
        740, 908, 877, 739, 739, 667, 667, 953, 938, 497, 287, 271, 271, 683, 606, 590, 712, 726,
        574, 302, 302, 738, 736, 481, 286, 526, 725, 605, 711, 636, 724, 696, 651, 589, 681, 666,
        710, 364, 467, 573, 695, 466, 466, 301, 465, 379, 379, 709, 604, 665, 679, 316, 316, 634,
        633, 436, 436, 464, 269, 424, 394, 452, 332, 438, 363, 347, 408, 393, 448, 331, 422, 362,
        407, 392, 421, 346, 406, 391, 376, 375, 359, 1441, 1306, -2367, 1290, -2383, 1337, -2399,
        -2415, 1426, 1321, -2431, 1411, 1336, -2447, -2463, -2479, 1169, 1169, 1049, 1049, 1424,
        1289, 1412, 1352, 1319, -2495, 1154, 1154, 1064, 1064, 1153, 1153, 416, 390, 360, 404, 403,
        389, 344, 374, 373, 343, 358, 372, 327, 357, 342, 311, 356, 326, 1395, 1394, 1137, 1137,
        1047, 1047, 1365, 1392, 1287, 1379, 1334, 1364, 1349, 1378, 1318, 1363, 792, 792, 792, 792,
        1152, 1152, 1032, 1032, 1121, 1121, 1046, 1046, 1120, 1120, 1030, 1030, -2895, 1106, 1061,
        1104, 849, 849, 789, 789, 1091, 1076, 1029, 1090, 1060, 1075, 833, 833, 309, 324, 532, 532,
        832, 772, 818, 803, 561, 561, 531, 560, 515, 546, 289, 274, 288, 258, -250, -1179, -1579,
        -1836, -1996, -2124, -2253, -2333, -2413, -2477, -2542, -2574, -2607, -2622, -2655, 1314,
        1313, 1298, 1312, 1282, 785, 785, 785, 785, 1040, 1040, 1025, 1025, 768, 768, 768, 768,
        -766, -798, -830, -862, -895, -911, -927, -943, -959, -975, -991, -1007, -1023, -1039,
        -1055, -1070, 1724, 1647, -1103, -1119, 1631, 1767, 1662, 1738, 1708, 1723, -1135, 1780,
        1615, 1779, 1599, 1677, 1646, 1778, 1583, -1151, 1777, 1567, 1737, 1692, 1765, 1722, 1707,
        1630, 1751, 1661, 1764, 1614, 1736, 1676, 1763, 1750, 1645, 1598, 1721, 1691, 1762, 1706,
        1582, 1761, 1566, -1167, 1749, 1629, 767, 766, 751, 765, 494, 494, 735, 764, 719, 749, 734,
        763, 447, 447, 748, 718, 477, 506, 431, 491, 446, 476, 461, 505, 415, 430, 475, 445, 504,
        399, 460, 489, 414, 503, 383, 474, 429, 459, 502, 502, 746, 752, 488, 398, 501, 473, 413,
        472, 486, 271, 480, 270, -1439, -1455, 1357, -1471, -1487, -1503, 1341, 1325, -1519, 1489,
        1463, 1403, 1309, -1535, 1372, 1448, 1418, 1476, 1356, 1462, 1387, -1551, 1475, 1340, 1447,
        1402, 1386, -1567, 1068, 1068, 1474, 1461, 455, 380, 468, 440, 395, 425, 410, 454, 364,
        467, 466, 464, 453, 269, 409, 448, 268, 432, 1371, 1473, 1432, 1417, 1308, 1460, 1355,
        1446, 1459, 1431, 1083, 1083, 1401, 1416, 1458, 1445, 1067, 1067, 1370, 1457, 1051, 1051,
        1291, 1430, 1385, 1444, 1354, 1415, 1400, 1443, 1082, 1082, 1173, 1113, 1186, 1066, 1185,
        1050, -1967, 1158, 1128, 1172, 1097, 1171, 1081, -1983, 1157, 1112, 416, 266, 375, 400,
        1170, 1142, 1127, 1065, 793, 793, 1169, 1033, 1156, 1096, 1141, 1111, 1155, 1080, 1126,
        1140, 898, 898, 808, 808, 897, 897, 792, 792, 1095, 1152, 1032, 1125, 1110, 1139, 1079,
        1124, 882, 807, 838, 881, 853, 791, -2319, 867, 368, 263, 822, 852, 837, 866, 806, 865,
        -2399, 851, 352, 262, 534, 534, 821, 836, 594, 594, 549, 549, 593, 593, 533, 533, 848, 773,
        579, 579, 564, 578, 548, 563, 276, 276, 577, 576, 306, 291, 516, 560, 305, 305, 275, 259,
        -251, -892, -2058, -2620, -2828, -2957, -3023, -3039, 1041, 1041, 1040, 1040, 769, 769,
        769, 769, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
        -511, -527, -543, -559, 1530, -575, -591, 1528, 1527, 1407, 1526, 1391, 1023, 1023, 1023,
        1023, 1525, 1375, 1268, 1268, 1103, 1103, 1087, 1087, 1039, 1039, 1523, -604, 815, 815,
        815, 815, 510, 495, 509, 479, 508, 463, 507, 447, 431, 505, 415, 399, -734, -782, 1262,
        -815, 1259, 1244, -831, 1258, 1228, -847, -863, 1196, -879, 1253, 987, 987, 748, -767, 493,
        493, 462, 477, 414, 414, 686, 669, 478, 446, 461, 445, 474, 429, 487, 458, 412, 471, 1266,
        1264, 1009, 1009, 799, 799, -1019, -1276, -1452, -1581, -1677, -1757, -1821, -1886, -1933,
        -1997, 1257, 1257, 1483, 1468, 1512, 1422, 1497, 1406, 1467, 1496, 1421, 1510, 1134, 1134,
        1225, 1225, 1466, 1451, 1374, 1405, 1252, 1252, 1358, 1480, 1164, 1164, 1251, 1251, 1238,
        1238, 1389, 1465, -1407, 1054, 1101, -1423, 1207, -1439, 830, 830, 1248, 1038, 1237, 1117,
        1223, 1148, 1236, 1208, 411, 426, 395, 410, 379, 269, 1193, 1222, 1132, 1235, 1221, 1116,
        976, 976, 1192, 1162, 1177, 1220, 1131, 1191, 963, 963, -1647, 961, 780, -1663, 558, 558,
        994, 993, 437, 408, 393, 407, 829, 978, 813, 797, 947, -1743, 721, 721, 377, 392, 844, 950,
        828, 890, 706, 706, 812, 859, 796, 960, 948, 843, 934, 874, 571, 571, -1919, 690, 555, 689,
        421, 346, 539, 539, 944, 779, 918, 873, 932, 842, 903, 888, 570, 570, 931, 917, 674, 674,
        -2575, 1562, -2591, 1609, -2607, 1654, 1322, 1322, 1441, 1441, 1696, 1546, 1683, 1593,
        1669, 1624, 1426, 1426, 1321, 1321, 1639, 1680, 1425, 1425, 1305, 1305, 1545, 1668, 1608,
        1623, 1667, 1592, 1638, 1666, 1320, 1320, 1652, 1607, 1409, 1409, 1304, 1304, 1288, 1288,
        1664, 1637, 1395, 1395, 1335, 1335, 1622, 1636, 1394, 1394, 1319, 1319, 1606, 1621, 1392,
        1392, 1137, 1137, 1137, 1137, 345, 390, 360, 375, 404, 373, 1047, -2751, -2767, -2783,
        1062, 1121, 1046, -2799, 1077, -2815, 1106, 1061, 789, 789, 1105, 1104, 263, 355, 310, 340,
        325, 354, 352, 262, 339, 324, 1091, 1076, 1029, 1090, 1060, 1075, 833, 833, 788, 788, 1088,
        1028, 818, 818, 803, 803, 561, 561, 531, 531, 816, 771, 546, 546, 289, 274, 288, 258, -253,
        -317, -381, -446, -478, -509, 1279, 1279, -811, -1179, -1451, -1756, -1900, -2028, -2189,
        -2253, -2333, -2414, -2445, -2511, -2526, 1313, 1298, -2559, 1041, 1041, 1040, 1040, 1025,
        1025, 1024, 1024, 1022, 1007, 1021, 991, 1020, 975, 1019, 959, 687, 687, 1018, 1017, 671,
        671, 655, 655, 1016, 1015, 639, 639, 758, 758, 623, 623, 757, 607, 756, 591, 755, 575, 754,
        559, 543, 543, 1009, 783, -575, -621, -685, -749, 496, -590, 750, 749, 734, 748, 974, 989,
        1003, 958, 988, 973, 1002, 942, 987, 957, 972, 1001, 926, 986, 941, 971, 956, 1000, 910,
        985, 925, 999, 894, 970, -1071, -1087, -1102, 1390, -1135, 1436, 1509, 1451, 1374, -1151,
        1405, 1358, 1480, 1420, -1167, 1507, 1494, 1389, 1342, 1465, 1435, 1450, 1326, 1505, 1310,
        1493, 1373, 1479, 1404, 1492, 1464, 1419, 428, 443, 472, 397, 736, 526, 464, 464, 486, 457,
        442, 471, 484, 482, 1357, 1449, 1434, 1478, 1388, 1491, 1341, 1490, 1325, 1489, 1463, 1403,
        1309, 1477, 1372, 1448, 1418, 1433, 1476, 1356, 1462, 1387, -1439, 1475, 1340, 1447, 1402,
        1474, 1324, 1461, 1371, 1473, 269, 448, 1432, 1417, 1308, 1460, -1711, 1459, -1727, 1441,
        1099, 1099, 1446, 1386, 1431, 1401, -1743, 1289, 1083, 1083, 1160, 1160, 1458, 1445, 1067,
        1067, 1370, 1457, 1307, 1430, 1129, 1129, 1098, 1098, 268, 432, 267, 416, 266, 400, -1887,
        1144, 1187, 1082, 1173, 1113, 1186, 1066, 1050, 1158, 1128, 1143, 1172, 1097, 1171, 1081,
        420, 391, 1157, 1112, 1170, 1142, 1127, 1065, 1169, 1049, 1156, 1096, 1141, 1111, 1155,
        1080, 1126, 1154, 1064, 1153, 1140, 1095, 1048, -2159, 1125, 1110, 1137, -2175, 823, 823,
        1139, 1138, 807, 807, 384, 264, 368, 263, 868, 838, 853, 791, 867, 822, 852, 837, 866, 806,
        865, 790, -2319, 851, 821, 836, 352, 262, 850, 805, 849, -2399, 533, 533, 835, 820, 336,
        261, 578, 548, 563, 577, 532, 532, 832, 772, 562, 562, 547, 547, 305, 275, 560, 515, 290,
        290, 288, 258,
    ];
    const TABINDEX: [u16; 32] = [
        0, 32, 64, 98, 0, 132, 180, 218, 292, 364, 426, 538, 648, 746, 0, 1126, 1460, 1460, 1460,
        1460, 1460, 1460, 1460, 1460, 1842, 1842, 1842, 1842, 1842, 1842, 1842, 1842,
    ];
    const G_LINBITS: [u8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 6, 8, 10, 13, 4, 5, 6, 7, 8, 9,
        11, 13,
    ];
    let mut big_val_cnt = gr.big_values as isize;
    let sfb = gr.sfbtab;
    let mut sfb_ix = 0;
    let mut scf_ix = 0;
    let mut dst_ix = 0;
    let mut cache = BsCache::new(bs);

    let mut ireg = 0;
    let mut one = 0.0;
    while big_val_cnt > 0 {
        let tab_num = gr.table_select[ireg] as usize;
        let sfb_cnt = gr.region_count[ireg];
        ireg += 1;
        let codebook_ix = TABINDEX[tab_num] as usize;
        let linbits = G_LINBITS[tab_num];
        dump!("linbits", &[linbits as f32, sfb_cnt as f32]);
        if linbits > 0 {
            for _ in 0..=sfb_cnt {
                let np = (sfb[sfb_ix] / 2) as isize;
                sfb_ix += 1;
                let pairs_to_decode = big_val_cnt.min(np);
                one = scf[scf_ix];
                scf_ix += 1;
                for _ in 0..pairs_to_decode {
                    let mut w = 5;
                    let mut leaf = TABS[codebook_ix + cache.peek_bits(w) as usize];
                    while leaf < 0 {
                        cache.flush_bits(w);
                        w = (leaf & 7) as usize;
                        let ix = (cache.peek_bits(w) as usize).wrapping_sub((leaf >> 3) as usize);
                        leaf = TABS[codebook_ix + ix];
                    }
                    cache.flush_bits((leaf >> 8) as usize);
                    for _ in 0..2 {
                        let lsb = leaf & 0x0f;
                        if lsb == 15 {
                            let lsb_sum = lsb as i32 + cache.peek_bits(linbits as usize) as i32;
                            cache.flush_bits(linbits as usize);
                            cache.check_bits(bs);
                            let sign = if cache.peek_bit() { -1. } else { 1. };
                            dst[dst_ix] = one * l3_pow_43(lsb_sum) * sign;
                            dump!("dst1", &dst[dst_ix..][..1]);
                        } else {
                            dst[dst_ix] =
                                G_POW43[16 + lsb as usize - 16 * cache.peek_bit() as usize] * one;
                            dump!("dst2", &dst[dst_ix..][..1]);
                        }
                        cache.flush_bits((lsb > 0) as usize);
                        dst_ix += 1;
                        leaf >>= 4;
                    }
                    cache.check_bits(bs);
                }
                big_val_cnt -= np;
                if big_val_cnt <= 0 {
                    break;
                }
            }
        } else {
            for _ in 0..=sfb_cnt {
                let np = (sfb[sfb_ix] / 2) as isize;
                sfb_ix += 1;
                let pairs_to_decode = big_val_cnt.min(np);
                one = scf[scf_ix];
                scf_ix += 1;
                for _ in 0..pairs_to_decode {
                    let mut w = 5;
                    let mut leaf = TABS[codebook_ix + cache.peek_bits(w) as usize];
                    while leaf < 0 {
                        cache.flush_bits(w);
                        w = (leaf & 7) as usize;
                        let ix = (cache.peek_bits(w) as usize).wrapping_sub((leaf >> 3) as usize);
                        leaf = TABS[codebook_ix + ix];
                    }
                    cache.flush_bits((leaf >> 8) as usize);
                    for _ in 0..2 {
                        let lsb = leaf & 0x0f;
                        dst[dst_ix] =
                            G_POW43[16 + lsb as usize - 16 * cache.peek_bit() as usize] * one;
                        dump!("dst3", &dst[dst_ix..][..1]);
                        cache.flush_bits((lsb > 0) as usize);
                        dst_ix += 1;
                        leaf >>= 4;
                    }
                    cache.check_bits(bs);
                }
                big_val_cnt -= np;
                if big_val_cnt <= 0 {
                    break;
                }
            }
        }
    }

    let mut np = 1 - big_val_cnt;
    const TAB32: [u8; 28] = [
        130, 162, 193, 209, 44, 28, 76, 140, 9, 9, 9, 9, 9, 9, 9, 9, 190, 254, 222, 238, 126, 94,
        157, 157, 109, 61, 173, 205,
    ];
    const TAB33: [u8; 16] = [
        252, 236, 220, 204, 188, 172, 156, 140, 124, 108, 92, 76, 60, 44, 28, 12,
    ];
    let codebook_count1 = if gr.count1_table {
        &TAB33[..]
    } else {
        &TAB32[..]
    };

    'outer: loop {
        let mut leaf = codebook_count1[cache.peek_bits(4) as usize];
        dump!("leaf", &[leaf as f32]);
        if (leaf & 8) == 0 {
            let ix = (leaf >> 3) as u32 + (cache.cache << 4 >> (32 - (leaf & 3)));
            leaf = codebook_count1[ix as usize];
            dump!("leaf2", &[leaf as f32]);
        }
        cache.flush_bits((leaf & 7) as usize);
        if cache.bspos() > layer3gr_limit {
            break;
        }
        for j in (0..4).step_by(2) {
            np -= 1;
            if np == 0 {
                np = (sfb[sfb_ix] / 2) as isize;
                sfb_ix += 1;
                if np == 0 {
                    break 'outer;
                }
                one = scf[scf_ix];
                scf_ix += 1;
            }
            if (leaf & (128 >> j)) != 0 {
                dst[dst_ix + j] = if cache.peek_bit() { -one } else { one };
                cache.flush_bits(1);
            }
            if (leaf & (128 >> (j + 1))) != 0 {
                dst[dst_ix + j + 1] = if cache.peek_bit() { -one } else { one };
                cache.flush_bits(1);
            }
            dump!("dst4", &dst[dst_ix + j..][..2]);
        }
        cache.check_bits(bs);
        dst_ix += 4;
    }
    bs.core.pos = layer3gr_limit;
}

fn l3_midside_stereo(left_right: &mut [[f32; 576]; 2], ix: usize, n: usize) {
    for i in ix..ix + n {
        let a = left_right[0][i];
        let b = left_right[1][i];
        left_right[0][i] = a + b;
        left_right[1][i] = a - b;
    }
}

fn l3_intensity_stereo_band(
    left_right: &mut [[f32; 576]; 2],
    ix: usize,
    n: usize,
    kl: f32,
    kr: f32,
) {
    for i in ix..ix + n {
        let a = left_right[0][i];
        left_right[0][i] = a * kl;
        left_right[1][i] = a * kr;
    }
}

fn l3_stereo_top_band(right: &[f32; 576], sfb: &[u8], nbands: usize) -> [isize; 3] {
    let mut max_band = [-1; 3];
    let mut ix = 0;
    for i in 0..nbands {
        for k in (0..sfb[i]).step_by(2) {
            let base = ix + k as usize;
            if right[base] != 0.0 || right[base + 1] != 0.0 {
                max_band[i % 3] = i as isize;
                break;
            }
        }
        ix += sfb[i] as usize;
    }
    max_band
}

fn l3_stereo_process(
    left_right: &mut [[f32; 576]; 2],
    ist_pos: &[u8],
    sfb: &[u8],
    hdr: Header,
    max_band: [isize; 3],
    mpeg2_sh: u16,
) {
    const G_PAN: [(f32, f32); 7] = [
        (0., 1.),
        (0.21132487, 0.78867513),
        (0.36602540, 0.63397460),
        (0.5, 0.5),
        (0.63397460, 0.36602540),
        (0.78867513, 0.21132487),
        (1., 0.),
    ];
    let max_pos = if hdr.test_mpeg1() { 7 } else { 64 };

    let mut ix = 0;
    for (i, sfb_i) in sfb.iter().enumerate() {
        let n = *sfb_i as usize;
        if n == 0 {
            break;
        }
        let ipos = ist_pos[i] as usize;
        if i as isize > max_band[i % 3] && ipos < max_pos {
            let s = if hdr.test_ms_stereo() {
                1.41421356
            } else {
                1.0
            };
            let (kl, kr) = if hdr.test_mpeg1() {
                G_PAN[ipos]
            } else {
                let k = l3_ldexp_q2(1.0, ((ipos + 1) >> 1 << mpeg2_sh) as i32);
                if ipos % 2 == 0 { (1.0, k) } else { (k, 1.0) }
            };
            l3_intensity_stereo_band(left_right, ix, n, kl * s, kr * s);
        } else if hdr.test_ms_stereo() {
            l3_midside_stereo(left_right, ix, n);
        }
        ix += n;
    }
}

fn l3_intensity_stereo(
    left_right: &mut [[f32; 576]; 2],
    ist_pos: &mut [u8],
    gr: &[GrInfo],
    header: Header,
) {
    let n_sfb = (gr[0].n_long_sfb + gr[0].n_short_sfb) as usize;
    let max_blocks = if gr[0].n_short_sfb > 0 { 3 } else { 1 };

    let mut max_band = l3_stereo_top_band(&left_right[1], &gr[0].sfbtab, n_sfb);
    if gr[0].n_long_sfb > 0 {
        max_band = [max_band[0].max(max_band[1]).max(max_band[2]); 3];
    }
    let default_pos = if header.test_mpeg1() { 3 } else { 0 };
    for i in 0..max_blocks {
        let itop = n_sfb - max_blocks + i;
        let prev = itop - max_blocks;
        ist_pos[itop] = if max_band[i] >= prev as isize {
            default_pos
        } else {
            ist_pos[prev]
        };
    }
    l3_stereo_process(
        left_right,
        ist_pos,
        &gr[0].sfbtab,
        header,
        max_band,
        gr[1].scalefac_compress & 1,
    );
}

fn l3_reorder(grbuf: &mut [f32], scratch: &mut [f32], sfb: &[u8]) {
    let mut src_ix = 0;
    let mut dst_ix = 0;
    for sfb_ix in (0..sfb.len()).step_by(3) {
        let len = sfb[sfb_ix] as usize;
        if len == 0 {
            break;
        }
        for _ in 0..len {
            scratch[dst_ix] = grbuf[src_ix];
            scratch[dst_ix + 1] = grbuf[src_ix + len];
            scratch[dst_ix + 2] = grbuf[src_ix + 2 * len];
            src_ix += 1;
            dst_ix += 3;
        }
        src_ix += 2 * len;
    }
    grbuf[0..dst_ix].copy_from_slice(&scratch[0..dst_ix]);
}

fn l3_antialias(grbuf: &mut [f32; 576], nbands: usize) {
    const G_AA: [[f32; 8]; 2] = [
        [
            0.85749293, 0.88174200, 0.94962865, 0.98331459, 0.99551782, 0.99916056, 0.99989920,
            0.99999316,
        ],
        [
            0.51449576, 0.47173197, 0.31337745, 0.18191320, 0.09457419, 0.04096558, 0.01419856,
            0.00369997,
        ],
    ];
    for band in 0..nbands {
        for i in 0..8 {
            let u = grbuf[band * 18 + 18 + i];
            let d = grbuf[band * 18 + 17 - i];
            grbuf[band * 18 + 18 + i] = u * G_AA[0][i] - d * G_AA[1][i];
            grbuf[band * 18 + 17 - i] = u * G_AA[1][i] + d * G_AA[0][i];
        }
    }
}

fn l3_dct_9(y: &mut [f32; 9]) {
    dump!("l3_dct_9_in", y);
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
    s0 = t0 - t4 + t2;
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
    dump!("l3_dct_9_out", y);
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
    let co = l3_idct3(-x[0], x[6] + x[3], x[12] + x[9]);
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

fn l3_change_sign(grbuf: &mut [f32; 576]) {
    for j in (18..576).step_by(36) {
        for i in (1..18).step_by(2) {
            grbuf[i + j] = -grbuf[i + j];
        }
    }
}

fn l3_imdct_gr(grbuf: &mut [f32; 576], overlap: &mut [f32], block_type: u8, n_long_bands: usize) {
    const G_MDCT_WINDOW: [[f32; 18]; 2] = [
        [
            0.99904822, 0.99144486, 0.97629601, 0.95371695, 0.92387953, 0.88701083, 0.84339145,
            0.79335334, 0.73727734, 0.04361938, 0.13052619, 0.21643961, 0.30070580, 0.38268343,
            0.46174861, 0.53729961, 0.60876143, 0.67559021,
        ],
        [
            1., 1., 1., 1., 1., 1., 0.99144486, 0.92387953, 0.79335334, 0., 0., 0., 0., 0., 0.,
            0.13052619, 0.38268343, 0.60876143,
        ],
    ];
    if n_long_bands > 0 {
        l3_imdct36(grbuf, overlap, &G_MDCT_WINDOW[0], n_long_bands);
        dump!("imdct36", grbuf);
    }
    let gr_slice = &mut grbuf[18 * n_long_bands..];
    let overlap_slice = &mut overlap[9 * n_long_bands..];
    if block_type == SHORT_BLOCK_TYPE {
        l3_imdct_short(gr_slice, overlap_slice, 32 - n_long_bands);
        dump!("imdct_short", grbuf);
    } else {
        let window = &G_MDCT_WINDOW[(block_type == STOP_BLOCK_TYPE) as usize];
        l3_imdct36(gr_slice, overlap_slice, window, 32 - n_long_bands);
        dump!("imdct36_b", grbuf);
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

    fn l3_save_reservoir(&mut self, scratch: &Scratch) {
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

    /// Decode one frame.
    ///
    /// Return value is the number of samples decoded.
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
        let Some(pcm) = pcm else {
            return header.frame_samples();
        };
        let Some(bs_buf) = mp3[i..][..frame_size].get(HDR_SIZE..) else {
            self.header.0[0] = 0;
            return 0;
        };
        let mut bs_core = BsCore::new(bs_buf.len());
        let mut bs_frame = Bs::new(bs_buf, &mut bs_core);
        if header.is_crc() {
            _ = bs_frame.get_bits(16);
        }
        let mut scratch = Scratch::default();
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
                    mp3d_synth_granule(
                        &mut self.qmf_state,
                        &mut scratch.grbuf,
                        18,
                        info.channels,
                        &mut pcm[igr * info.channels * 576..],
                        &mut scratch.syn,
                    );
                }
            }
            self.l3_save_reservoir(&scratch);
            success as usize * self.header.frame_samples()
        } else {
            // optional TODO: implement level 1 & 2
            return 0;
        }
    }

    fn l3_decode(&mut self, scratch: &mut Scratch, igr: usize, nch: usize) {
        for ch in 0..nch {
            let gr = &scratch.gr_info[igr + ch];
            let layer3gr_limit = scratch.bs.pos + gr.part_23_length as usize;
            let mut bs = Bs::new(&scratch.maindata, &mut scratch.bs);
            l3_decode_scalefactors(
                self.header,
                &mut scratch.ist_pos[ch],
                &mut bs,
                gr,
                &mut scratch.scf,
                ch,
            );
            l3_huffman(
                &mut scratch.grbuf[ch],
                &mut bs,
                gr,
                &scratch.scf,
                layer3gr_limit,
            );
            dump!("huffman", &scratch.grbuf[ch]);
        }
        if self.header.test_i_stereo() {
            l3_intensity_stereo(
                &mut scratch.grbuf,
                &mut scratch.ist_pos[1],
                &scratch.gr_info[igr..],
                self.header,
            );
            dump!("i_stereo", &scratch.grbuf[0]);
        } else if self.header.is_ms_stereo() {
            l3_midside_stereo(&mut scratch.grbuf, 0, 576);
            dump!("ms_stereo_l", &scratch.grbuf[0]);
            dump!("ms_stereo_r", &scratch.grbuf[1]);
        }
        for ch in 0..nch {
            let mut aa_bands = 31;
            let gr = &scratch.gr_info[igr + ch];
            let grbuf = &mut scratch.grbuf[ch];
            let n_long_bands = if !gr.mixed_block_flag {
                0usize
            } else if self.header.get_my_sample_rate() == 2 {
                4
            } else {
                2
            };
            if gr.n_short_sfb != 0 {
                aa_bands = n_long_bands.saturating_sub(1);
                let sfb = &gr.sfbtab[gr.n_long_sfb as usize..];
                l3_reorder(&mut grbuf[n_long_bands * 18..], &mut scratch.syn, sfb);
                dump!("reorder", grbuf);
            }
            l3_antialias(grbuf, aa_bands);
            dump!("antialias", grbuf);
            l3_imdct_gr(
                grbuf,
                &mut self.mdct_overlap[ch],
                gr.block_type,
                n_long_bands,
            );
            dump!("imdct", grbuf);
            l3_change_sign(grbuf);
            dump!("change_sign", grbuf);
        }
    }
}

impl Default for Scratch {
    fn default() -> Self {
        Scratch {
            bs: Default::default(),
            maindata: [0; _],
            gr_info: Default::default(),
            grbuf: [[0.0; 576]; 2],
            scf: [0.0; _],
            syn: [0.0; _],
            ist_pos: [[0; _]; _],
        }
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
        if !header.compare(Header::new(&mp3[i..])) {
            return false;
        }
    }
    true
}

/// Returns offset of frame start and frame_bytes
fn mp3d_find_frame(mp3: &[u8], free_format_bytes: &mut usize) -> (usize, usize) {
    for i in 0..mp3.len().saturating_sub(HDR_SIZE) {
        let header = Header::new(&mp3[i..]);
        if header.is_valid() {
            let mut frame_bytes = header.frame_bytes(*free_format_bytes);
            let mut frame_and_padding = frame_bytes + header.padding();
            if frame_bytes == 0 {
                for k in HDR_SIZE..MAX_FREE_FORMAT_FRAME_SIZE {
                    if i + 2 * k >= mp3.len() - HDR_SIZE {
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
    // MPEG-2.5 8 kHz: 6 long bands (72 lines), then short bands 3..12.
    // (minimp3 and libmad use a 36-line split here, which doesn't match
    // n_long_bands or the scalefactor partitions.)
    [
        12, 12, 12, 12, 12, 12, 12, 12, 12, 16, 16, 16, 20, 20, 20, 24, 24, 24, 28, 28, 28, 36,
        36, 36, 2, 2, 2, 2, 2, 2, 2, 2, 2, 26, 26, 26, 0, 0, 0, 0,
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
            gr.mixed_block_flag = bs.get_bits(1) != 0;
            gr.region_count[0] = 7;
            gr.region_count[1] = 255;
            if gr.block_type == SHORT_BLOCK_TYPE {
                scfsi &= 0xf0f;
                if !gr.mixed_block_flag {
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
            gr.mixed_block_flag = false;
            tables = bs.get_bits(15);
            gr.region_count[0] = bs.get_bits(4) as u8;
            gr.region_count[1] = bs.get_bits(3) as u8;
            gr.region_count[2] = 255;
        }
        gr.table_select[0] = (tables >> 10) as u8;
        gr.table_select[1] = (tables >> 5) as u8 & 31;
        gr.table_select[2] = tables as u8 & 31;
        gr.preflag = if header.test_mpeg1() {
            bs.get_bits(1) != 0
        } else {
            gr.scalefac_compress >= 500
        };
        gr.scalefac_scale = bs.get_bits(1) as u8;
        gr.count1_table = bs.get_bits(1) != 0;
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
            if bits == 0 {
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
    while exp_q2 >= 30 * 4 {
        y *= 1.0 / (1 << 30) as f32;
        exp_q2 -= 30 * 4;
    }
    y * G_EXPFRAC[(exp_q2 & 3) as usize] * (1 << 30 >> (exp_q2 >> 2)) as f32
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
        scf_partition_ix += k;
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
    } else if gr.preflag {
        const G_PREAMP: [u8; 10] = [1, 1, 1, 1, 2, 2, 3, 3, 3, 2];
        for i in 0..10 {
            iscf[11 + i] += G_PREAMP[i];
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

fn mp3d_dct_ii(grbuf: &mut [f32; 576], n: usize) {
    const G_SEC: [f32; 24] = [10.19000816,0.50060302,0.50241929,3.40760851,0.50547093,0.52249861,2.05778098,0.51544732,0.56694406,1.48416460,0.53104258,0.64682180,1.16943991,0.55310392,0.78815460,0.97256821,0.58293498,1.06067765,0.83934963,0.62250412,1.72244716,0.74453628,0.67480832,5.10114861];
    let mut t: [[f32; 8]; 4] = [[0.0; 8]; 4];
    for k in 0..n {
        for i in 0..8 {
            let x0 = grbuf[k + i * 18];
            let x1 = grbuf[k + (15 - i) * 18];
            let x2 = grbuf[k + (16 + i) * 18];
            let x3 = grbuf[k + (31 - i) * 18];
            let t0 = x0 + x3;
            let t1 = x1 + x2;
            let t2 = (x1 - x2) * G_SEC[3 * i];
            let t3 = (x0 - x3) * G_SEC[3 * i + 1];
            t[0][i] = t0 + t1;
            t[1][i] = (t0 - t1) * G_SEC[3 * i + 2];
            t[2][i] = t3 + t2;
            t[3][i] = (t3 - t2) * G_SEC[3 * i + 2];
        }
        dump!("t", t.as_flattened());
        for i in 0..4 {
            let mut x0 = t[i][0];
            let mut x1 = t[i][1];
            let mut x2 = t[i][2];
            let mut x3 = t[i][3];
            let mut x4 = t[i][4];
            let mut x5 = t[i][5];
            let mut x6 = t[i][6];
            let mut x7 = t[i][7];
            let mut xt = x0 - x7;
            x0 += x7;
            x7 = x1 - x6;
            x1 += x6;
            x6 = x2 - x5;
            x2 += x5;
            x5 = x3 - x4;
            x3 += x4;
            x4 = x0 - x3;
            x0 += x3;
            x3 = x1 - x2;
            x1 += x2;
            t[i][0] = x0 + x1;
            t[i][4] = (x0 - x1) * 0.70710677;
            x5 += x6;
            x6 = (x6 + x7) * 0.70710677;
            x7 += xt;
            x3 = (x3 + x4) * 0.70710677;
            x5 -= x7 * 0.198912367;
            x7 += x5 * 0.382683432;
            x5 -= x7 * 0.198912367;
            x0 = xt - x6;
            xt += x6;
            t[i][1] = (xt + x7) * 0.50979561;
            t[i][2] = (x4 + x3) * 0.54119611;
            t[i][3] = (x0 - x5) * 0.60134488;
            t[i][5] = (x0 + x5) * 0.89997619;
            t[i][6] = (x4 - x3) * 1.30656302;
            t[i][7] = (xt - x7) * 2.56291556;
        }
        dump!("t2", t.as_flattened());
        for i in 0..7 {
            grbuf[(i * 4) * 18 + k] = t[0][i];
            grbuf[(i * 4 + 1) * 18 + k] = t[2][i] + t[3][i] + t[3][i + 1];
            grbuf[(i * 4 + 2) * 18 + k] = t[1][i] + t[1][i + 1];
            grbuf[(i * 4 + 3) * 18 + k] = t[2][i + 1] + t[3][i] + t[3][i + 1];
        }
        grbuf[28 * 18 + k] = t[0][7];
        grbuf[29 * 18 + k] = t[2][7] + t[3][7];
        grbuf[30 * 18 + k] = t[1][7];
        grbuf[31 * 18 + k] = t[3][7];
    }
}

fn mp3d_scale_pcm(sample: f32) -> Sample {
    // Note: when core_float_math lands, switch to round
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

fn mp3d_synth(grbuf: &[[f32; 576]; 2], band: usize, dstl: &mut [Sample], nch: usize, lins: &mut [f32]) {
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
    lins[Z_OFF + 4 * 15] = grbuf[0][18 * 16 + band];
    lins[Z_OFF + 4 * 15 + 1] = grbuf[right_ix][18 * 16 + band];
    lins[Z_OFF + 4 * 15 + 2] = grbuf[0][band];
    lins[Z_OFF + 4 * 15 + 3] = grbuf[right_ix][band];

    lins[Z_OFF + 4 * 31] = grbuf[0][1 + 18 * 16 + band];
    lins[Z_OFF + 4 * 31 + 1] = grbuf[right_ix][1 + 18 * 16 + band];
    lins[Z_OFF + 4 * 31 + 2] = grbuf[0][1 + band];
    lins[Z_OFF + 4 * 31 + 3] = grbuf[right_ix][1 + band];

    mp3d_synth_pair(&mut dstl[right_ix..], nch, &lins[4 * 15 + 1..]);
    mp3d_synth_pair(
        &mut dstl[right_ix + 32 * nch..],
        nch,
        &lins[4 * 15 + 64 + 1..],
    );
    mp3d_synth_pair(dstl, nch, &lins[4 * 15..]);
    mp3d_synth_pair(&mut dstl[32 * nch..], nch, &lins[4 * 15 + 64..]);

    for i in (0..15).rev() {
        let ix = Z_OFF + 4 * i;
        lins[ix] = grbuf[0][18 * (31 - i) + band];
        lins[ix + 1] = grbuf[right_ix][18 * (31 - i) + band];
        lins[ix + 2] = grbuf[0][1 + 18 * (31 - i) + band];
        lins[ix + 3] = grbuf[right_ix][1 + 18 * (31 - i) + band];
        lins[ix + 64] = grbuf[0][1 + 18 * (1 + i) + band];
        lins[ix + 65] = grbuf[right_ix][1 + 18 * (1 + i) + band];
        lins[ix - 62] = grbuf[0][18 * (1 + i) + band];
        lins[ix - 61] = grbuf[right_ix][18 * (1 + i) + band];
        let mut a = [0.0; 4];
        let mut b = [0.0; 4];
        for k in 0..4 {
            let w0 = G_WIN[(14 - i) * 16 + k * 4];
            let w1 = G_WIN[(14 - i) * 16 + k * 4 + 1];
            let z_ix = ix - k * 2 * 64;
            let y_ix = ix - (15 - k * 2) * 64;
            for j in 0..4 {
                b[j] += lins[z_ix + j] * w1 + lins[y_ix + j] * w0;
                a[j] += lins[z_ix + j] * w0 - lins[y_ix + j] * w1;
            }
            let w2 = G_WIN[(14 - i) * 16 + k * 4 + 2];
            let w3 = G_WIN[(14 - i) * 16 + k * 4 + 3];
            for j in 0..4 {
                b[j] += lins[z_ix - 64 + j] * w3 + lins[y_ix + 64 + j] * w2;
                a[j] += lins[y_ix + 64 + j] * w3 - lins[z_ix - 64 + j] * w2;
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

fn mp3d_synth_granule(
    qmf_state: &mut [f32; 15 * 64],
    grbuf: &mut [[f32; 576]; 2],
    nbands: usize,
    nch: usize,
    pcm: &mut [Sample],
    lins: &mut [f32],
) {
    for gr in &mut grbuf[..nch] {
        dump!("nbands", &[nbands as f32]);
        mp3d_dct_ii(gr, nbands);
        dump!("dct_ii", gr);
    }

    lins[..15 * 64].copy_from_slice(qmf_state);
    for i in (0..nbands).step_by(2) {
        mp3d_synth(
            grbuf,
            i,
            &mut pcm[32 * nch * i..],
            nch,
            &mut lins[i * 64..],
        );
    }
    if nch == 1 {
        for i in (0..15 * 64).step_by(2) {
            qmf_state[i] = lins[nbands * 64 + i];
        }
    } else {
        qmf_state.copy_from_slice(&lins[nbands * 64..][..15 * 64]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sfb_tables_cover_granule() {
        for sr_idx in 0..8 {
            let sum = |t: &[u8]| t.iter().map(|&x| x as usize).sum::<usize>();
            assert_eq!(sum(&G_SCF_LONG[sr_idx]), 576);
            assert_eq!(sum(&G_SCF_SHORT[sr_idx]), 576);
            let mixed = &G_SCF_MIXED[sr_idx];
            // Matches n_long_sfb and n_long_bands in l3_read_side_info / l3_decode.
            let (n_long_sfb, n_long_bands) = match sr_idx {
                5.. => (8, 2),
                1 => (6, 4),
                _ => (6, 2),
            };
            assert_eq!(sum(&mixed[..n_long_sfb]), 18 * n_long_bands);
            assert_eq!(sum(mixed), 576);
            assert_eq!(mixed[n_long_sfb + 30], 0);
        }
    }
}
