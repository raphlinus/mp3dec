pub struct Bs<'a> {
    pub buf: &'a [u8],
    pub core: &'a mut BsCore,
}

#[derive(Default)]
pub struct BsCore {
    pub pos: usize,
    pub limit: usize,
}

pub struct BsCache {
    ix: usize,
    pub cache: u32,
    shift: isize,
}

impl<'a> Bs<'a> {
    pub fn new(buf: &'a [u8], core: &'a mut BsCore) -> Self {
        Self { buf, core }
    }

    pub fn get_bits(&mut self, n: usize) -> u32 {
        self.core.get_bits(self.buf, n)
    }
}

impl BsCore {
    pub fn new(len: usize) -> Self {
        Self {
            pos: 0,
            limit: len * 8,
        }
    }

    pub fn get_bits(&mut self, buf: &[u8], n: usize) -> u32 {
        let s = self.pos % 8;
        let mut ix = self.pos / 8;
        self.pos += n;
        if self.pos > self.limit {
            return 0;
        }
        let mut cache = 0;
        let mut next = buf[ix] as u32 & (0xff >> s);
        let mut shl = n + s;
        while shl > 8 {
            shl -= 8;
            cache |= next << shl;
            ix += 1;
            next = buf[ix] as u32;
        }
        cache | (next >> (8 - shl))
    }
}

impl BsCache {
    pub fn new(bs: &Bs) -> Self {
        let mut ix = bs.core.pos / 8;
        if let Ok(bytes) = bs.buf[ix..][..4].try_into() {
            let cache = u32::from_be_bytes(bytes) << (bs.core.pos & 7);
            let shift = (bs.core.pos & 7) as isize - 8;
            ix += 4;
            Self { ix, cache, shift }
        } else {
            // TODO: do we need to handle end-of-buffer, or is there always slack?
            unimplemented!()
        }
    }

    pub fn peek_bits(&self, n: usize) -> u32 {
        self.cache >> (32 - n)
    }

    pub fn peek_bit(&self) -> bool {
        (self.cache as i32) < 0
    }

    pub fn flush_bits(&mut self, n: usize) {
        self.cache <<= n;
        self.shift += n as isize;
    }

    pub fn check_bits(&mut self, bs: &mut Bs) {
        while self.shift >= 0 {
            self.cache |= (bs.buf.get(self.ix).copied().unwrap_or_default() as u32) << self.shift;
            self.ix += 1;
            self.shift -= 8;
        }
    }

    pub fn bspos(&self) -> usize {
        (self.ix * 8 - 24).wrapping_add(self.shift as usize)
    }
}
