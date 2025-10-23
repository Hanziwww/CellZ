/// 位写入器，用于 Golomb-Rice 和 unary 编码
pub struct BitWriter {
    buffer: Vec<u8>,
    current: u8,
    n_bits: u8,
}

impl BitWriter {
    pub fn new() -> Self {
        BitWriter {
            buffer: Vec::new(),
            current: 0,
            n_bits: 0,
        }
    }

    pub fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current |= 1 << (7 - self.n_bits);
        }
        self.n_bits += 1;
        if self.n_bits == 8 {
            self.buffer.push(self.current);
            self.current = 0;
            self.n_bits = 0;
        }
    }

    pub fn write_bits(&mut self, value: u64, nbits: u32) {
        if nbits == 0 {
            return;
        }
        for i in (0..nbits).rev() {
            self.write_bit((value >> i) & 1 == 1);
        }
    }

    pub fn write_unary_ones_then_zero(&mut self, q: u64) {
        for _ in 0..q {
            self.write_bit(true);
        }
        self.write_bit(false);
    }

    pub fn finish(mut self) -> Vec<u8> {
        if self.n_bits > 0 {
            self.buffer.push(self.current);
        }
        self.buffer
    }
}

/// 位读取器，用于解码 unary 和定长位值
pub struct BitReader {
    data: Vec<u8>,
    byte_index: usize,
    bit_index: u8,
}

impl BitReader {
    pub fn new(data: Vec<u8>) -> Self {
        BitReader {
            data,
            byte_index: 0,
            bit_index: 0,
        }
    }

    fn ensure_byte(&self) -> u8 {
        if self.byte_index >= self.data.len() {
            0
        } else {
            self.data[self.byte_index]
        }
    }

    pub fn read_bit(&mut self) -> bool {
        let b = self.ensure_byte();
        let bit = (b >> (7 - self.bit_index)) & 1 == 1;
        self.bit_index += 1;
        if self.bit_index == 8 {
            self.bit_index = 0;
            self.byte_index += 1;
        }
        bit
    }

    pub fn read_bits(&mut self, nbits: u32) -> u64 {
        if nbits == 0 {
            return 0;
        }
        let mut v = 0u64;
        for _ in 0..nbits {
            v = (v << 1) | if self.read_bit() { 1 } else { 0 };
        }
        v
    }

    pub fn read_unary_ones_then_zero_count(&mut self) -> u64 {
        let mut q = 0u64;
        loop {
            if self.read_bit() {
                q += 1;
            } else {
                break;
            }
        }
        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitwriter_reader_roundtrip() {
        let mut bw = BitWriter::new();
        bw.write_bits(0b1010, 4);
        bw.write_unary_ones_then_zero(3);
        bw.write_bits(0b11, 2);
        
        let bytes = bw.finish();
        let mut br = BitReader::new(bytes);
        
        assert_eq!(br.read_bits(4), 0b1010);
        assert_eq!(br.read_unary_ones_then_zero_count(), 3);
        assert_eq!(br.read_bits(2), 0b11);
    }
}

