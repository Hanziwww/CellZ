use std::io::{self, Read, Write};

/// 写入 LEB128 风格的无符号 varint
pub fn write_varint<W: Write>(writer: &mut W, mut value: u64) -> io::Result<usize> {
    let mut bytes_written = 0;
    loop {
        let to_write = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            writer.write_all(&[to_write | 0x80])?;
        } else {
            writer.write_all(&[to_write])?;
            bytes_written += 1;
            break;
        }
        bytes_written += 1;
    }
    Ok(bytes_written)
}

/// 读取 LEB128 风格的无符号 varint
pub fn read_varint<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut shift = 0;
    let mut result = 0u64;
    loop {
        let mut buf = [0u8; 1];
        reader.read_exact(&mut buf)?;
        let byte_val = buf[0];
        result |= ((byte_val & 0x7F) as u64) << shift;
        if (byte_val & 0x80) == 0 {
            break;
        }
        shift += 7;
        if shift > 63 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "varint too long"));
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_varint_roundtrip() {
        let test_values = vec![0, 127, 128, 255, 256, 16384, 1_000_000, u64::MAX >> 8];
        
        for &val in &test_values {
            let mut buf = Vec::new();
            write_varint(&mut buf, val).unwrap();
            let mut cursor = Cursor::new(buf);
            let decoded = read_varint(&mut cursor).unwrap();
            assert_eq!(val, decoded);
        }
    }
}

