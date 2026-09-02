use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anlg_audio_utils::Source;

use crate::encoder::f32_to_i16;
use crate::{Error, MonoStreamEncoder, StereoStreamEncoder};

const ENCODE_FRAMES: usize = 4096;

/// Re-encodes any supported audio file as a low-bitrate MP3 for a
/// speech-to-text upload. Channel layout is preserved so multichannel
/// providers keep their per-channel speaker attribution.
pub fn encode_for_upload(
    source_path: &Path,
    output_path: &Path,
    kbps_per_channel: u32,
    mut on_progress: Option<&mut dyn FnMut(f64)>,
) -> Result<(), Error> {
    let source = anlg_audio_utils::source_from_path(source_path)?;
    let sample_rate: u32 = source.sample_rate().into();
    let channels: u16 = source.channels().into();
    let total_frames = source
        .total_duration()
        .map(|duration| (duration.as_secs_f64() * f64::from(sample_rate)).ceil() as usize)
        .filter(|frames| *frames > 0);

    let mut writer = BufWriter::new(File::create(output_path)?);
    let mut encoded = Vec::new();
    let mut frames = 0usize;
    let mut report = |frames: usize| {
        if let (Some(on_progress), Some(total)) = (on_progress.as_deref_mut(), total_frames) {
            on_progress((frames as f64 / total as f64).min(1.0));
        }
    };

    match channels {
        1 => {
            let mut encoder = MonoStreamEncoder::for_upload(sample_rate, kbps_per_channel)?;
            let mut pcm = Vec::with_capacity(ENCODE_FRAMES);
            for sample in source {
                pcm.push(f32_to_i16(sample));
                if pcm.len() < ENCODE_FRAMES {
                    continue;
                }
                frames += pcm.len();
                encoded.clear();
                encoder.encode_i16(&pcm, &mut encoded)?;
                writer.write_all(&encoded)?;
                pcm.clear();
                report(frames);
            }
            encoded.clear();
            encoder.encode_i16(&pcm, &mut encoded)?;
            encoder.flush(&mut encoded)?;
            writer.write_all(&encoded)?;
        }
        2 => {
            let mut encoder = StereoStreamEncoder::for_upload(sample_rate, kbps_per_channel)?;
            let mut left = Vec::with_capacity(ENCODE_FRAMES);
            let mut right = Vec::with_capacity(ENCODE_FRAMES);
            let mut source = source.into_iter();
            while let Some(sample) = source.next() {
                left.push(f32_to_i16(sample));
                right.push(f32_to_i16(source.next().unwrap_or(0.0)));
                if left.len() < ENCODE_FRAMES {
                    continue;
                }
                frames += left.len();
                encoded.clear();
                encoder.encode_i16(&left, &right, &mut encoded)?;
                writer.write_all(&encoded)?;
                left.clear();
                right.clear();
                report(frames);
            }
            encoded.clear();
            encoder.encode_i16(&left, &right, &mut encoded)?;
            encoder.flush(&mut encoded)?;
            writer.write_all(&encoded)?;
        }
        count => return Err(Error::UnsupportedChannelCount(count)),
    }

    writer.flush()?;
    writer.get_ref().sync_all()?;
    report(usize::MAX);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tone(path: &Path, channels: u16, seconds: u32) {
        let sample_rate = 16_000;
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for frame in 0..(sample_rate * seconds) {
            let sample =
                ((frame as f32 / sample_rate as f32) * 440.0 * std::f32::consts::TAU).sin();
            for channel in 0..channels {
                let gain = if channel == 0 { 0.5 } else { 0.25 };
                writer
                    .write_sample((sample * gain * i16::MAX as f32) as i16)
                    .unwrap();
            }
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn shrinks_stereo_audio_and_keeps_both_channels() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.wav");
        let output = dir.path().join("upload.mp3");
        write_tone(&source, 2, 4);

        let mut progress = Vec::new();
        encode_for_upload(&source, &output, 24, Some(&mut |p| progress.push(p))).unwrap();

        let source_size = std::fs::metadata(&source).unwrap().len();
        let output_size = std::fs::metadata(&output).unwrap().len();
        assert!(
            output_size * 4 < source_size,
            "{output_size} bytes is not much smaller than {source_size}"
        );

        let decoded = anlg_audio_utils::source_from_path(&output).unwrap();
        assert_eq!(u16::from(decoded.channels()), 2);
        assert_eq!(u32::from(decoded.sample_rate()), 16_000);

        assert!(progress.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(progress.last().copied(), Some(1.0));
    }

    #[test]
    fn encodes_mono_audio() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.wav");
        let output = dir.path().join("upload.mp3");
        write_tone(&source, 1, 2);

        encode_for_upload(&source, &output, 24, None).unwrap();

        let decoded = anlg_audio_utils::source_from_path(&output).unwrap();
        assert_eq!(u16::from(decoded.channels()), 1);
    }
}
