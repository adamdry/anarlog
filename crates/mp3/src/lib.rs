mod encoder;
mod error;
mod segment;
mod upload;
mod wav;

pub use encoder::{Mono, MonoStreamEncoder, Stereo, StereoStreamEncoder, StreamEncoder};
pub use error::Error;
pub use segment::encode_mono_segments;
pub use upload::encode_for_upload;
pub use wav::{concat_files, decode_to_wav, encode_wav};
