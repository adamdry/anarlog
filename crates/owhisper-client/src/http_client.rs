use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use reqwest_tracing::TracingMiddleware;

use crate::upload_progress::{UploadProgressFn, UploadProgressMiddleware};

pub fn create_client() -> ClientWithMiddleware {
    ClientBuilder::new(reqwest::Client::new())
        .with(TracingMiddleware::default())
        .build()
}

pub fn create_client_with_upload_progress(on_progress: UploadProgressFn) -> ClientWithMiddleware {
    ClientBuilder::new(reqwest::Client::new())
        .with(TracingMiddleware::default())
        .with(UploadProgressMiddleware::new(on_progress))
        .build()
}
