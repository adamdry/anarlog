use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use bytes::Bytes;
use http_body::{Body, Frame, SizeHint};
use reqwest_middleware::{Middleware, Next};

/// Receives `(bytes handed to the transport so far, total body bytes)`.
pub type UploadProgressFn = Arc<dyn Fn(u64, u64) + Send + Sync>;

/// Bodies below this size are polling or metadata requests, not the audio.
const MIN_TRACKED_BODY_BYTES: u64 = 64 * 1024;

/// Wraps outgoing request bodies so callers can observe how much of an audio
/// upload has left the client. Adapters keep building requests however they
/// like; anything with a known body length gets reported.
pub(crate) struct UploadProgressMiddleware {
    on_progress: UploadProgressFn,
}

impl UploadProgressMiddleware {
    pub(crate) fn new(on_progress: UploadProgressFn) -> Self {
        Self { on_progress }
    }
}

#[async_trait::async_trait]
impl Middleware for UploadProgressMiddleware {
    async fn handle(
        &self,
        mut req: reqwest::Request,
        extensions: &mut http::Extensions,
        next: Next<'_>,
    ) -> reqwest_middleware::Result<reqwest::Response> {
        if let Some(total) = request_body_len(&req).filter(|total| *total >= MIN_TRACKED_BODY_BYTES)
            && let Some(inner) = req.body_mut().take()
        {
            *req.body_mut() = Some(reqwest::Body::wrap(ProgressBody {
                inner,
                sent: 0,
                total,
                on_progress: self.on_progress.clone(),
            }));
        }

        next.run(req, extensions).await
    }
}

fn request_body_len(req: &reqwest::Request) -> Option<u64> {
    let body = req.body()?;
    body.size_hint().exact().or_else(|| {
        req.headers()
            .get(http::header::CONTENT_LENGTH)?
            .to_str()
            .ok()?
            .parse()
            .ok()
    })
}

struct ProgressBody {
    inner: reqwest::Body,
    sent: u64,
    total: u64,
    on_progress: UploadProgressFn,
}

impl Body for ProgressBody {
    type Data = Bytes;
    type Error = reqwest::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.inner).poll_frame(cx);
        if let Poll::Ready(Some(Ok(frame))) = &polled
            && let Some(data) = frame.data_ref()
        {
            this.sent = this.sent.saturating_add(data.len() as u64);
            (this.on_progress)(this.sent, this.total);
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

    use super::*;

    async fn post_with_progress(server: &MockServer, body: Vec<u8>) -> Vec<(u64, u64)> {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let client = reqwest_middleware::ClientBuilder::new(reqwest::Client::new())
            .with(UploadProgressMiddleware::new({
                let seen = seen.clone();
                Arc::new(move |sent, total| seen.lock().unwrap().push((sent, total)))
            }))
            .build();

        let length = body.len();
        client
            .post(format!("{}/upload", server.uri()))
            .header(http::header::CONTENT_LENGTH, length)
            .body(reqwest::Body::wrap_stream(tokio_stream::iter(
                body.chunks(50 * 1024)
                    .map(|chunk| Ok::<_, std::io::Error>(Bytes::copy_from_slice(chunk)))
                    .collect::<Vec<_>>(),
            )))
            .send()
            .await
            .unwrap();

        seen.lock().unwrap().clone()
    }

    #[tokio::test]
    async fn reports_bytes_sent_against_content_length() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let total = 200 * 1024;
        let progress = post_with_progress(&server, vec![b'a'; total]).await;

        assert!(!progress.is_empty());
        assert!(
            progress
                .iter()
                .all(|(_, seen_total)| *seen_total == total as u64)
        );
        assert!(progress.windows(2).all(|pair| pair[0].0 <= pair[1].0));
        assert_eq!(progress.last().unwrap().0, total as u64);
    }

    #[tokio::test]
    async fn ignores_small_request_bodies() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let progress = post_with_progress(&server, vec![b'a'; 1024]).await;

        assert!(progress.is_empty());
    }
}
