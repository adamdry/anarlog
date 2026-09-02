use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use owhisper_client::{
    AdapterKind, AnarlogAdapter, AquaVoiceAdapter, ArgmaxAdapter, AssemblyAIAdapter,
    AwsTranscribeAdapter, AzureSpeechAdapter, BatchSttAdapter, BatchUploadLimit, CartesiaAdapter,
    CohereAdapter, DeepgramAdapter, ElevenLabsAdapter, FireworksAdapter, GladiaAdapter,
    GoogleCloudAdapter, GoogleGenerativeAiAdapter, GroqAdapter, MistralAdapter, OpenAIAdapter,
    OpenRouterAdapter, PyannoteAdapter, RevAiAdapter, SiliconFlowAdapter, SonioxAdapter,
    SpeechmaticsAdapter, TogetherAdapter, XaiAdapter, ZaiAdapter,
};
use owhisper_interface::batch::{Alternatives, Channel, Response, Results};
use owhisper_interface::batch_stream::{BatchProgressStage, BatchStreamEvent};
use tracing::Instrument;

use super::super::upload::{audio_duration, segment_plan, split_batch_upload};
use super::super::{
    BatchParams, BatchRunMode, BatchRunOutput, format_user_friendly_error, session_span,
};
use crate::{BatchEvent, BatchRuntime};

pub(super) const DIRECT_BATCH_TIMEOUT_FLOOR: Duration = Duration::from_secs(15 * 60);
pub(super) const DIRECT_BATCH_TIMEOUT_CEILING: Duration = Duration::from_secs(6 * 60 * 60);
const DIRECT_BATCH_TIMEOUT_BUFFER: Duration = Duration::from_secs(5 * 60);
const DIRECT_BATCH_AUDIO_DURATION_MULTIPLIER: u32 = 2;
const ANARLOG_PROXY_MAX_AUDIO_BYTES: u64 = 512 * 1024 * 1024;
/// Speech survives this comfortably; recordings are stored at 64 kbps per
/// channel for playback, so uploads shrink by more than half.
const UPLOAD_KBPS_PER_CHANNEL: u32 = 24;
/// Below this a re-encode takes longer than the upload it would save.
const UPLOAD_COMPRESSION_MIN_BYTES: u64 = 1024 * 1024;
const UPLOAD_COMPRESSION_MIN_RATIO: f64 = 1.25;
const PROGRESS_EMIT_INTERVAL: Duration = Duration::from_millis(250);

/// Forwards preparation and upload progress to the UI, throttled so a fast
/// link does not flood the event channel.
pub(super) struct BatchProgress {
    runtime: Arc<dyn BatchRuntime>,
    session_id: String,
    last_emit: Mutex<Option<(BatchProgressStage, Instant)>>,
}

impl BatchProgress {
    pub(super) fn new(runtime: Arc<dyn BatchRuntime>, session_id: String) -> Arc<Self> {
        Arc::new(Self {
            runtime,
            session_id,
            last_emit: Mutex::new(None),
        })
    }

    fn emit(&self, stage: BatchProgressStage, percentage: f64) {
        let percentage = percentage.clamp(0.0, 1.0);
        let now = Instant::now();
        let settled = percentage <= 0.0 || percentage >= 1.0;
        let mut last_emit = self.last_emit.lock().unwrap();
        if !settled
            && let Some((last_stage, at)) = *last_emit
            && last_stage == stage
            && now.duration_since(at) < PROGRESS_EMIT_INTERVAL
        {
            return;
        }
        *last_emit = Some((stage, now));
        drop(last_emit);

        self.runtime.emit(BatchEvent::BatchResponseStreamed {
            session_id: self.session_id.clone(),
            event: BatchStreamEvent::Progress {
                percentage,
                partial_text: None,
                stage: Some(stage),
            },
        });
    }
}

/// Maps one request body's upload onto the whole recording, which may be
/// split across several requests.
#[derive(Clone)]
pub(super) struct UploadProgress {
    progress: Arc<BatchProgress>,
    base_bytes: u64,
    request_bytes: u64,
    total_bytes: u64,
}

impl UploadProgress {
    pub(super) fn whole(progress: Arc<BatchProgress>, file_path: &str) -> Self {
        let bytes = std::fs::metadata(file_path).map(|m| m.len()).unwrap_or(0);
        Self {
            progress,
            base_bytes: 0,
            request_bytes: bytes,
            total_bytes: bytes,
        }
    }

    fn report(&self, sent: u64, body_len: u64) {
        if self.total_bytes == 0 || body_len == 0 {
            return;
        }
        let request_fraction = (sent as f64 / body_len as f64).min(1.0);
        let uploaded = self.base_bytes as f64 + request_fraction * self.request_bytes as f64;
        self.progress.emit(
            BatchProgressStage::Uploading,
            uploaded / self.total_bytes as f64,
        );
        if sent >= body_len && self.base_bytes + self.request_bytes >= self.total_bytes {
            self.progress.emit(BatchProgressStage::Transcribing, 0.0);
        }
    }
}

pub(super) enum PreparedBatchUpload {
    Original(PathBuf),
    Compressed {
        _temp_dir: tempfile::TempDir,
        path: PathBuf,
    },
}

impl PreparedBatchUpload {
    pub(super) fn path(&self) -> &Path {
        match self {
            Self::Original(path) | Self::Compressed { path, .. } => path,
        }
    }

    fn is_compressed(&self) -> bool {
        matches!(self, Self::Compressed { .. })
    }
}

macro_rules! dispatch_batch {
    ($ak:expr, $params:expr, $lp:expr, $limit:expr, $progress:expr,
     { $($var:ident => $adapter:ty),+ $(,)? },
     unsupported: [$($unsup:ident),* $(,)?]
    ) => {
        match $ak {
            $(AdapterKind::$var => {
                run_direct_batch::<$adapter>(
                    &AdapterKind::$var.to_string(),
                    $params,
                    $lp,
                    $limit,
                    $progress,
                )
                .await
            })+
            $(AdapterKind::$unsup => {
                Err(crate::BatchFailure::DirectBatchUnsupported {
                    provider: AdapterKind::$unsup.to_string(),
                }.into())
            })*
        }
    };
}

pub(in crate::batch) async fn run_direct_batch_for_adapter_kind(
    runtime: Arc<dyn BatchRuntime>,
    adapter_kind: AdapterKind,
    mut params: BatchParams,
    listen_params: owhisper_interface::ListenParams,
) -> crate::Result<BatchRunOutput> {
    let progress = BatchProgress::new(runtime, params.session_id.clone());

    if adapter_kind == AdapterKind::Anarlog {
        return run_anarlog_batch(progress, params, listen_params).await;
    }

    let upload = prepare_direct_batch_upload(
        &params.file_path,
        &adapter_kind.to_string(),
        false,
        Some(progress.clone()),
    )
    .await?;
    params.file_path = upload.path().to_string_lossy().into_owned();

    let limit = adapter_kind.batch_upload_limit(listen_params.model.as_deref());

    dispatch_batch!(adapter_kind, params, listen_params, limit, progress, {
        Argmax => ArgmaxAdapter,
        Cartesia => CartesiaAdapter,
        Deepgram => DeepgramAdapter,
        Soniox => SonioxAdapter,
        AssemblyAI => AssemblyAIAdapter,
        Fireworks => FireworksAdapter,
        OpenAI => OpenAIAdapter,
        OpenRouter => OpenRouterAdapter,
        SiliconFlow => SiliconFlowAdapter,
        Zai => ZaiAdapter,
        Gladia => GladiaAdapter,
        ElevenLabs => ElevenLabsAdapter,
        Pyannote => PyannoteAdapter,
        Mistral => MistralAdapter,
        Anarlog => AnarlogAdapter,
        AquaVoice => AquaVoiceAdapter,
        Cohere => CohereAdapter,
        AwsTranscribe => AwsTranscribeAdapter,
        AzureSpeech => AzureSpeechAdapter,
        GoogleCloud => GoogleCloudAdapter,
        GoogleGenerativeAi => GoogleGenerativeAiAdapter,
        Groq => GroqAdapter,
        RevAi => RevAiAdapter,
        Speechmatics => SpeechmaticsAdapter,
        Together => TogetherAdapter,
        Xai => XaiAdapter,
    }, unsupported: [DashScope])
}

async fn run_anarlog_batch(
    progress: Arc<BatchProgress>,
    mut params: BatchParams,
    listen_params: owhisper_interface::ListenParams,
) -> crate::Result<BatchRunOutput> {
    let upload = prepare_anarlog_batch_upload(
        &params.file_path,
        ANARLOG_PROXY_MAX_AUDIO_BYTES,
        Some(progress.clone()),
    )
    .await?;
    params.file_path = upload.path().to_string_lossy().into_owned();
    run_direct_batch::<AnarlogAdapter>(
        &AdapterKind::Anarlog.to_string(),
        params,
        listen_params,
        None,
        progress,
    )
    .await
}

pub(super) async fn prepare_anarlog_batch_upload(
    file_path: &str,
    max_bytes: u64,
    progress: Option<Arc<BatchProgress>>,
) -> crate::Result<PreparedBatchUpload> {
    let provider = AdapterKind::Anarlog.to_string();
    let mut upload =
        prepare_direct_batch_upload(file_path, &provider, false, progress.clone()).await?;
    if tokio::fs::metadata(upload.path()).await?.len() <= max_bytes {
        return Ok(upload);
    }

    if !upload.is_compressed() {
        upload = prepare_direct_batch_upload(file_path, &provider, true, progress).await?;
    }

    if tokio::fs::metadata(upload.path()).await?.len() > max_bytes {
        return Err(crate::BatchFailure::DirectRequestFailed {
            provider,
            message:
                "This recording is too large for cloud transcription. Split it into smaller files and try again."
                    .to_string(),
        }
        .into());
    }

    Ok(upload)
}

/// Re-encodes a recording to a lean speech bitrate before it goes over the
/// wire. A recording that is already small, or already lean, is sent as-is;
/// `force` re-encodes regardless (for providers with a hard size cap). A failed
/// re-encode falls back to the original file unless forced.
pub(super) async fn prepare_direct_batch_upload(
    file_path: &str,
    provider: &str,
    force: bool,
    progress: Option<Arc<BatchProgress>>,
) -> crate::Result<PreparedBatchUpload> {
    let source_path = PathBuf::from(file_path);
    let source_size = tokio::fs::metadata(&source_path).await?.len();
    let channels = tokio::task::spawn_blocking({
        let path = source_path.clone();
        move || anlg_audio_utils::audio_file_metadata(path).map(|m| m.channels)
    })
    .await
    .ok()
    .and_then(Result::ok);
    let duration = audio_duration(file_path);

    if !force && !should_compress_for_upload(source_size, duration, channels) {
        return Ok(PreparedBatchUpload::Original(source_path));
    }

    let failure = |message: &str| crate::BatchFailure::DirectRequestFailed {
        provider: provider.to_string(),
        message: message.to_string(),
    };
    let fallback = |error: &dyn std::fmt::Display, what: &'static str| {
        if force {
            tracing::error!(%error, "batch_upload_compression_failed");
            Err(failure("Anarlog couldn't prepare this large recording for transcription.").into())
        } else {
            tracing::warn!(%error, what, "batch_upload_compression_skipped");
            Ok(PreparedBatchUpload::Original(source_path.clone()))
        }
    };

    let temp_dir = match tempfile::tempdir() {
        Ok(dir) => dir,
        Err(error) => return fallback(&error, "temp_dir"),
    };
    let encoded_path = temp_dir.path().join("upload.mp3");

    if let Some(progress) = &progress {
        progress.emit(BatchProgressStage::Preparing, 0.0);
    }
    let encode_result = tokio::task::spawn_blocking({
        let source = source_path.clone();
        let target = encoded_path.clone();
        let progress = progress.clone();
        move || {
            let mut on_progress = progress
                .map(|progress| move |p: f64| progress.emit(BatchProgressStage::Preparing, p));
            anlg_mp3::encode_for_upload(
                &source,
                &target,
                UPLOAD_KBPS_PER_CHANNEL,
                on_progress.as_mut().map(|f| f as &mut dyn FnMut(f64)),
            )
        }
    })
    .await;

    let encode_result = match encode_result {
        Ok(result) => result,
        Err(error) => return fallback(&error, "encode_task"),
    };
    if let Err(error) = encode_result {
        return fallback(&error, "encode");
    }

    let encoded_size = tokio::fs::metadata(&encoded_path).await?.len();
    if !force && encoded_size >= source_size {
        tracing::info!(
            source_size,
            encoded_size,
            "batch_upload_compression_not_smaller"
        );
        return Ok(PreparedBatchUpload::Original(source_path));
    }

    tracing::info!(
        source_size,
        encoded_size,
        "batch_audio_compressed_for_upload"
    );

    Ok(PreparedBatchUpload::Compressed {
        _temp_dir: temp_dir,
        path: encoded_path,
    })
}

pub(super) fn should_compress_for_upload(
    size_bytes: u64,
    duration: Option<Duration>,
    channels: Option<u8>,
) -> bool {
    if size_bytes < UPLOAD_COMPRESSION_MIN_BYTES {
        return false;
    }
    let Some(seconds) = duration.map(|d| d.as_secs_f64()).filter(|s| *s > 0.0) else {
        return false;
    };
    let Some(channels) = channels.filter(|c| matches!(c, 1 | 2)) else {
        return false;
    };

    let source_kbps = size_bytes as f64 * 8.0 / seconds / 1000.0;
    let target_kbps = f64::from(UPLOAD_KBPS_PER_CHANNEL * u32::from(channels));
    source_kbps > target_kbps * UPLOAD_COMPRESSION_MIN_RATIO
}

pub(super) async fn run_direct_batch<A: BatchSttAdapter>(
    provider: &str,
    params: BatchParams,
    listen_params: owhisper_interface::ListenParams,
    limit: Option<BatchUploadLimit>,
    progress: Arc<BatchProgress>,
) -> crate::Result<BatchRunOutput> {
    let audio_duration = audio_duration(&params.file_path);
    let timeout = direct_batch_timeout_for_audio(audio_duration);

    match segment_plan(&params.file_path, audio_duration, limit) {
        Some(segment_duration) => {
            run_segmented_batch::<A>(
                provider,
                params,
                listen_params,
                segment_duration,
                timeout,
                progress,
            )
            .await
        }
        None => {
            let upload = UploadProgress::whole(progress, &params.file_path);
            run_direct_batch_with_timeout::<A>(provider, params, listen_params, timeout, upload)
                .await
        }
    }
}

async fn run_segmented_batch<A: BatchSttAdapter>(
    provider: &str,
    params: BatchParams,
    mut listen_params: owhisper_interface::ListenParams,
    segment_duration: Duration,
    timeout: Duration,
    progress: Arc<BatchProgress>,
) -> crate::Result<BatchRunOutput> {
    let segments = split_batch_upload(&params.file_path, segment_duration, provider).await?;
    listen_params.channels = 1;

    let segment_sizes: Vec<u64> = segments
        .paths()
        .iter()
        .map(|path| std::fs::metadata(path).map(|m| m.len()).unwrap_or(0))
        .collect();
    let total_bytes: u64 = segment_sizes.iter().sum();
    let mut base_bytes = 0;

    let mut responses = Vec::with_capacity(segments.paths().len());
    for (path, request_bytes) in segments.paths().iter().zip(segment_sizes) {
        let mut segment_params = params.clone();
        segment_params.file_path = path.to_string_lossy().into_owned();

        let output = run_direct_batch_with_timeout::<A>(
            provider,
            segment_params,
            listen_params.clone(),
            timeout,
            UploadProgress {
                progress: progress.clone(),
                base_bytes,
                request_bytes,
                total_bytes,
            },
        )
        .await?;
        base_bytes += request_bytes;
        responses.push(output.response);
    }

    Ok(BatchRunOutput {
        session_id: params.session_id,
        mode: BatchRunMode::Direct,
        response: merge_segment_responses(responses, segment_duration),
    })
}

/// Segments are transcribed independently, so their timestamps restart at zero.
pub(super) fn merge_segment_responses(
    responses: Vec<Response>,
    segment_duration: Duration,
) -> Response {
    let mut metadata = serde_json::Value::Null;
    let mut speaker_labels = Vec::new();
    let mut speaker_segments = Vec::new();
    let mut speaker_offset = 0;
    let mut transcripts: Vec<String> = Vec::new();
    let mut words = Vec::new();

    for (index, response) in responses.into_iter().enumerate() {
        let offset = segment_duration.as_secs_f64() * index as f64;
        let segment_speaker_labels = response
            .metadata
            .get("speaker_labels")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        speaker_labels.extend(segment_speaker_labels.iter().cloned());
        speaker_segments.extend(
            response
                .metadata
                .get("speaker_segments")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
                .cloned()
                .map(|mut segment| {
                    for field in ["start", "end"] {
                        if let Some(value) = segment.get_mut(field)
                            && let Some(time) = value.as_f64()
                        {
                            *value = serde_json::json!(time + offset);
                        }
                    }
                    segment
                }),
        );
        if metadata.is_null() {
            metadata = response.metadata;
        }

        let Some(alternative) = response
            .results
            .channels
            .into_iter()
            .next()
            .and_then(|channel| channel.alternatives.into_iter().next())
        else {
            continue;
        };

        let transcript = alternative.transcript.trim();
        if !transcript.is_empty() {
            transcripts.push(transcript.to_string());
        }
        let segment_speaker_count = alternative
            .words
            .iter()
            .filter_map(|word| word.speaker)
            .max()
            .map_or(0, |speaker| speaker + 1)
            .max(segment_speaker_labels.len());
        words.extend(alternative.words.into_iter().map(|mut word| {
            word.start += offset;
            word.end += offset;
            word.speaker = word.speaker.map(|speaker| speaker + speaker_offset);
            word
        }));
        speaker_offset += segment_speaker_count;
    }

    if let Some(object) = metadata.as_object_mut() {
        if !speaker_labels.is_empty() {
            object.insert(
                "speaker_labels".to_string(),
                serde_json::Value::Array(speaker_labels),
            );
        }
        if !speaker_segments.is_empty() {
            object.insert(
                "speaker_segments".to_string(),
                serde_json::Value::Array(speaker_segments),
            );
        }
    }

    Response {
        metadata: if metadata.is_null() {
            serde_json::json!({})
        } else {
            metadata
        },
        results: Results {
            channels: vec![Channel {
                alternatives: vec![Alternatives {
                    transcript: transcripts.join(" "),
                    confidence: 1.0,
                    words,
                }],
            }],
        },
    }
}

pub(super) async fn run_direct_batch_with_timeout<A: BatchSttAdapter>(
    provider: &str,
    params: BatchParams,
    listen_params: owhisper_interface::ListenParams,
    timeout: Duration,
    upload: UploadProgress,
) -> crate::Result<BatchRunOutput> {
    let span = session_span(&params.session_id);

    async {
        let client = owhisper_client::BatchClient::<A>::builder()
            .api_base(params.base_url.clone())
            .api_key(params.api_key.clone())
            .params(listen_params)
            .upload_progress(Arc::new(move |sent, body_len| {
                upload.report(sent, body_len)
            }))
            .build();

        tracing::debug!("transcribing file: {}", params.file_path);
        let response =
            match tokio::time::timeout(timeout, client.transcribe_file(&params.file_path)).await {
                Ok(Ok(response)) => response,
                Ok(Err(err)) => {
                    let raw_error = format!("{err:?}");
                    let message = format_user_friendly_error(&raw_error);
                    tracing::error!(
                        error = %raw_error,
                        anarlog.error.user_message = %message,
                        "batch transcription failed"
                    );
                    return Err(crate::BatchFailure::DirectRequestFailed {
                        provider: provider.to_string(),
                        message,
                    }
                    .into());
                }
                Err(_) => {
                    tracing::error!(
                        timeout_seconds = timeout.as_secs(),
                        "batch transcription timed out"
                    );
                    return Err(crate::BatchFailure::DirectRequestTimedOut {
                        provider: provider.to_string(),
                        timeout_seconds: timeout.as_secs(),
                    }
                    .into());
                }
            };
        tracing::info!("batch transcription completed");

        Ok(BatchRunOutput {
            session_id: params.session_id,
            mode: BatchRunMode::Direct,
            response,
        })
    }
    .instrument(span)
    .await
}

pub(super) fn direct_batch_timeout_for_audio(audio_duration: Option<Duration>) -> Duration {
    let timeout = audio_duration
        .map(|duration| {
            duration
                .saturating_mul(DIRECT_BATCH_AUDIO_DURATION_MULTIPLIER)
                .saturating_add(DIRECT_BATCH_TIMEOUT_BUFFER)
        })
        .unwrap_or(DIRECT_BATCH_TIMEOUT_FLOOR);

    timeout
        .max(DIRECT_BATCH_TIMEOUT_FLOOR)
        .min(DIRECT_BATCH_TIMEOUT_CEILING)
}
