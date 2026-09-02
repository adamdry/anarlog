import { t } from "@lingui/core/macro";
import type { StoreApi } from "zustand";

import { commands as notificationCommands } from "@anlg/plugin-notification";
import {
  type BatchErrorCode,
  type TranscriptionParams,
  commands as transcriptionCommands,
  events as transcriptionEvents,
} from "@anlg/plugin-transcription";

import {
  EMPTY_BATCH_TRANSCRIPT_ERROR,
  type BatchActions,
  type BatchState,
} from "./batch";

import { trackAnalyticsEvent } from "~/analytics";
import { requestAppAttention } from "~/shared/app-attention";
import { isAppWindowInactive } from "~/shared/window-activity";
import { createBatchCompletedNotificationKey } from "~/stt/batch-completed-notification";
import { BatchResponseProcessingError } from "~/stt/batch-response-processing-error";

type BatchStore = BatchActions & BatchState;

const BATCH_COMPLETED_NOTIFICATION_TIMEOUT_SECONDS = 15;

export async function showBatchCompletedNotification(
  sessionId: string,
  options?: { force?: boolean },
) {
  if (!options?.force && !(await isAppWindowInactive())) {
    return;
  }

  try {
    const result = await notificationCommands.showNotification({
      key: createBatchCompletedNotificationKey(sessionId),
      title: t`Transcription complete`,
      message: t`Your transcript is ready.`,
      timeout: {
        secs: BATCH_COMPLETED_NOTIFICATION_TIMEOUT_SECONDS,
        nanos: 0,
      },
      source: { type: "session", session_id: sessionId },
      start_time: null,
      participants: null,
      event_details: null,
      action_label: t`Open Anarlog`,
      action_variant: null,
      options: null,
      footer: null,
      icon: null,
    });

    if (result.status === "error") {
      console.error(
        "[runBatch] failed to show completion notification",
        result.error,
      );
    }
  } catch (error) {
    console.error("[runBatch] failed to show completion notification", error);
  }
}

export const runBatchSession = async <T extends BatchStore>(
  get: StoreApi<T>["getState"],
  sessionId: string,
  params: TranscriptionParams,
  options?: { notifyOnCompletion?: boolean },
) => {
  get().handleBatchStarted(sessionId);

  let unlisten: (() => void) | undefined;
  let settled = false;

  const cleanup = (clearSession = true) => {
    if (unlisten) {
      unlisten();
      unlisten = undefined;
    }

    get().clearBatchPersist(sessionId);

    if (clearSession) {
      get().clearBatchSession(sessionId);
    }
  };

  const resolveSuccess = (
    output: {
      response: Parameters<BatchStore["handleBatchResponse"]>[1];
    },
    resolve: () => void,
    reject: (reason?: unknown) => void,
  ) => {
    if (settled) {
      return;
    }

    settled = true;

    try {
      const handled = get().handleBatchResponse(sessionId, output.response);
      if (handled === false) {
        throw new Error(EMPTY_BATCH_TRANSCRIPT_ERROR);
      }
      trackAnalyticsEvent("transcription_completed", {
        mode: "batch",
        provider: params.provider,
      });
      cleanup();
    } catch (error) {
      console.error("[runBatch] error handling batch response", error);
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      get().handleBatchFailed(sessionId, errorMessage);
      trackAnalyticsEvent("transcription_failed", {
        mode: "batch",
        failure_stage: "persist",
      });
      cleanup(false);
      reject(
        error instanceof Error && error.message === EMPTY_BATCH_TRANSCRIPT_ERROR
          ? error
          : new BatchResponseProcessingError(error),
      );
      return;
    }

    resolve();
  };

  const rejectFailure = (
    error: unknown,
    reject: (reason?: unknown) => void,
    options?: {
      clearSession?: boolean;
      terminalReason?: "failed" | "timed_out";
      errorCode?: BatchErrorCode;
    },
  ) => {
    if (settled) {
      return;
    }

    settled = true;

    const errorMessage = error instanceof Error ? error.message : String(error);
    get().handleBatchFailed(
      sessionId,
      errorMessage,
      options?.terminalReason,
      options?.errorCode,
    );
    trackAnalyticsEvent("transcription_failed", {
      mode: "batch",
      failure_stage: options?.terminalReason ?? "provider",
      error_code: options?.errorCode ?? "unknown",
      provider: params.provider,
    });
    cleanup(options?.clearSession ?? false);
    reject(error);
  };

  const rejectStopped = (reject: (reason?: unknown) => void) => {
    if (settled) {
      return;
    }

    settled = true;
    get().handleBatchStopped(sessionId);
    cleanup(false);
    reject(new Error("Transcription stopped."));
  };

  await new Promise<void>((resolve, reject) => {
    transcriptionEvents.transcriptionEvent
      .listen(({ payload }) => {
        if (settled || payload.session_id !== sessionId) {
          return;
        }

        if (payload.type === "started") {
          return;
        }

        if (payload.type === "progress") {
          get().handleBatchResponseStreamed(sessionId, payload.event);
          return;
        }

        if (payload.type === "completed") {
          resolveSuccess(
            {
              response: payload.response,
            },
            resolve,
            reject,
          );
          return;
        }

        if (payload.type === "stopped") {
          rejectStopped(reject);
          return;
        }

        if (payload.type === "failed") {
          rejectFailure(payload.error, reject, {
            terminalReason:
              payload.code === "timed_out" ? "timed_out" : "failed",
            errorCode: payload.code,
          });
        }
      })
      .then((fn) => {
        unlisten = fn;

        transcriptionCommands
          .startTranscription(params)
          .then((result) => {
            if (settled) {
              return;
            }

            if (result.status === "error") {
              console.error(result.error);
              rejectFailure(result.error, reject);
            }
          })
          .catch((error) => {
            console.error(error);
            rejectFailure(error, reject);
          });
      })
      .catch((error) => {
        console.error(error);
        rejectFailure(error, reject);
      });
  });

  if (options?.notifyOnCompletion !== false) {
    await showBatchCompletedNotification(sessionId);
    void requestAppAttention();
  }
};
