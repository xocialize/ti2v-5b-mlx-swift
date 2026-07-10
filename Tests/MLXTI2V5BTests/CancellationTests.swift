// CancellationTests.swift — TI2V-5B through the engine's CAN gate (offline, no MLX
// kernels). CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires before
// the notLoaded guard or weights); CAN-3 is the document of record for the checkpoint
// cadence: every sampler path (t2v / i2v / t2i) threads a throwing `onStep` closure —
// `try Task.checkCancellation()` once per denoising step (TI2V5BSampling, rethrown
// unchanged through the pipeline) — and the wan-core streaming vae22 decode bails per
// temporal chunk (`Task.isCancelled` in decodeStreaming22), with the wrapper's post-core
// `try Task.checkCancellation()` discarding a truncated result.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXTI2V5B

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = MLXTI2V5BPackage(configuration: TI2V5BConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: T2VRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // textToVideo is a long-run capability — the sub-second exemption is not available.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: MLXTI2V5BPackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: MLXTI2V5BPackage.manifest,
            posture: .cadence([
                // Per denoising step: the throwing onStep closure threaded into every
                // pipeline sampler call (t2v/i2v/t2i — MLXTI2V5BPackage.swift →
                // TI2V5BSampling `try onStep?(i, steps, latents)`).
                .init(phase: .denoise, unit: .step),
                // Per vae22-decode temporal chunk: wan-core decodeStreaming22 bails on
                // Task.isCancelled; the wrapper's post-core checkpoint rethrows.
                .init(phase: .decode, unit: .chunk),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
