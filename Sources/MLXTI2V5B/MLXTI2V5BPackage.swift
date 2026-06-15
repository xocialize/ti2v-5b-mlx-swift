import Foundation
import MLX
import MLXToolKit
import TI2V5B
import WanCore

/// MLXEngine package: Wan2.2-TI2V-5B (single-expert text+image-to-video) exposing the
/// canonical `textToVideo` + `textToImage` surfaces from ONE loaded pipeline. Unlike
/// Bernini's A14B (dual-expert, text-only), TI2V-5B is `dual_model:false` and natively
/// supports image conditioning — so `textToVideo` with a `T2VRequest.initImage` runs the
/// i2v mask-blend path (the init image is frozen at frame 0).
///
/// Engine-owned lifecycle (C13): construct from `TI2V5BConfiguration`, page the working
/// set in with `load()`, drive `run(_:)`, reclaim with `unload()`. Lifecycle is isolated
/// to `InferenceActor`; the non-`Sendable` `TI2V5BPipeline` never crosses the boundary.
/// Cancellation is honored at every denoising-step boundary via the core's `onStep`.
///
/// The generation engine (vae22 both directions, t2v + i2v denoise) is bit-exact
/// parity-locked against the mlx-video oracle.
@InferenceActor
public final class MLXTI2V5BPackage: ModelPackage {
    public typealias Configuration = TI2V5BConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: "Wan-AI/Wan2.2-TI2V-5B",
                revision: "main",
                tier: 1
            ),
            requirements: RequirementsManifest(
                // RE-GROUNDED 2026-06-15 on the MEASURED 720p peak (Xcode agent, int4 5f @
                // 1280×704, fp32 compute). residentBytes = the OS phys_footprint (task_info), the
                // governor's true max-simultaneous basis — NOT MLX's `Memory.peakMemory`, which under
                // the decode cache cap counts cumulative allocations (it read 76 GB while phys was
                // 41 — misleading post-cap; profiler caveat logged separately).
                //
                // The 720p admission was unlocked by capping the buffer cache during decode
                // (`Memory.cacheLimit = 0` in `decodeLatent`, ti2v c98cdc6): without it the freed
                // full-res decode intermediates accumulated to phys ~110 GB (> budget); with it,
                // continuous reclamation collapses the transient high-water → **int4 5f @ 720p =
                // 41.1 GB phys** (max RSS 35.8), bit-identical output. The earlier "~47.8 GB spatial
                // decode floor" was itself cache-inflated; the live per-chunk working set is far
                // lower, so streaming (temporal) decode + the cap bound the decode at ANY length.
                //
                // int4 = measured 41 GB + headroom. bf16 (→ fp32 compute) is DERIVED, not measured
                // at 720p: same fp32 activations as int4, +~17 GB for fp32 DiT weights (20 vs int4's
                // ~3) → ~58 GB; the peak phase is now the DENOISE (decode is capped). Static-manifest
                // caveat: one figure per quant, validated at 5f. Higher frame counts raise the
                // DENOISE activations (seqLen scaling = the chunked-attention wall, separate) — a
                // chipFloor drop below .max should be validated at the production frame count, not 5f.
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 58_000_000_000),  // → fp32 compute (derived)
                    QuantFootprint(quant: .int4, residentBytes: 45_000_000_000),  // measured 41.1 + headroom
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                // KEPT .max pending a production-frame-count re-measure: int4 5f is 41 GB (admits with
                // ~55 GB headroom on the 96.2 GB pro budget, and would fit a 64 GB-class machine), but
                // 81f denoise activations are unmeasured and scale with seqLen. Drop only after that.
                chipFloor: .max
            ),
            specialties: [
                SpecialtyWeight(.general, strength: 0.6),
            ],
            surfaces: [
                T2VContract.descriptor(
                    name: "ti2v-5b-t2v",
                    summary: "Wan2.2-TI2V-5B single-expert text/image-to-video (MLX). "
                        + "Provide a prompt for text-to-video, or a `T2VRequest.initImage` to "
                        + "animate from a first frame (i2v). 704×1280 native, frames 4n+1. "
                        + "`.fast` mode (DPM++/16) is ~2.5× quicker at near-identical quality.",
                    modes: [.fast, .quality]
                ),
                T2IContract.descriptor(
                    name: "ti2v-5b-t2i",
                    summary: "Text-to-image via single-frame Wan2.2-TI2V-5B diffusion (MLX). "
                        + "`.fast` mode (DPM++/16) is ~2.5× quicker.",
                    modes: [.fast, .quality]
                ),
            ]
        )
    }

    private let configuration: Configuration
    /// The resident pipeline (DiT + vae22 + tokenizer), paged in by `load()`. umT5 is NOT
    /// resident — paged in per request and evicted before denoise (§2.4).
    private var pipeline: TI2V5BPipeline?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard pipeline == nil else { return }
        let directory: URL
        if let explicit = configuration.modelDirectory {
            directory = explicit
        } else {
            directory = try await WeightLoader.snapshotDownload(repoID: configuration.repo)
        }
        pipeline = try await TI2V5BPipeline.fromPretrained(modelDir: directory)
    }

    public func unload() async {
        pipeline = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let pipeline else { throw PackageError.notLoaded }
        switch request.capability {
        case .textToImage:
            guard let t2i = request as? T2IRequest else {
                throw PackageError.configurationMismatch(
                    expected: "T2IRequest", got: String(describing: type(of: request)))
            }
            return try runT2I(t2i, pipeline: pipeline)
        case .textToVideo:
            guard let t2v = request as? T2VRequest else {
                throw PackageError.configurationMismatch(
                    expected: "T2VRequest", got: String(describing: type(of: request)))
            }
            return try await runT2V(t2v, pipeline: pipeline)
        default:
            throw PackageError.unsupportedCapability(request.capability)
        }
    }

    // MARK: - Surfaces

    private func runT2I(_ request: T2IRequest, pipeline: TI2V5BPipeline) throws -> T2IResponse {
        try Task.checkCancellation()
        let sampling = resolveSampling(mode: request.mode, steps: request.steps)
        let frames = try pipeline.t2i(
            prompt: request.prompt,
            negativePrompt: request.negativePrompt,
            width: request.width ?? 1280,
            height: request.height ?? 704,
            steps: sampling.steps,
            guideScale: request.guidanceScale,
            seed: request.seed)
        // vae22 yields ≥1 frame per latent frame (the still is frame 0). Channels-last
        // [1, T, H, W, 3] → frame 0 is [H, W, 3].
        let (data, width, height) = try encodePNG(frame: frames[0, 0, 0..., 0..., 0...])
        return T2IResponse(image: Image(format: .png, data: data, width: width, height: height))
    }

    private func runT2V(_ request: T2VRequest, pipeline: TI2V5BPipeline) async throws
        -> T2VResponse
    {
        try Task.checkCancellation()
        let numFrames = request.numFrames ?? 81
        let fps = request.fps ?? 24
        let width = request.width ?? 1280
        let height = request.height ?? 704
        let sampling = resolveSampling(mode: request.mode, steps: request.steps)
        let onStep: (Int, Int, MLXArray) throws -> Void = { _, _, _ in
            try Task.checkCancellation()  // C13: per-denoising-step cancellation
        }

        let frames: MLXArray
        if let initImage = request.initImage {
            // i2v: the init image is the (frozen) first frame.
            let image = try decodeInitImage(initImage, width: width, height: height)
            frames = try pipeline.i2v(
                image: image, prompt: request.prompt, negativePrompt: request.negativePrompt,
                numFrames: numFrames, steps: sampling.steps,
                guideScale: request.guidanceScale, scheduler: sampling.scheduler,
                seed: request.seed, onStep: onStep)
        } else {
            frames = try pipeline.t2v(
                prompt: request.prompt, negativePrompt: request.negativePrompt,
                width: width, height: height, numFrames: numFrames, steps: sampling.steps,
                guideScale: request.guidanceScale, scheduler: sampling.scheduler,
                seed: request.seed, onStep: onStep)
        }
        return try await framesToVideoResponse(frames, fps: fps)
    }

    private func framesToVideoResponse(_ frames: MLXArray, fps: Double) async throws
        -> T2VResponse
    {
        let mp4 = try await encodeMP4(frames: frames, fps: fps)
        return T2VResponse(
            video: Video(format: .mp4, data: mp4,
                         durationSeconds: Double(frames.dim(1)) / fps, frameRate: fps))
    }
}

extension MLXTI2V5BPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(MLXTI2V5BPackage.self)
    }
}
