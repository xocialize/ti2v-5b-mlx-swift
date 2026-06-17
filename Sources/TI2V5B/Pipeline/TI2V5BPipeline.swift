// TI2V5BPipeline — the text+image-to-video entry for Wan2.2-TI2V-5B. Owns the
// component loads (single-expert DiT + vae22 decoder/encoder + tokenizer) and the
// prompt(+image) → frames path. The DiT is a plain `WanModel` (dual_model:false),
// so there is NO high/low expert switch. umT5 is paged in per request and evicted
// before denoise (the §2.4 lever) — the consumer-tier memory recipe.
//
// SCAFFOLD STATE (task #11): component loading + §2.4 eviction are real and
// load-smoke-tested by RunTI2V5B. The denoise loop and i2v conditioning are wired
// but stubbed (`TI2V5BError.notImplemented`) — task #12.

import Foundation
import MLX
import MLXNN
import MLXRandom
import Tokenizers
import WanCore

public final class TI2V5BPipeline: @unchecked Sendable {
    public let config: WanConfig
    /// Single-expert Wan DiT (`dual_model: false`). nil when `pageDiT` — then it is
    /// loaded per request and evicted before the VAE decode (§2.12 max-phase residency:
    /// only one heavy component — umT5 OR DiT OR VAE — resident at a time).
    public let dit: WanModel?
    /// 48-ch channels-last vae22 — decode (latent→frames) and encode (i2v image→latent).
    public let vaeDecoder: Wan22VAEDecoder
    public let vaeEncoder: Wan22VAEEncoder
    /// Checkpoint dir — kept so umT5 (always) and the DiT (when `pageDiT`) can be
    /// (re)loaded per request and evicted, rather than held resident (§2.4 / §2.12).
    public let modelDir: URL
    public let tokenizer: any Tokenizer
    let quantization: WanQuantization?
    let ditDType: DType

    public init(
        config: WanConfig, dit: WanModel?, vaeDecoder: Wan22VAEDecoder,
        vaeEncoder: Wan22VAEEncoder, modelDir: URL, tokenizer: any Tokenizer,
        quantization: WanQuantization?, ditDType: DType
    ) {
        self.config = config
        self.dit = dit
        self.vaeDecoder = vaeDecoder
        self.vaeEncoder = vaeEncoder
        self.modelDir = modelDir
        self.tokenizer = tokenizer
        self.quantization = quantization
        self.ditDType = ditDType
    }

    /// Load all components from a converted checkpoint directory (flat layout:
    /// `model.safetensors` + `vae.safetensors` + `t5_encoder.safetensors` +
    /// `config.json`). The tokenizer comes from google/umt5-xxl (HF), like mlx-video.
    /// - ditDType: the DiT compute precision. Defaults to **fp32** — REQUIRED for correct
    ///   video-scale (large-seqLen) output: Metal bf16 attention over long sequences is
    ///   numerically unstable (NaN) AND, even mitigated, too imprecise (the latent
    ///   over-grows; fp32 matches the oracle). bf16 is fine only for small-seqLen (e.g.
    ///   single-frame t2i) — pass `.bfloat16` there to halve the DiT footprint. Ignored
    ///   for quantized checkpoints (the quant path owns dtype).
    /// - pageDiT: when true, the DiT is NOT held resident — it is paged in per request
    ///   for denoise and evicted before the VAE decode (§2.12 max-phase: peak =
    ///   max(umT5, DiT, decode) not their sum). The consumer-tier memory lever (costs a
    ///   ~per-generation DiT reload). Default false (DiT resident — faster for repeated
    ///   runs / high-memory tiers).
    public static func fromPretrained(
        modelDir: URL, quantization explicitQuantization: WanQuantization? = nil,
        ditDType: DType = .float32, pageDiT: Bool = false
    ) async throws -> TI2V5BPipeline {
        let config = try WanConfig.load(
            from: modelDir.appendingPathComponent("config.json"))
        let quantization = explicitQuantization ?? config.quantization

        // DiT: resident now, or paged per request when pageDiT.
        let dit = pageDiT
            ? nil
            : try loadDiT(modelDir: modelDir, config: config, quantization: quantization, ditDType: ditDType)

        // --- vae22 (fp32), decoder + encoder from the one vae.safetensors ---
        let (vaeDecoder, vaeEncoder) = try loadVAE(
            url: modelDir.appendingPathComponent("vae.safetensors"),
            zDim: config.vaeZDim)

        let tokenizer = try await AutoTokenizer.from(pretrained: umt5TokenizerRepo)
        return TI2V5BPipeline(
            config: config, dit: dit, vaeDecoder: vaeDecoder, vaeEncoder: vaeEncoder,
            modelDir: modelDir, tokenizer: tokenizer,
            quantization: quantization, ditDType: ditDType)
    }

    /// Build + load a single-expert DiT (fp32 compute for video-scale correctness; int4
    /// keeps the quant path). Drops the stray int4 `freqs` table.
    static func loadDiT(
        modelDir: URL, config: WanConfig, quantization: WanQuantization?, ditDType: DType
    ) throws -> WanModel {
        let dit = WanModel(config)
        if let quantization {
            WeightLoader.applyQuantization(to: dit, quantization: quantization)
        }
        var ditWeights = try WeightLoader.loadSafetensors(
            url: modelDir.appendingPathComponent("model.safetensors"))
        ditWeights = ditWeights.filter { $0.key != "freqs" }
        if quantization == nil, ditDType == .float32 {
            ditWeights = ditWeights.mapValues { $0.asType(.float32) }
        }
        WeightLoader.materialize(ditWeights)
        try dit.update(
            parameters: ModuleParameters.unflattened(ditWeights), verify: [.noUnusedKeys])
        eval(dit.parameters())
        return dit
    }

    /// Run `body` with the DiT (resident, or paged-in then evicted before return — the
    /// §2.12 lever). ⚠️ `body` MUST `eval` everything it returns (else the latent holds
    /// the DiT graph past the evict, defeating it).
    func withDiT<R>(_ body: (WanModel) throws -> R) throws -> R {
        if let dit { return try body(dit) }  // resident
        var paged: WanModel? = try Self.loadDiT(
            modelDir: modelDir, config: config, quantization: quantization, ditDType: ditDType)
        let result = try body(paged!)
        paged = nil
        MLX.GPU.clearCache()  // reclaim the ~20 GB (fp32) before the VAE decode
        return result
    }

    /// Build + load both vae22 halves (decoder uses `decoder.`/`conv2.` keys, encoder
    /// uses `encoder.`/`conv1.`). The VAE runs fp32 (parity), on the CPU stream during
    /// load so the upcast doesn't trip the GPU watchdog.
    private static func loadVAE(
        url: URL, zDim: Int
    ) throws -> (Wan22VAEDecoder, Wan22VAEEncoder) {
        try Device.withDefaultDevice(.cpu) {
            let all = try WeightLoader.loadSafetensors(url: url)
                .mapValues { $0.asType(.float32) }

            let decoder = Wan22VAEDecoder(zDim: zDim)
            let decW = all.filter {
                $0.key.hasPrefix("decoder.") || $0.key.hasPrefix("conv2.")
            }
            try decoder.update(
                parameters: ModuleParameters.unflattened(decW), verify: [.all])

            let encoder = Wan22VAEEncoder(zDim: zDim)
            let encW = all.filter {
                $0.key.hasPrefix("encoder.") || $0.key.hasPrefix("conv1.")
            }
            try encoder.update(
                parameters: ModuleParameters.unflattened(encW), verify: [.all])

            eval(decoder.parameters(), encoder.parameters())
            return (decoder, encoder)
        }
    }

    // MARK: - umT5 (§2.4 post-encode eviction)

    /// Load the fp32 umT5 encoder from the checkpoint (fp32 like mlx-video's
    /// `load_t5_encoder`). On demand, not held resident.
    private func loadTextEncoder() throws -> UMT5EncoderModel {
        let textEncoder = UMT5EncoderModel.fromConfig(config)
        let t5Weights = try WeightLoader.loadVerifiedSafetensors(
            url: modelDir.appendingPathComponent("t5_encoder.safetensors"),
            expectedKeys: BerniniWeightKeys.t5Keys(layers: config.t5NumLayers)
        ).mapValues { $0.asType(.float32) }
        WeightLoader.materialize(t5Weights)
        try textEncoder.update(
            parameters: ModuleParameters.unflattened(t5Weights),
            verify: [.noUnusedKeys])
        return textEncoder
    }

    /// §2.4 T5 eviction: load umT5, run `body` to produce its text contexts, then drop
    /// the encoder and reclaim its working set before returning — so the denoise loop
    /// never co-resides with the encoder. ⚠️ `body` MUST `eval` everything it returns.
    func withTextEncoder<R>(_ body: (UMT5EncoderModel) throws -> R) throws -> R {
        var encoder: UMT5EncoderModel? = try loadTextEncoder()
        let result = try body(encoder!)
        encoder = nil
        MLX.GPU.clearCache()
        return result
    }

    // MARK: - Generation (task #12)

    /// Text-to-video. Relay: umT5 encode→evict → single-expert DiT denoise → vae22
    /// decode. Returns decoded frames [1, T', H', W', 3] in [-1, 1] (channels-last).
    public func t2v(
        prompt: String,
        negativePrompt: String? = nil,
        width: Int = TI2V5BDefaults.width,
        height: Int = TI2V5BDefaults.height,
        numFrames: Int = TI2V5BDefaults.numFrames,
        steps: Int? = nil,
        guideScale: Double? = nil,
        scheduler: TI2VScheduler = .unipc,
        seed: UInt64? = nil,
        onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
    ) throws -> MLXArray {
        let negative = negativePrompt ?? config.sampleNegPrompt

        // §2.4: page umT5 in, encode cond/uncond, evict before denoise.
        let (contextCond, contextNull) = try withTextEncoder { enc -> (MLXArray, MLXArray) in
            let c = encodeText(
                encoder: enc, tokenizer: tokenizer, prompt: prompt, textLen: config.textLen)
            let n = encodeText(
                encoder: enc, tokenizer: tokenizer, prompt: negative, textLen: config.textLen)
            eval(c, n)
            return (c, n)
        }

        // Latent geometry from the vae strides ([4,16,16]); channels-first.
        let tLat = (numFrames - 1) / config.vaeStride[0] + 1
        let hLat = height / config.vaeStride[1]
        let wLat = width / config.vaeStride[2]
        if let seed { MLXRandom.seed(seed) }
        let noise = MLXRandom.normal([config.vaeZDim, tLat, hLat, wLat])

        let latent = try withDiT { dit -> MLXArray in
            let l = try denoiseTI2V(
                dit: dit, config: config, contextCond: contextCond, contextNull: contextNull,
                noise: noise, steps: steps ?? config.sampleSteps, shift: config.sampleShift,
                guideScale: guideScale ?? (config.sampleGuideScale.first ?? 5.0),
                scheduler: scheduler, onStep: onStep)
            eval(l)  // §2.12: materialize before the DiT is evicted for the decode
            return l
        }

        return decodeLatent(latent)
    }

    /// Text-to-image = single-frame t2v. Returns [1, 1, H', W', 3] in [-1, 1].
    public func t2i(
        prompt: String, negativePrompt: String? = nil,
        width: Int = TI2V5BDefaults.width, height: Int = TI2V5BDefaults.height,
        steps: Int? = nil, guideScale: Double? = nil, seed: UInt64? = nil
    ) throws -> MLXArray {
        try t2v(
            prompt: prompt, negativePrompt: negativePrompt, width: width, height: height,
            numFrames: 1, steps: steps, guideScale: guideScale, seed: seed)
    }

    /// Decode a channels-first DiT latent [C, T_lat, H_lat, W_lat] through vae22:
    /// → channels-last [1, T_lat, H_lat, W_lat, C] → denormalize → STREAMING decode →
    /// frames [1, T', H', W', 3] in [-1, 1]. Run on the CPU stream (fp32 VAE).
    ///
    /// Uses `decodeStreaming22` (temporal-chunked, bit-identical to whole-seq) — the
    /// whole-sequence decode costs ~27 GB PER LATENT FRAME and OOMs at video length;
    /// chunking to 1 latent frame caps the decode peak ~flat (the 720p memory unlock).
    func decodeLatent(_ latent: MLXArray) -> MLXArray {
        // phys_footprint (the governor's basis) tracks MLX's buffer-cache high-water, NOT
        // just live allocations — and the decisive datapoint is that GPU *active* peak is
        // ~76 GB while phys is ~110: the ~34 GB delta is the buffer CACHE (freed full-res
        // 1024-ch conv intermediates MLX retains for reuse). A `clearCache` can't cap it —
        // the peak is transient *inside* the decode (and at 5f the decode is a single
        // chunk, so a per-chunk clear has no boundary to fire at). The cap that DOES bound
        // it is `Memory.cacheLimit`: with a low/zero limit MLX reclaims freed buffers on
        // the NEXT allocation instead of caching them, so the cache never accumulates to
        // the peak → phys collapses toward the ~76 GB active set. Scoped to the decode and
        // restored after (denoise already clears per-step).
        //
        // DEFAULT 2048 MB (measured 2026-06-15, int4 5f @ 720p): a 2 GB working cache holds phys at
        // the SAME 41.1 GB as a zero cap (the 2 GB doesn't accumulate into the high-water) while
        // giving the BEST wall time — 172.4 s vs 211.8 s at cap=0 (−19%, dodges the zero-cap realloc
        // churn) and even vs 195.8 s uncapped (the bounded cache also avoids the unbounded default's
        // fragmentation). Env `DECODE_CACHE_MB` overrides (0 = max reclaim; raise for more reuse).
        let prevCacheLimit = Memory.cacheLimit
        let capMB = ProcessInfo.processInfo.environment["DECODE_CACHE_MB"].flatMap { Int($0) } ?? 2048
        Memory.cacheLimit = capMB * 1_000_000
        defer { Memory.cacheLimit = prevCacheLimit }
        MLX.GPU.clearCache()  // drop the denoise cache before the capped decode begins
        // DECODE_DEVICE=gpu opts the streaming vae22 decode onto the GPU stream (the 16-ch VACE win:
        // >27 min CPU → 46.6 s GPU, bounded, no watchdog at chunkLat=1). Default stays .cpu here:
        // unlike the 16-ch path, the 48-ch vae22 720p GPU-decode envelope (~76 GB active per the note
        // above; a heavier single-chunk command buffer) is NOT yet validated for watchdog/OOM — flip
        // the default once the testing agent confirms GPU vae22 decode at 720p.
        let decodeDevice: Device = (ProcessInfo.processInfo.environment["DECODE_DEVICE"] == "gpu") ? .gpu : .cpu
        return Device.withDefaultDevice(decodeDevice) {
            let z = latent.transposed(1, 2, 3, 0).expandedDimensions(axis: 0)
            let video = decodeStreaming22(vaeDecoder, denormalizeLatents22(z))
            eval(video)
            return video
        }
    }

    /// Image+text-to-video (TI2V-5B mask-blend conditioning). The input image is
    /// vae22-encoded to a normalized latent that occupies (and stays frozen at) video
    /// frame 0; the rest is generated. Per the oracle, TI2V-5B uses MASK-BLEND (not the
    /// channel-concat `y:` path). Returns frames [1, T', H', W', 3] in [-1, 1].
    /// - Parameter image: [1, 1, H, W, 3] in [-1, 1] (channels-last).
    public func i2v(
        image: MLXArray,
        prompt: String,
        negativePrompt: String? = nil,
        numFrames: Int = TI2V5BDefaults.numFrames,
        steps: Int? = nil,
        guideScale: Double? = nil,
        scheduler: TI2VScheduler = .unipc,
        seed: UInt64? = nil,
        onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
    ) throws -> MLXArray {
        let negative = negativePrompt ?? config.sampleNegPrompt

        // §2.4: encode cond/uncond, evict umT5 before denoise.
        let (contextCond, contextNull) = try withTextEncoder { enc -> (MLXArray, MLXArray) in
            let c = encodeText(
                encoder: enc, tokenizer: tokenizer, prompt: prompt, textLen: config.textLen)
            let n = encodeText(
                encoder: enc, tokenizer: tokenizer, prompt: negative, textLen: config.textLen)
            eval(c, n)
            return (c, n)
        }

        // vae22-encode the image → normalized latent, channels-first [C, 1, Hl, Wl].
        let zImg = Device.withDefaultDevice(.cpu) { () -> MLXArray in
            let zCL = vaeEncoder(image)             // [1, 1, Hl, Wl, C] normalized
            let z = zCL[0].transposed(3, 0, 1, 2)   // [C, 1, Hl, Wl]
            eval(z)
            return z
        }
        let (hLat, wLat) = (zImg.dim(2), zImg.dim(3))
        let tLat = (numFrames - 1) / config.vaeStride[0] + 1

        let (mask, maskTokens) = buildI2VMask(
            channels: config.vaeZDim, tLat: tLat, hLat: hLat, wLat: wLat,
            patchSize: config.patchSize)

        if let seed { MLXRandom.seed(seed) }
        let noise = MLXRandom.normal([config.vaeZDim, tLat, hLat, wLat])

        let latent = try withDiT { dit -> MLXArray in
            let l = try denoiseTI2V(
                dit: dit, config: config, contextCond: contextCond, contextNull: contextNull,
                noise: noise, steps: steps ?? config.sampleSteps, shift: config.sampleShift,
                guideScale: guideScale ?? (config.sampleGuideScale.first ?? 5.0),
                scheduler: scheduler,
                i2v: I2VCondition(zImg: zImg, mask: mask, maskTokens: maskTokens),
                onStep: onStep)
            eval(l)  // §2.12: materialize before the DiT is evicted for the decode
            return l
        }

        return decodeLatent(latent)
    }
}
