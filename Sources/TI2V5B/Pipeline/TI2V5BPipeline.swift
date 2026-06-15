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
    /// Single-expert Wan DiT (TI2V-5B is `dual_model: false`).
    public let dit: WanModel
    /// 48-ch channels-last vae22 — decode (latent→frames) and encode (i2v image→latent).
    public let vaeDecoder: Wan22VAEDecoder
    public let vaeEncoder: Wan22VAEEncoder
    /// Checkpoint dir — kept so umT5 can be (re)loaded per request and evicted before
    /// denoise (§2.4), rather than held resident.
    public let modelDir: URL
    public let tokenizer: any Tokenizer

    public init(
        config: WanConfig, dit: WanModel, vaeDecoder: Wan22VAEDecoder,
        vaeEncoder: Wan22VAEEncoder, modelDir: URL, tokenizer: any Tokenizer
    ) {
        self.config = config
        self.dit = dit
        self.vaeDecoder = vaeDecoder
        self.vaeEncoder = vaeEncoder
        self.modelDir = modelDir
        self.tokenizer = tokenizer
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
    public static func fromPretrained(
        modelDir: URL, quantization explicitQuantization: WanQuantization? = nil,
        ditDType: DType = .float32
    ) async throws -> TI2V5BPipeline {
        let config = try WanConfig.load(
            from: modelDir.appendingPathComponent("config.json"))
        let quantization = explicitQuantization ?? config.quantization

        // --- DiT (single expert) ---
        let dit = WanModel(config)
        if let quantization {
            WeightLoader.applyQuantization(to: dit, quantization: quantization)
        }
        var ditWeights = try WeightLoader.loadSafetensors(
            url: modelDir.appendingPathComponent("model.safetensors"))
        if quantization == nil, ditDType == .float32 {
            ditWeights = ditWeights.mapValues { $0.asType(.float32) }  // video-scale correctness
        }
        WeightLoader.materialize(ditWeights)
        try dit.update(
            parameters: ModuleParameters.unflattened(ditWeights),
            verify: [.noUnusedKeys])
        eval(dit.parameters())

        // --- vae22 (fp32), decoder + encoder from the one vae.safetensors ---
        let (vaeDecoder, vaeEncoder) = try loadVAE(
            url: modelDir.appendingPathComponent("vae.safetensors"),
            zDim: config.vaeZDim)

        let tokenizer = try await AutoTokenizer.from(pretrained: umt5TokenizerRepo)
        return TI2V5BPipeline(
            config: config, dit: dit, vaeDecoder: vaeDecoder, vaeEncoder: vaeEncoder,
            modelDir: modelDir, tokenizer: tokenizer)
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

        let latent = try denoiseTI2V(
            dit: dit, config: config, contextCond: contextCond, contextNull: contextNull,
            noise: noise, steps: steps ?? config.sampleSteps, shift: config.sampleShift,
            guideScale: guideScale ?? (config.sampleGuideScale.first ?? 5.0),
            scheduler: scheduler, onStep: onStep)

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
    /// → channels-last [1, T_lat, H_lat, W_lat, C] → denormalize → decode → frames
    /// [1, T', H', W', 3] in [-1, 1]. Run on the CPU stream (fp32 VAE, watchdog-safe).
    func decodeLatent(_ latent: MLXArray) -> MLXArray {
        Device.withDefaultDevice(.cpu) {
            let z = latent.transposed(1, 2, 3, 0).expandedDimensions(axis: 0)
            let video = vaeDecoder(denormalizeLatents22(z))
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

        let latent = try denoiseTI2V(
            dit: dit, config: config, contextCond: contextCond, contextNull: contextNull,
            noise: noise, steps: steps ?? config.sampleSteps, shift: config.sampleShift,
            guideScale: guideScale ?? (config.sampleGuideScale.first ?? 5.0),
            scheduler: scheduler,
            i2v: I2VCondition(zImg: zImg, mask: mask, maskTokens: maskTokens),
            onStep: onStep)

        return decodeLatent(latent)
    }
}
