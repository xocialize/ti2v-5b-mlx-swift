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
    public static func fromPretrained(
        modelDir: URL, quantization explicitQuantization: WanQuantization? = nil
    ) async throws -> TI2V5BPipeline {
        let config = try WanConfig.load(
            from: modelDir.appendingPathComponent("config.json"))
        let quantization = explicitQuantization ?? config.quantization

        // --- DiT (single expert), checkpoint dtype (bf16) ---
        let dit = WanModel(config)
        if let quantization {
            WeightLoader.applyQuantization(to: dit, quantization: quantization)
        }
        let ditWeights = try WeightLoader.loadSafetensors(
            url: modelDir.appendingPathComponent("model.safetensors"))
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

    /// Text-to-video. Relay: umT5 encode→evict → DiT denoise → vae22 decode.
    /// - Returns: decoded frames [1, T', H', W', 3] in [-1, 1] (channels-last vae22).
    public func t2v(
        prompt: String,
        negativePrompt: String? = nil,
        width: Int = TI2V5BDefaults.width,
        height: Int = TI2V5BDefaults.height,
        numFrames: Int = TI2V5BDefaults.numFrames,
        steps: Int? = nil,
        guideScale: Double? = nil,
        seed: UInt64? = nil
    ) throws -> MLXArray {
        // Wiring sketch (task #12):
        //   let contexts = try withTextEncoder { enc in encodePrompt(enc, prompt, neg) }
        //   let latent = denoise(contexts, shape, steps, guideScale, seed)   // sampler
        //   return vaeDecoder(denormalizeLatents22(latent))
        throw TI2V5BError.notImplemented("t2v denoise loop (sampler) — task #12")
    }

    /// Image+text-to-video. As t2v, but the input image is vae22-encoded to a latent
    /// and used as i2v conditioning (concatenated / masked into the noised latent).
    /// - Parameter image: [1, 1, H, W, 3] in [-1, 1] (channels-last).
    public func i2v(
        image: MLXArray,
        prompt: String,
        negativePrompt: String? = nil,
        numFrames: Int = TI2V5BDefaults.numFrames,
        steps: Int? = nil,
        guideScale: Double? = nil,
        seed: UInt64? = nil
    ) throws -> MLXArray {
        // Wiring sketch (task #12):
        //   let condLatent = vaeEncoder(image)          // i2v conditioning latent
        //   let contexts = try withTextEncoder { ... }
        //   let latent = denoise(contexts, condLatent, ...)
        //   return vaeDecoder(denormalizeLatents22(latent))
        throw TI2V5BError.notImplemented("i2v conditioning + denoise — task #12")
    }
}
