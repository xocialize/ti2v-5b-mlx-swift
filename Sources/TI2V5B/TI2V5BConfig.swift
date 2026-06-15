// TI2V5BConfig — TI2V-5B is configured entirely by the converted checkpoint's
// `config.json`, which decodes directly into wan-core's `WanConfig` (same field
// names: dim 3072, ffn_dim 14336, num_layers 30, in_dim/out_dim 48, patch_size
// [1,2,2], qk_norm, cross_attn_norm, vae_z_dim 48, sample_shift 5.0, sample_steps
// 40, sample_guide_scale 5.0). `dual_model` is false → single-expert DiT.
//
// This thin layer just exposes the TI2V sampling defaults read off `WanConfig`, so
// callers don't restate the oracle truths.

import Foundation
import WanCore

public enum TI2V5BDefaults {
    /// Native generation geometry (oracle: 704×1280 area-capped; we default to a
    /// 480p-class frame that fits the light tier). Frame count maps to latent T via
    /// the vae temporal stride (4): a 81-frame clip → 21 latent frames.
    public static let width = 1280
    public static let height = 704
    public static let numFrames = 81

    /// Sampler defaults live in `WanConfig` (sampleShift 5.0, sampleSteps 40,
    /// sampleGuideScale 5.0). Helpers so the pipeline reads one source of truth.
    public static func steps(_ c: WanConfig) -> Int { c.sampleSteps }
    public static func shift(_ c: WanConfig) -> Double { c.sampleShift }
    public static func guideScale(_ c: WanConfig) -> Double {
        c.sampleGuideScale.first ?? 5.0
    }
}

public enum TI2V5BError: Error, CustomStringConvertible {
    /// A path that is wired but not yet implemented (denoise loop / i2v — task #12).
    case notImplemented(String)

    public var description: String {
        switch self {
        case .notImplemented(let what): return "TI2V-5B: not yet implemented — \(what)"
        }
    }
}
