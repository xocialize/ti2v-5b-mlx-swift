import Foundation
import MLX
import MLXNN
import XCTest

@testable import TI2V5B
import WanCore

/// i2v mask-blend denoise parity gate: the Swift i2v path (mask-blend init, per-token
/// timesteps, re-blend) + `buildI2VMask` must reproduce the oracle's TI2V-5B i2v loop
/// given the same injected z_img + noise + raw contexts. fp32 CPU stream → bit-exact.
///
/// Fixtures from `gen_i2v_latent_golden.py` (3 steps, shift 5, gs 5, 48×3×8×8, the
/// image frozen at frame 0). Fixture-gated.
final class TI2VDenoiseI2VParityTests: XCTestCase {
    static let root = URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/ti2v-5b-measure")
    var parityDir: URL { Self.root.appendingPathComponent("vae22_parity") }
    var modelDir: URL { Self.root.appendingPathComponent("models/ti2v-5b-bf16") }

    func testDenoiseI2VMatchesOracle() throws {
        let fm = FileManager.default
        for f in ["i2v_noise.npy", "i2v_zimg.npy", "i2v_latent.npy"] {
            if !fm.fileExists(atPath: parityDir.appendingPathComponent(f).path) {
                throw XCTSkip("i2v parity fixtures absent")
            }
        }
        try Device.withDefaultDevice(Device(.cpu)) {
            let config = try WanConfig.load(
                from: modelDir.appendingPathComponent("config.json"))

            let dit = WanModel(config)
            let weights = try WeightLoader.loadSafetensors(
                url: modelDir.appendingPathComponent("model.safetensors")
            ).mapValues { $0.asType(.float32) }
            WeightLoader.materialize(weights)
            try dit.update(
                parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
            eval(dit.parameters())

            let noise = try loadNumpy(url: parityDir.appendingPathComponent("i2v_noise.npy"))
            let zImg = try loadNumpy(url: parityDir.appendingPathComponent("i2v_zimg.npy"))
            let ctxCond = try loadNumpy(url: parityDir.appendingPathComponent("i2v_ctx_cond.npy"))
            let ctxNull = try loadNumpy(url: parityDir.appendingPathComponent("i2v_ctx_null.npy"))
            let golden = try loadNumpy(url: parityDir.appendingPathComponent("i2v_latent.npy"))

            // Build the mask in Swift (validates buildI2VMask against the oracle's).
            let (mask, maskTokens) = buildI2VMask(
                channels: noise.dim(0), tLat: noise.dim(1), hLat: noise.dim(2),
                wLat: noise.dim(3), patchSize: config.patchSize)

            let latent = try denoiseTI2V(
                dit: dit, config: config, contextCond: ctxCond, contextNull: ctxNull,
                noise: noise, steps: 3, shift: 5.0, guideScale: 5.0, scheduler: .unipc,
                i2v: I2VCondition(zImg: zImg, mask: mask, maskTokens: maskTokens))
            eval(latent)

            XCTAssertEqual(latent.shape, golden.shape, "latent shape mismatch")
            let maxd = abs(latent - golden).max().item(Float.self)
            let meand = abs(latent - golden).mean().item(Float.self)
            print("[i2v parity] denoise max-abs=\(maxd) mean-abs=\(meand) shape=\(latent.shape)")
            XCTAssertLessThan(meand, 1e-5, "i2v denoise mean-abs \(meand)")
            XCTAssertLessThan(maxd, 1e-3, "i2v denoise max-abs \(maxd)")
        }
    }
}
