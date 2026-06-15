import Foundation
import MLX
import MLXNN
import XCTest

@testable import TI2V5B
import WanCore

/// t2v denoise-loop parity gate: the Swift `denoiseTI2V` must reproduce the oracle's
/// single-expert CFG loop given the same injected noise + raw text contexts. Run on
/// the CPU stream in fp32 (DiT cast to float32) so the comparison is strict — like the
/// vae22 gate. Validates the WIRING (CFG formula, scheduler usage, forward args, latent
/// geometry); the DiT forward + FlowUniPC are wan-core's, already exercised by Bernini.
///
/// Fixtures from `gen_t2v_latent_golden.py` (3 steps, shift 5, gs 5, 48×1×8×8).
/// Fixture-gated: skips cleanly without DEV_ARCHIVE.
final class TI2VDenoiseT2VParityTests: XCTestCase {
    static let root = URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/ti2v-5b-measure")
    var parityDir: URL { Self.root.appendingPathComponent("vae22_parity") }
    var modelDir: URL { Self.root.appendingPathComponent("models/ti2v-5b-bf16") }

    func testDenoiseT2VMatchesOracle() throws {
        let fm = FileManager.default
        for f in ["t2v_noise.npy", "t2v_latent.npy", "t2v_ctx_cond.npy"] {
            if !fm.fileExists(atPath: parityDir.appendingPathComponent(f).path) {
                throw XCTSkip("t2v parity fixtures absent")
            }
        }
        try Device.withDefaultDevice(Device(.cpu)) {
            let config = try WanConfig.load(
                from: modelDir.appendingPathComponent("config.json"))

            // DiT in fp32 (bit-exact CPU parity, matching the fixture).
            let dit = WanModel(config)
            let weights = try WeightLoader.loadSafetensors(
                url: modelDir.appendingPathComponent("model.safetensors")
            ).mapValues { $0.asType(.float32) }
            WeightLoader.materialize(weights)
            try dit.update(
                parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
            eval(dit.parameters())

            let noise = try loadNumpy(url: parityDir.appendingPathComponent("t2v_noise.npy"))
            let ctxCond = try loadNumpy(url: parityDir.appendingPathComponent("t2v_ctx_cond.npy"))
            let ctxNull = try loadNumpy(url: parityDir.appendingPathComponent("t2v_ctx_null.npy"))
            let golden = try loadNumpy(url: parityDir.appendingPathComponent("t2v_latent.npy"))

            let latent = try denoiseTI2V(
                dit: dit, config: config, contextCond: ctxCond, contextNull: ctxNull,
                noise: noise, steps: 3, shift: 5.0, guideScale: 5.0, scheduler: .unipc)
            eval(latent)

            XCTAssertEqual(latent.shape, golden.shape, "latent shape mismatch")
            let maxd = abs(latent - golden).max().item(Float.self)
            let meand = abs(latent - golden).mean().item(Float.self)
            print("[t2v parity] denoise max-abs=\(maxd) mean-abs=\(meand) shape=\(latent.shape)")
            // Bit-exact (0.0) on fp32 CPU in practice — the DiT forward matches the
            // oracle exactly. A wiring bug (wrong CFG sign/args/order) diverges by
            // O(1-10); tolerances leave only fp-jitter margin.
            XCTAssertLessThan(meand, 1e-5, "t2v denoise mean-abs \(meand)")
            XCTAssertLessThan(maxd, 1e-3, "t2v denoise max-abs \(maxd)")
        }
    }
}
