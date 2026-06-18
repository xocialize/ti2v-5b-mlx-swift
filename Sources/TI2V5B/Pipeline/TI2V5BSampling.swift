// TI2V-5B denoise loop — the SINGLE-EXPERT plain-CFG path. Isomorphic to the Bernini
// dual-expert `denoiseT2V` minus the high/low boundary switch (TI2V-5B is
// `dual_model:false`) and with a scalar guide scale (config `sample_guide_scale` 5.0).
// The DiT (`WanModel`), schedulers, and RoPE are all wan-core — already exercised by
// Bernini — so this is wiring, not new math.
//
// Conditioning enters as raw UMT5 features so the text encoder stays a separate,
// separately-evictable (§2.4) stage. Latent is channels-first [C, F, H, W] throughout
// the DiT (matches the oracle); the channels-last transpose happens at the vae22 edge.

import Foundation
import MLX
import WanCore

public enum TI2VScheduler: String, Sendable {
    case unipc
    case euler
    case dpmpp
}

/// TI2V-5B i2v mask-blend conditioning (channels-first latent space).
/// - zImg: the vae22-encoded, NORMALIZED conditioning image latent [C, 1, H, W].
/// - mask: [C, T, H, W] — 0 at t=0 (frozen image frame), 1 elsewhere (denoised).
/// - maskTokens: [1, L] — 0 for first-frame tokens, 1 for the rest (per-token timesteps).
public struct I2VCondition {
    public let zImg: MLXArray
    public let mask: MLXArray
    public let maskTokens: MLXArray
    public init(zImg: MLXArray, mask: MLXArray, maskTokens: MLXArray) {
        self.zImg = zImg
        self.mask = mask
        self.maskTokens = maskTokens
    }
}

/// Build the i2v temporal mask + token mask. The image occupies latent frame 0
/// (frozen); the rest is generated. Token order is (f, h, w) f-major, matching the
/// DiT patchify, so the first `hGrid*wGrid` tokens are the (clean) image frame.
public func buildI2VMask(
    channels c: Int, tLat: Int, hLat: Int, wLat: Int, patchSize: [Int]
) -> (mask: MLXArray, maskTokens: MLXArray) {
    let zeros = MLXArray.zeros([c, 1, hLat, wLat])
    let mask = tLat > 1
        ? concatenated([zeros, MLXArray.ones([c, tLat - 1, hLat, wLat])], axis: 1)
        : zeros
    let hGrid = hLat / patchSize[1]
    let wGrid = wLat / patchSize[2]
    let fGrid = tLat / patchSize[0]
    let frameTokens = hGrid * wGrid
    let seqLen = fGrid * frameTokens
    var mt = [Float](repeating: 1, count: seqLen)
    for k in 0..<min(frameTokens, seqLen) { mt[k] = 0 }  // frame-0 tokens clean
    return (mask, MLXArray(mt, [1, seqLen]))
}

/// Run the single-expert CFG denoising loop and return the final latent
/// [C, T_lat, H_lat, W_lat] (channels-first).
/// - contextCond/contextNull: raw UMT5 features [L, text_dim].
/// - noise: initial latent [C, T_lat, H_lat, W_lat] (caller-seeded).
/// - guideScale ≤ 1 → CFG-free (B=1, no uncond pass).
public func denoiseTI2V(
    dit: WanModel,
    config: WanConfig,
    contextCond: MLXArray,
    contextNull: MLXArray,
    noise: MLXArray,
    steps: Int,
    shift: Double,
    guideScale: Double,
    scheduler: TI2VScheduler = .unipc,
    i2v: I2VCondition? = nil,
    onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
) rethrows -> MLXArray {
    let cfg = guideScale > 1.0

    // Pre-embed conditioning (the DiT's text MLP); CFG → [cond, uncond] (B=2).
    let textIn = cfg ? [contextCond, contextNull] : [contextCond]
    let contextCfg = dit.embedText(textIn)
    eval(contextCfg)
    let crossKV = dit.prepareCrossKV(contextCfg)

    // RoPE + seq_len from the (constant) patch grid.
    let (tLat, hLat, wLat) = (noise.dim(1), noise.dim(2), noise.dim(3))
    let fGrid = tLat / config.patchSize[0]
    let hGrid = hLat / config.patchSize[1]
    let wGrid = wLat / config.patchSize[2]
    let grid = (fGrid, hGrid, wGrid)
    let ropeCosSin = dit.prepareRope(cfg ? [grid, grid] : [grid])
    let seqLen = fGrid * hGrid * wGrid

    // Scheduler (FlowUniPC is the official TI2V default).
    let unipc = scheduler == .unipc
        ? FlowUniPCScheduler(numTrainTimesteps: config.numTrainTimesteps) : nil
    let euler = scheduler == .euler
        ? FlowMatchEulerScheduler(numTrainTimesteps: config.numTrainTimesteps) : nil
    let dpmpp = scheduler == .dpmpp
        ? FlowDPMPP2MScheduler(numTrainTimesteps: config.numTrainTimesteps) : nil
    unipc?.setTimesteps(steps, shift: shift)
    euler?.setTimesteps(steps, shift: shift)
    dpmpp?.setTimesteps(steps, shift: shift)
    let timesteps = unipc?.timesteps ?? euler?.timesteps ?? dpmpp!.timesteps

    var latents = noise
    // i2v: blend the frozen image latent into frame 0 before denoising.
    if let i2v { latents = (1 - i2v.mask) * i2v.zImg + i2v.mask * latents }

    for i in 0..<steps {
        let t = Double(timesteps[i])

        // Timestep: scalar [B] (t2v) or per-token [B, seqLen] (i2v — frame-0 tokens
        // get t=0 i.e. already-clean, the rest get the full step timestep).
        func timestepArray(batch: Int) -> MLXArray {
            guard let i2v else {
                return MLXArray(Array(repeating: Float(t), count: batch))
            }
            var tTokens = i2v.maskTokens * Float(t)  // [1, Lmasked]
            let padLen = seqLen - tTokens.dim(1)
            if padLen > 0 {
                let pad = MLXArray(Array(repeating: Float(t), count: padLen), [1, padLen])
                tTokens = concatenated([tTokens, pad], axis: 1)
            }
            return batch == 1 ? tTokens : concatenated([tTokens, tTokens], axis: 0)
        }

        let noisePred: MLXArray
        if cfg {
            let preds = dit(
                [latents, latents], t: timestepArray(batch: 2), context: .embedded(contextCfg),
                seqLen: seqLen, crossKVCaches: crossKV, ropeCosSin: ropeCosSin)
            noisePred = preds[1] + Float(guideScale) * (preds[0] - preds[1])
        } else {
            let preds = dit(
                [latents], t: timestepArray(batch: 1), context: .embedded(contextCfg),
                seqLen: seqLen, crossKVCaches: crossKV, ropeCosSin: ropeCosSin)
            noisePred = preds[0]
        }

        let predB = noisePred.expandedDimensions(axis: 0)
        let sampleB = latents.expandedDimensions(axis: 0)
        let stepped: MLXArray
        if let unipc {
            stepped = unipc.step(modelOutput: predB, timestep: Float(t), sample: sampleB)
        } else if let euler {
            stepped = euler.step(modelOutput: predB, timestep: Float(t), sample: sampleB)
        } else {
            stepped = dpmpp!.step(modelOutput: predB, timestep: Float(t), sample: sampleB)
        }
        latents = stepped.squeezed(axis: 0)
        // i2v: re-freeze the image frame after each step.
        if let i2v { latents = (1 - i2v.mask) * i2v.zImg + i2v.mask * latents }

        eval(latents)
        MLX.GPU.clearCache()  // per-step buffer-cache discipline (long configs)
        WanDebug.stats("denoise step \(i + 1)/\(steps)", latents)  // WAN_DEBUG_STATS (latent divergence/zeroing)
        try onStep?(i, steps, latents)
    }
    return latents
}
