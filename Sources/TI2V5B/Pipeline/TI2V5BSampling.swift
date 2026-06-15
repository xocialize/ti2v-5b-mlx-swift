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
    for i in 0..<steps {
        let t = Double(timesteps[i])
        let noisePred: MLXArray
        if cfg {
            let tBatch = MLXArray([Float(t), Float(t)])
            let preds = dit(
                [latents, latents], t: tBatch, context: .embedded(contextCfg),
                seqLen: seqLen, crossKVCaches: crossKV, ropeCosSin: ropeCosSin)
            noisePred = preds[1] + Float(guideScale) * (preds[0] - preds[1])
        } else {
            let preds = dit(
                [latents], t: MLXArray([Float(t)]), context: .embedded(contextCfg),
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

        eval(latents)
        MLX.GPU.clearCache()  // per-step buffer-cache discipline (long configs)
        try onStep?(i, steps, latents)
    }
    return latents
}
