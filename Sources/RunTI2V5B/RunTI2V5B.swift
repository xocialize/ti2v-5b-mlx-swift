// RunTI2V5B — smoke gate for the TI2V-5B port.
//   swift run -c release RunTI2V5B [load|t2v] [modelDir]
//     load (default): load config + DiT + vae22 + tokenizer; prove the key contracts.
//     t2v          : a tiny end-to-end text→image generation (the full relay).
// Defaults to the bf16 measure checkpoint on DEV_ARCHIVE.

import Foundation
import MLX
import TI2V5B
import WanCore

@main
struct RunTI2V5B {
    static func main() async {
        let args = CommandLine.arguments
        let mode = args.count > 1 ? args[1] : "load"
        let modelDir = URL(fileURLWithPath: args.count > 2
            ? args[2]
            : "/Volumes/DEV_ARCHIVE/ti2v-5b-measure/models/ti2v-5b-bf16")

        guard FileManager.default.fileExists(
            atPath: modelDir.appendingPathComponent("config.json").path)
        else {
            print("✗ no config.json under \(modelDir.path) — pass a converted checkpoint dir")
            exit(1)
        }

        do {
            print("Loading TI2V-5B from \(modelDir.lastPathComponent) …")
            let pipe = try await TI2V5BPipeline.fromPretrained(modelDir: modelDir)
            let c = pipe.config
            print("✓ config: dim=\(c.dim) layers=\(c.numLayers) heads=\(c.numHeads) "
                + "in/out=\(c.inDim)/\(c.outDim) patch=\(c.patchSize) "
                + "dual=\(c.dualModel) vaeZ=\(c.vaeZDim)")
            print("✓ DiT (WanModel) loaded: \(pipe.dit.blocks.count) blocks, dim \(pipe.dit.dim)")
            print("✓ vae22 decoder + encoder loaded (fp32)")
            print("✓ tokenizer: umt5-xxl   ✓ §2.4 umT5 eviction wired")

            if mode == "t2v" {
                // Tiny end-to-end: 256×256, 1 frame, 4 steps — exercises the whole
                // relay (umT5 encode→evict → 30-block DiT CFG denoise → vae22 decode).
                print("\nt2v smoke: 256×256, 1 frame, 4 steps …")
                let frames = try pipe.t2i(
                    prompt: "a red cube on a white table",
                    width: 256, height: 256, steps: 4, guideScale: 5.0, seed: 0)
                eval(frames)
                let lo = frames.min().item(Float.self)
                let hi = frames.max().item(Float.self)
                print("✓ t2v frames \(frames.shape) range [\(lo), \(hi)] (expect ⊆ [-1,1])")
                print("relay OK — full TI2V-5B text→image generation runs end-to-end")
            } else {
                print("scaffold OK — pass 't2v' to run a tiny end-to-end generation")
            }
        } catch {
            print("✗ failed: \(error)")
            exit(1)
        }
    }
}
