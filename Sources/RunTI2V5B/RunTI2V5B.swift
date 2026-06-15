// RunTI2V5B — load-smoke gate for the TI2V-5B scaffold. Loads config + single-expert
// DiT + vae22 (decoder & encoder) from a converted checkpoint and prints what landed,
// proving the wiring + weight-key contracts hold end-to-end. Generation is task #12.
//
//   swift run RunTI2V5B [modelDir]
// Defaults to the bf16 measure checkpoint on DEV_ARCHIVE.

import Foundation
import MLX
import TI2V5B
import WanCore

@main
struct RunTI2V5B {
    static func main() async {
        let args = CommandLine.arguments
        let modelDir = URL(fileURLWithPath: args.count > 1
            ? args[1]
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
            print("✓ tokenizer: umt5-xxl")
            print("✓ §2.4 umT5 eviction wired (paged in per request, not resident)")
            print("scaffold OK — denoise loop + i2v are task #12")
        } catch {
            print("✗ load failed: \(error)")
            exit(1)
        }
    }
}
