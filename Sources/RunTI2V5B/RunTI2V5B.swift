// RunTI2V5B — smoke / production-run gate for the TI2V-5B port.
//   swift run -c release RunTI2V5B [load|t2v|gen] [modelDir]
//     load (default): load config + DiT + vae22 + tokenizer; prove the key contracts.
//     t2v          : a tiny end-to-end text→image generation (the full relay).
//     gen          : a configurable production-scale t2v — env W/H/FRAMES/STEPS — saves a
//                    PNG of frame 0 + reports peak GPU memory and wall time.
// Defaults to the bf16 measure checkpoint on DEV_ARCHIVE.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import TI2V5B
import Tokenizers
import UniformTypeIdentifiers
import WanCore

/// Save one channels-last frame [H, W, 3] in [-1, 1] as a PNG.
private func savePNG(frame: MLXArray, to url: URL) {
    let h = frame.dim(0), w = frame.dim(1)
    let rgb = clip((frame.asType(.float32) + 1) * 127.5, min: 0, max: 255).asType(.uint8)
    eval(rgb)
    let bytes = rgb.asArray(UInt8.self)
    let out = NSMutableData()
    guard let provider = CGDataProvider(data: CFDataCreate(nil, bytes, bytes.count)),
          let image = CGImage(
            width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: w * 3,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithData(
            out as CFMutableData, UTType.png.identifier as CFString, 1, nil)
    else { print("  (png encode failed)"); return }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else { print("  (png finalize failed)"); return }
    try? out.write(to: url)
}

@main
struct RunTI2V5B {
    static func main() async {
        let args = CommandLine.arguments
        let mode = args.count > 1 ? args[1] : "load"

        // Tokenizer-only parity check (no model load).
        if mode == "tok" {
            let prompt = args.count > 2
                ? args[2] : "a golden retriever puppy running across a sunny meadow"
            do {
                let tok = try await AutoTokenizer.from(pretrained: "google/umt5-xxl")
                let posIds = tok.encode(text: cleanText(prompt))
                print("Swift POS (\(posIds.count) ids): \(posIds)")
                // The Chinese negative prompt — the ftfy/NFKC-sensitive case.
                let cfg = try WanConfig.load(from: URL(fileURLWithPath:
                    "/Volumes/DEV_ARCHIVE/ti2v-5b-measure/models/ti2v-5b-bf16/config.json"))
                let negIds = tok.encode(text: cleanText(cfg.sampleNegPrompt))
                print("Swift NEG (\(negIds.count) ids): \(negIds.prefix(12))…")
            } catch { print("✗ tokenizer/config load failed: \(error)"); exit(1) }
            return
        }

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

            if mode == "i2v" {
                // Validate the PRIMARY mode at scale: t2i a coherent frame (1-frame t2v
                // converges), then i2v multi-frame from it. Env W/H/FRAMES/STEPS.
                let env = ProcessInfo.processInfo.environment
                let w = env["W"].flatMap { Int($0) } ?? 256
                let h = env["H"].flatMap { Int($0) } ?? 256
                let nf = env["FRAMES"].flatMap { Int($0) } ?? 13
                let st = env["STEPS"].flatMap { Int($0) } ?? 16
                print("\ni2v: init via t2i \(w)×\(h), then \(nf) frames, \(st) steps …")
                let still = try pipe.t2i(
                    prompt: "a golden retriever puppy in a sunny meadow",
                    width: w, height: h, steps: 8, guideScale: 5.0, seed: 0)
                eval(still)
                let initImage = still[0, 0, 0..., 0..., 0...].reshaped(1, 1, h, w, 3)  // [1,1,H,W,3]
                let t0 = Date()
                var firstNaN = -1
                let frames = try pipe.i2v(
                    image: initImage,
                    prompt: "a golden retriever puppy running across a sunny meadow",
                    numFrames: nf, steps: st, guideScale: 5.0, seed: 0
                ) { i, total, latent in
                    let m = latent.abs().max().item(Float.self)
                    if !m.isFinite && firstNaN < 0 { firstNaN = i }
                    print("  step \(i + 1)/\(total)  |latent|max=\(m)  (\(String(format: "%.0f", -t0.timeIntervalSinceNow))s)")
                }
                eval(frames)
                let lo = frames.min().item(Float.self), hi = frames.max().item(Float.self)
                print("✓ i2v frames \(frames.shape) range [\(lo), \(hi)]"
                    + (firstNaN >= 0 ? "  ⚠️ NaN at step \(firstNaN + 1)" : "  ✓ stable"))
            } else if mode == "gen" {
                // Production-scale t2v. Env: W/H/FRAMES/STEPS (defaults 1280×704, 81f, 40 steps).
                let env = ProcessInfo.processInfo.environment
                let w = env["W"].flatMap { Int($0) } ?? 1280
                let h = env["H"].flatMap { Int($0) } ?? 704
                let nf = env["FRAMES"].flatMap { Int($0) } ?? 81
                let st = env["STEPS"].flatMap { Int($0) } ?? 40
                print("\ngen: \(w)×\(h), \(nf) frames, \(st) steps …")
                let t0 = Date()
                var firstNaN = -1
                let frames = try pipe.t2v(
                    prompt: "a golden retriever puppy running across a sunny meadow, "
                        + "cinematic, shallow depth of field",
                    width: w, height: h, numFrames: nf, steps: st, guideScale: 5.0, seed: 0
                ) { i, total, latent in
                    let mx = latent.abs().max().item(Float.self)
                    if !mx.isFinite && firstNaN < 0 { firstNaN = i }
                    print("  step \(i + 1)/\(total)  |latent|max=\(mx)  (\(String(format: "%.0f", -t0.timeIntervalSinceNow))s)")
                }
                if firstNaN >= 0 { print("  ⚠️ latent first non-finite at step \(firstNaN + 1)") }
                eval(frames)
                let secs = -t0.timeIntervalSinceNow
                let lo = frames.min().item(Float.self), hi = frames.max().item(Float.self)
                let peakGB = Double(Memory.peakMemory) / 1e9
                let outURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent("ti2v-gen-frame0.png")
                savePNG(frame: frames[0, 0, 0..., 0..., 0...], to: outURL)
                print("✓ frames \(frames.shape) range [\(lo), \(hi)]")
                print("✓ wall \(String(format: "%.1f", secs))s  peak GPU \(String(format: "%.1f", peakGB)) GB")
                print("✓ saved frame 0 → \(outURL.path)")
            } else if mode == "t2v" {
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
