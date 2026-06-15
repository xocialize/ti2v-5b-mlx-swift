// Canonical-artifact codecs for the TI2V-5B wrapper. UNLIKE Bernini (channels-first
// [1,3,T,H,W]), vae22 frames are CHANNELS-LAST [1,T,H,W,3] in [-1,1], so the pixel
// extraction needs no transpose. Pure AVFoundation/CoreGraphics — no MLX beyond
// reading the frame tensor out / building the input tensor in.

import AVFoundation
import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import MLX
import MLXToolKit
import UniformTypeIdentifiers
import WanCore

enum FrameCodecError: Error {
    case pixelBufferAllocation
    case writerSetup(String)
    case pngEncode
    case badFrames(String)
    case appendFailed(String)
    case writeIncomplete(String)
    case imageDecode
}

// MARK: - Encode (channels-last frame tensor → PNG / MP4)

/// One channels-last frame [H, W, 3] in [-1, 1] → interleaved RGB bytes [H·W·3].
private func rgbBytes(_ frame: MLXArray) -> (bytes: [UInt8], width: Int, height: Int) {
    let h = frame.dim(0)
    let w = frame.dim(1)
    let scaled = (frame.asType(.float32) + 1) * Float(127.5)
    let rgb = clip(scaled, min: 0, max: 255).asType(.uint8)  // [H, W, 3], already interleaved
    eval(rgb)
    return (rgb.asArray(UInt8.self), w, h)
}

/// Encode one channels-last frame [H, W, 3] as PNG.
func encodePNG(frame: MLXArray) throws -> (data: Data, width: Int, height: Int) {
    let (bytes, w, h) = rgbBytes(frame)
    let cfData = CFDataCreate(nil, bytes, bytes.count)!
    guard
        let provider = CGDataProvider(data: cfData),
        let image = CGImage(
            width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 24,
            bytesPerRow: w * 3, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    else { throw FrameCodecError.pngEncode }
    let out = NSMutableData()
    guard
        let dest = CGImageDestinationCreateWithData(
            out, UTType.png.identifier as CFString, 1, nil)
    else { throw FrameCodecError.pngEncode }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else { throw FrameCodecError.pngEncode }
    return (out as Data, w, h)
}

private func pixelBuffer(
    rgb: [UInt8], width: Int, height: Int, pool: CVPixelBufferPool
) throws -> CVPixelBuffer {
    var bufferOut: CVPixelBuffer?
    CVPixelBufferPoolCreatePixelBuffer(nil, pool, &bufferOut)
    guard let buffer = bufferOut else { throw FrameCodecError.pixelBufferAllocation }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    for y in 0..<height {
        for x in 0..<width {
            let src = (y * width + x) * 3
            let dst = y * stride + x * 4
            base[dst + 0] = rgb[src + 2]  // B
            base[dst + 1] = rgb[src + 1]  // G
            base[dst + 2] = rgb[src + 0]  // R
            base[dst + 3] = 255  // A
        }
    }
    return buffer
}

/// Encode channels-last frames [1, T, H, W, 3] in [-1, 1] as an H.264 MP4 at `fps`.
@InferenceActor
func encodeMP4(frames: MLXArray, fps: Double) async throws -> Data {
    guard frames.ndim == 5, frames.dim(1) > 0, frames.dim(2) > 0, frames.dim(3) > 0 else {
        throw FrameCodecError.badFrames("expected [1,T,H,W,3] with T>0, got \(frames.shape)")
    }
    let t = frames.dim(1)
    let h = frames.dim(2)
    let w = frames.dim(3)

    let url = FileManager.default.temporaryDirectory
        .appending(path: "ti2v-\(UUID().uuidString).mp4")
    defer { try? FileManager.default.removeItem(at: url) }

    let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let input = AVAssetWriterInput(
        mediaType: .video,
        outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: w, AVVideoHeightKey: h,
        ])
    input.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: input,
        sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
        ])
    guard writer.canAdd(input) else { throw FrameCodecError.writerSetup("cannot add input") }
    writer.add(input)
    guard writer.startWriting() else {
        throw FrameCodecError.writerSetup(writer.error?.localizedDescription ?? "startWriting")
    }
    writer.startSession(atSourceTime: .zero)

    let frameDuration = CMTime(value: CMTimeValue((600.0 / fps).rounded()), timescale: 600)
    for i in 0..<t {
        let (bytes, fw, fh) = rgbBytes(frames[0, i, 0..., 0..., 0...])  // [H,W,3]
        guard let pool = adaptor.pixelBufferPool else {
            throw FrameCodecError.writerSetup("no pixel buffer pool")
        }
        let buffer = try pixelBuffer(rgb: bytes, width: fw, height: fh, pool: pool)
        while !input.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(5)) }
        guard adaptor.append(buffer, withPresentationTime:
                  CMTimeMultiply(frameDuration, multiplier: Int32(i))) else {
            throw FrameCodecError.appendFailed(
                "frame \(i)/\(t), status=\(writer.status.rawValue), err=\(String(describing: writer.error))")
        }
    }
    input.markAsFinished()
    await writer.finishWriting()
    let exists = FileManager.default.fileExists(atPath: url.path)
    guard writer.status == .completed, exists else {
        throw FrameCodecError.writeIncomplete(
            "status=\(writer.status.rawValue) err=\(String(describing: writer.error)) exists=\(exists)")
    }
    return try Data(contentsOf: url)
}

// MARK: - Decode (Image → channels-last init tensor for i2v)

/// `Image` → channels-LAST pixels [1, 1, H, W, 3] in [-1, 1], top-down (matching the
/// oracle's PIL preprocessing). One temporal frame — the i2v conditioning image.
func decodeInitImage(_ image: Image, width: Int, height: Int) throws -> MLXArray {
    guard let src = CGImageSourceCreateWithData(image.data as CFData, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
    else { throw FrameCodecError.imageDecode }

    var rgba = [UInt8](repeating: 0, count: width * height * 4)
    let ctx = CGContext(
        data: &rgba, width: width, height: height, bitsPerComponent: 8,
        bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.interpolationQuality = .high
    ctx.translateBy(x: 0, y: CGFloat(height))  // flip to top-down
    ctx.scaleBy(x: 1, y: -1)
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))

    var hwc = [Float](repeating: 0, count: height * width * 3)  // [H, W, 3] interleaved
    for y in 0..<height {
        for x in 0..<width {
            let p = (y * width + x) * 4
            let i = (y * width + x) * 3
            hwc[i + 0] = Float(rgba[p + 0]) / 255 * 2 - 1  // R
            hwc[i + 1] = Float(rgba[p + 1]) / 255 * 2 - 1  // G
            hwc[i + 2] = Float(rgba[p + 2]) / 255 * 2 - 1  // B
        }
    }
    return MLXArray(hwc, [1, 1, height, width, 3])
}
