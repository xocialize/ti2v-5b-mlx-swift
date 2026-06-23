// swift-tools-version: 6.2
// ti2v-5b-mlx-swift — Swift/MLX port of Wan2.2-TI2V-5B (the light/mid-tier
// text+image-to-video model; single-expert DiT, 48-ch vae22, umT5). The 3rd
// wan-core consumer after Bernini-R and Helios. Python oracle / converted weights:
// /Volumes/DEV_ARCHIVE/ti2v-5b-measure (mlx-video pin 87db56a). See WAN-STACK-PLAN
// Phase C. TI2V-5B is `dual_model: false`, so the DiT is a single `WanModel` — no
// high/low expert switch (simpler than Bernini's A14B).
//
// Scaffold (task #11): TI2V5B core (config + pipeline component loading + §2.4 umT5
// eviction) + a RunTI2V5B load-smoke CLI. The denoise loop, i2v conditioning, and the
// MLXTI2V5B `ModelPackage` wrapper land in task #12.

import PackageDescription

let package = Package(
    name: "TI2V5B",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "TI2V5B", targets: ["TI2V5B"]),
        // The MLXEngine wrapper: a conformant `ModelPackage` over the core pipeline.
        .library(name: "MLXTI2V5B", targets: ["MLXTI2V5B"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        // MLXEngine contract (MLXToolKit) for the wrapper target — local-path-style dep
        // like the other model wrappers; the core `TI2V5B` target stays engine-agnostic.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.9.1"),
        // The neutral Wan substrate (DiT + vae22 + umT5 + RoPE + schedulers + loader).
        .package(path: "../wan-core-mlx-swift"),
    ],
    targets: [
        .target(
            name: "TI2V5B",
            dependencies: [
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/TI2V5B"
        ),
        .target(
            name: "MLXTI2V5B",
            dependencies: [
                "TI2V5B",
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ],
            path: "Sources/MLXTI2V5B"
        ),
        .executableTarget(
            name: "RunTI2V5B",
            dependencies: [
                "TI2V5B",
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
            ],
            path: "Sources/RunTI2V5B"
        ),
        .testTarget(
            name: "TI2V5BTests",
            dependencies: [
                "TI2V5B",
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Tests/TI2V5BTests"
        ),
    ]
)
