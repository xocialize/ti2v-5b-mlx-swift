import TI2V5B
import WanCore
import MLXToolKit

extension Mode {
    /// DPM++(2M) at 16 steps — the quicker path (same lever as Bernini's `.fast`:
    /// ~2.5× faster than the 40-step UniPC default at near-identical quality).
    public static let fast: Mode = "fast"
    /// 40-step UniPC — the reference quality path (the package default).
    public static let quality: Mode = "quality"
}

/// Resolve a request `mode` (+ any explicit `steps`) to the core's scheduler + step
/// count. An explicit `steps` always wins; otherwise the mode picks (`.fast` → 16,
/// else config default 40).
func resolveSampling(mode: Mode?, steps: Int?) -> (scheduler: TI2VScheduler, steps: Int?) {
    switch mode {
    case .fast:
        return (.dpmpp, steps ?? 16)
    default:  // nil / .quality / unknown → reference path (config-default steps)
        return (.unipc, steps)
    }
}
