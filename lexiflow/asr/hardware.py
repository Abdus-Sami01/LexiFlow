"""Detect the host CPU and emit the CMake flags that make whisper.cpp fast."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class HardwareProfile:
    system: str
    machine: str
    cpu_count: int
    physical_cores: int
    features: List[str] = field(default_factory=list)
    is_apple_silicon: bool = False

    def has(self, feature: str) -> bool:
        return feature.lower() in self.features


def _read_linux_flags() -> List[str]:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("flags") or line.startswith("Features"):
                    return line.split(":", 1)[1].split()
    except OSError:
        pass
    return []


def _read_sysctl(key: str) -> str:
    if not shutil.which("sysctl"):
        return ""
    try:
        return subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        return ""


def _read_macos_flags() -> List[str]:
    raw = " ".join(
        (
            _read_sysctl("machdep.cpu.features"),
            _read_sysctl("machdep.cpu.leaf7_features"),
            _read_sysctl("hw.optional.arm.FEAT_DotProd"),
        )
    )
    return [token.lower() for token in raw.split()]


def _physical_cores() -> int:
    system = platform.system()
    if system == "Darwin":
        value = _read_sysctl("hw.physicalcpu")
        if value.isdigit():
            return int(value)
    if system == "Linux":
        try:
            ids = set()
            core, package = None, None
            with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("core id"):
                        core = line.split(":", 1)[1].strip()
                    elif line.startswith("physical id"):
                        package = line.split(":", 1)[1].strip()
                    elif not line.strip() and core is not None:
                        ids.add((package, core))
                        core, package = None, None
            if ids:
                return len(ids)
        except OSError:
            pass
    return os.cpu_count() or 1


def detect_hardware() -> HardwareProfile:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        features = _read_macos_flags()
    elif system == "Linux":
        features = [token.lower() for token in _read_linux_flags()]
    else:
        features = []
    return HardwareProfile(
        system=system,
        machine=machine,
        cpu_count=os.cpu_count() or 1,
        physical_cores=_physical_cores(),
        features=features,
        is_apple_silicon=system == "Darwin" and machine in {"arm64", "aarch64"},
    )


def compiler_flags(profile: HardwareProfile | None = None) -> Dict[str, str]:
    """CMake definitions tuned to the detected architecture."""
    profile = profile or detect_hardware()
    flags: Dict[str, str] = {"GGML_NATIVE": "ON", "CMAKE_BUILD_TYPE": "Release"}

    if profile.is_apple_silicon:
        flags.update({"GGML_ACCELERATE": "ON", "GGML_METAL": "ON", "GGML_BLAS": "ON"})
        return flags

    flags["GGML_OPENMP"] = "ON"
    if profile.has("avx512f"):
        flags.update({"GGML_AVX512": "ON", "GGML_AVX2": "ON", "GGML_FMA": "ON", "GGML_AVX": "ON"})
    elif profile.has("avx2"):
        flags.update({"GGML_AVX2": "ON", "GGML_FMA": "ON", "GGML_AVX": "ON"})
    elif profile.has("avx"):
        flags["GGML_AVX"] = "ON"
    if profile.has("f16c"):
        flags["GGML_F16C"] = "ON"
    return flags


def build_command(source_dir: str = "whisper.cpp", profile: HardwareProfile | None = None) -> str:
    """The exact shell command to build a whisper.cpp tuned for this laptop."""
    flags = compiler_flags(profile)
    defines = " ".join(f"-D{key}={value}" for key, value in sorted(flags.items()))
    jobs = max(1, (profile or detect_hardware()).physical_cores)
    return (
        f"cmake -B {source_dir}/build -S {source_dir} {defines} && "
        f"cmake --build {source_dir}/build -j {jobs} --config Release"
    )


def describe() -> str:
    profile = detect_hardware()
    lines = [
        f"system           : {profile.system} ({profile.machine})",
        f"logical cores    : {profile.cpu_count}",
        f"physical cores   : {profile.physical_cores}",
        f"apple silicon    : {profile.is_apple_silicon}",
    ]
    for feature in ("avx", "avx2", "avx512f", "f16c", "neon"):
        lines.append(f"{feature:<17}: {profile.has(feature)}")
    lines.append("")
    lines.append("recommended build:")
    lines.append("  " + build_command(profile=profile))
    return "\n".join(lines)
