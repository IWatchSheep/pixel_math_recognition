"""Reproducible, display-independent export of threshold-growth experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
from threading import Event
from typing import Callable
import zipfile

from PIL import Image, ImageDraw, ImageFont

from threshold_growth import (
    GROWTH_BOUNDARY_COLOR,
    GROWTH_COMPONENT_COLOR,
    GROWTH_NEW_BOUNDARY_COLOR,
    GROWTH_NEW_PIXELS_COLOR,
    GrowthAnalysisCancelled,
    ThresholdGrowthFrame,
    create_growth_overlay,
)
from threshold_growth_instructions import (
    ThresholdGrowthConfig,
    ThresholdGrowthExperimentResult,
    run_threshold_growth_experiment,
)

ProgressCallback = Callable[[int, int, str], None]
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}\Z")
_RUN_DIRECTORY_NAME = re.compile(r"\d{8}_\d{6}_\d{6}_\d{3}\Z")
_MAX_SNAPSHOT_SIDE = 2400
_MAX_SNAPSHOT_SCALE = 12


class RecordingCancelled(Exception):
    """Raised when an export is canceled and its incomplete output is removed."""


def pictures_directory() -> Path:
    """Return the Windows known Pictures folder when available, with a home fallback."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class Guid(ctypes.Structure):
                _fields_ = [
                    ("Data1", wintypes.DWORD),
                    ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD),
                    ("Data4", ctypes.c_ubyte * 8),
                ]

            folder_id = Guid(
                0x33E28130,
                0x4E1E,
                0x4676,
                (ctypes.c_ubyte * 8)(0x83, 0x5A, 0x98, 0x39, 0x5C, 0x3B, 0xC3, 0xBB),
            )
            result = ctypes.c_wchar_p()
            status = ctypes.windll.shell32.SHGetKnownFolderPath(
                ctypes.byref(folder_id), 0, None, ctypes.byref(result)
            )
            if status == 0 and result.value:
                path = Path(result.value)
                ctypes.windll.ole32.CoTaskMemFree(result)
                return path
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    return Path.home() / "Pictures"


def _unique_run_directory(output_root: Path, experiment_name: str) -> Path:
    if not _SAFE_NAME.fullmatch(experiment_name):
        raise ValueError("Unsafe experiment name.")
    parent = output_root / experiment_name
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for suffix in range(1000):
        run_directory = parent / f"{timestamp}_{suffix:03d}"
        try:
            run_directory.mkdir()
        except FileExistsError:
            continue
        return run_directory
    raise FileExistsError("Could not create a unique experiment output directory.")


def _clear_previous_run_directories(output_root: Path, experiment_name: str) -> None:
    if not _SAFE_NAME.fullmatch(experiment_name):
        raise ValueError("Unsafe experiment name.")
    parent = output_root / experiment_name
    if parent.is_symlink():
        raise OSError(f"Experiment output directory must not be a symbolic link: {parent}")
    if not parent.exists():
        return
    for previous_run in parent.iterdir():
        if (
            _RUN_DIRECTORY_NAME.fullmatch(previous_run.name)
            and previous_run.is_dir()
            and not previous_run.is_symlink()
        ):
            shutil.rmtree(previous_run)


def _boundary_pixels(frame: ThresholdGrowthFrame, image_size: tuple[int, int]) -> frozenset[tuple[int, int]]:
    width, height = image_size
    component = set(frame.component_pixels)
    return frozenset(
        (x, y)
        for x, y in component
        if any(
            not (0 <= neighbor_x < width and 0 <= neighbor_y < height)
            or (neighbor_x, neighbor_y) not in component
            for neighbor_y in range(y - 1, y + 2)
            for neighbor_x in range(x - 1, x + 2)
        )
    )


def render_growth_snapshot(
    rgb_image: Image.Image,
    frame: ThresholdGrowthFrame,
    roi: tuple[int, int, int, int],
) -> Image.Image:
    """Render a labeled ROI snapshot at a fixed, display-independent scale."""
    left, top, right, bottom = roi
    overlay = create_growth_overlay(rgb_image, frame).crop((left, top, right, bottom))
    roi_width, roi_height = overlay.size
    scale = min(
        _MAX_SNAPSHOT_SCALE,
        _MAX_SNAPSHOT_SIDE / max(roi_width, roi_height, 1),
    )
    if scale != 1:
        overlay = overlay.resize(
            (
                max(1, round(roi_width * scale)),
                max(1, round(roi_height * scale)),
            ),
            Image.Resampling.NEAREST,
        )

    footer_height = 76
    snapshot = Image.new(
        "RGB", (max(420, overlay.width), overlay.height + footer_height), "white"
    )
    snapshot.paste(overlay, (0, 0))
    draw = ImageDraw.Draw(snapshot)
    font = ImageFont.load_default()
    bounds = frame.bounding_rectangle
    bounds_text = "none" if bounds is None else ",".join(str(value) for value in bounds)
    draw.text(
        (6, overlay.height + 5),
        f"Threshold {frame.threshold} | tracked {frame.pixel_count} px | "
        f"outside selected {frame.selected_pixel_count - frame.pixel_count} px | "
        f"absorbed {frame.absorbed_start_sections}",
        fill="black",
        font=font,
    )
    draw.text(
        (6, overlay.height + 20),
        f"Tracked bounds {bounds_text} | background dominated: "
        f"{'yes' if frame.background_dominated else 'no'}",
        fill="black",
        font=font,
    )
    draw.text((6, overlay.height + 38), "Observed: original image", fill="black", font=font)
    legend_y = overlay.height + 53
    for color, label, x in (
        (GROWTH_COMPONENT_COLOR, "Tracked", 6),
        (GROWTH_NEW_PIXELS_COLOR, "New", 74),
        (GROWTH_BOUNDARY_COLOR, "Boundary", 116),
        (GROWTH_NEW_BOUNDARY_COLOR, "New boundary", 181),
    ):
        draw.rectangle((x, legend_y, x + 7, legend_y + 7), fill=color)
        draw.text((x + 10, legend_y - 1), label, fill="black", font=font)
    return snapshot


def _csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    return str(value)


def _write_measurements(path: Path, frames: tuple[ThresholdGrowthFrame, ...]) -> None:
    fieldnames = (
        "threshold",
        "seed_x",
        "seed_y",
        "tracked_pixel_count",
        "newly_admitted_count",
        "newly_admitted_pixels",
        "absorbed_start_sections",
        "bounds_left",
        "bounds_top",
        "bounds_width",
        "bounds_height",
        "centroid_x",
        "centroid_y",
        "selected_pixel_count",
        "roi_pixel_count",
        "occupancy_percentage",
        "background_dominated",
        "events",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for frame in frames:
            bounds = frame.bounding_rectangle or ("", "", "", "")
            centroid = frame.centroid or ("", "")
            writer.writerow({
                "threshold": frame.threshold,
                "seed_x": frame.seed[0],
                "seed_y": frame.seed[1],
                "tracked_pixel_count": frame.pixel_count,
                "newly_admitted_count": frame.newly_admitted_count,
                "newly_admitted_pixels": _csv_value(frame.newly_admitted_pixels),
                "absorbed_start_sections": frame.absorbed_start_sections,
                "bounds_left": bounds[0],
                "bounds_top": bounds[1],
                "bounds_width": bounds[2],
                "bounds_height": bounds[3],
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
                "selected_pixel_count": frame.selected_pixel_count,
                "roi_pixel_count": frame.roi_pixel_count,
                "occupancy_percentage": f"{frame.occupancy_percentage:.6f}",
                "background_dominated": frame.background_dominated,
                "events": " | ".join(frame.events),
            })


def _settings_text(config: ThresholdGrowthConfig) -> str:
    if config.seed is None:
        raise ValueError("Recording settings require explicit seed coordinates.")
    left, top, right, bottom = config.roi
    fields = (
        ("experiment", config.experiment),
        ("experiment_name", config.experiment_name),
        ("roi_left", left),
        ("roi_top", top),
        ("roi_right", right),
        ("roi_bottom", bottom),
        ("brightness_threshold", config.brightness_threshold),
        ("maximum_threshold", config.maximum_threshold),
        ("threshold_increment", config.threshold_increment),
        ("background_limit_percent", config.background_limit_percentage),
        ("seed_x", config.seed[0]),
        ("seed_y", config.seed[1]),
        ("auto_record", config.auto_record),
        ("auto_zoom", config.auto_zoom),
        ("snapshot_mode", config.snapshot_mode),
    )
    return "".join(
        f"{key}={str(value).lower() if isinstance(value, bool) else value}\n"
        for key, value in fields
    )


def _has_connection_event(frame: ThresholdGrowthFrame) -> bool:
    return any(
        "connect" in event.lower() or "absorb" in event.lower()
        for event in frame.events
    )


def _snapshot_reasons(
    frame: ThresholdGrowthFrame,
    previous: ThresholdGrowthFrame | None,
    previous_boundary: frozenset[tuple[int, int]] | None,
    boundary: frozenset[tuple[int, int]],
    previous_outside_tracked_count: int | None,
    first: bool,
    final: bool,
    rendered_changed: bool,
) -> list[str]:
    reasons = []
    outside_tracked_count = frame.selected_pixel_count - frame.pixel_count
    if first:
        reasons.append("starting threshold")
    if previous is not None and frame.pixel_count > previous.pixel_count:
        reasons.append("tracked component gained pixels")
    if (
        previous_outside_tracked_count is not None
        and outside_tracked_count != previous_outside_tracked_count
    ):
        delta = outside_tracked_count - previous_outside_tracked_count
        reasons.append(
            "selected pixels outside tracked component changed "
            f"({previous_outside_tracked_count} to {outside_tracked_count}, {delta:+d})"
        )
    if previous is not None and frame.absorbed_start_sections > previous.absorbed_start_sections:
        reasons.append("starting section absorbed")
    if _has_connection_event(frame):
        reasons.append("connection event")
    if previous_boundary is not None and boundary != previous_boundary:
        reasons.append("tracked component boundary changed")
    if previous is not None and frame.background_dominated != previous.background_dominated:
        reasons.append("ROI background-dominance status changed")
    if rendered_changed and not reasons:
        reasons.append("tracked overlay or measurements changed")
    if final:
        reasons.append("final threshold")
    return list(dict.fromkeys(reasons))


def _write_zip(
    path: Path,
    files: tuple[Path, ...],
    cancel_event: Event,
    progress_callback: ProgressCallback | None,
    progress_start: int,
    progress_total: int,
) -> None:
    temporary_path = path.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for file_index, source in enumerate(files):
                if cancel_event.is_set():
                    raise RecordingCancelled("Recording canceled while packaging results.")
                with source.open("rb") as input_file, archive.open(
                    source.name, "w"
                ) as output_file:
                    while True:
                        if cancel_event.is_set():
                            raise RecordingCancelled(
                                "Recording canceled while packaging results."
                            )
                        chunk = input_file.read(1024 * 1024)
                        if not chunk:
                            break
                        output_file.write(chunk)
                if progress_callback is not None:
                    completed = progress_start + file_index + 1
                    progress_callback(
                        completed,
                        progress_total,
                        f"Packaging {source.name}",
                    )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class AutomaticGrowthRecordingResult:
    experiment: ThresholdGrowthExperimentResult
    output_directory: Path
    snapshot_files: tuple[str, ...]


def record_threshold_growth_experiment(
    config: ThresholdGrowthConfig,
    rgb_image: Image.Image,
    grayscale_image: Image.Image,
    output_root: Path | None = None,
    *,
    cancel_event: Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AutomaticGrowthRecordingResult:
    """Analyze and export an experiment; remove incomplete output on cancellation/error."""
    if not config.auto_record:
        raise ValueError("Automatic recording requires auto_record=true.")
    if config.seed is None:
        raise ValueError("Automatic recording requires explicit seed coordinates.")
    if not _SAFE_NAME.fullmatch(config.experiment_name):
        raise ValueError("Unsafe experiment name.")
    cancellation = cancel_event or Event()
    total_frames = len(
        range(
            config.brightness_threshold,
            config.maximum_threshold + 1,
            config.threshold_increment,
        )
    )
    if (
        config.brightness_threshold
        + (total_frames - 1) * config.threshold_increment
        != config.maximum_threshold
    ):
        total_frames += 1

    def report_analysis(completed: int, total: int, threshold: int) -> None:
        if cancellation.is_set():
            raise GrowthAnalysisCancelled("Threshold growth recording was canceled.")
        if progress_callback is not None:
            progress_callback(
                round(50 * completed / max(total, 1)),
                100,
                f"Analyzing threshold {threshold}",
            )

    try:
        experiment = run_threshold_growth_experiment(
            config,
            rgb_image,
            grayscale_image,
            cancel_event=cancellation,
            progress_callback=report_analysis,
        )
    except GrowthAnalysisCancelled as error:
        raise RecordingCancelled("Recording canceled during threshold analysis.") from error
    if experiment.history is None:
        raise ValueError("No growth history was produced.")
    if cancellation.is_set():
        raise RecordingCancelled("Recording canceled before file generation.")

    root = output_root or (pictures_directory() / "PixelMathExperiments")
    run_directory: Path | None = None
    try:
        _clear_previous_run_directories(root, config.experiment_name)
        run_directory = _unique_run_directory(root, config.experiment_name)
        frames = experiment.history.frames
        manifest_rows: list[dict[str, object]] = []
        snapshot_files: list[str] = []
        previous_frame: ThresholdGrowthFrame | None = None
        previous_boundary: frozenset[tuple[int, int]] | None = None
        previous_rendered: bytes | None = None
        previous_snapshot_name: str | None = None
        previous_outside_tracked_count: int | None = None

        for index, frame in enumerate(frames):
            if cancellation.is_set():
                raise RecordingCancelled("Recording canceled while rendering snapshots.")
            rendered = render_growth_snapshot(rgb_image, frame, config.roi)
            # Compare the image area, not the threshold label, to avoid threshold-only duplicates.
            image_area = rendered.crop((0, 0, rendered.width, rendered.height - 76))
            rendered_state = (
                image_area.tobytes(),
                frame.pixel_count,
                frame.selected_pixel_count - frame.pixel_count,
                frame.absorbed_start_sections,
                frame.bounding_rectangle,
                frame.centroid,
                frame.background_dominated,
            )
            boundary = _boundary_pixels(frame, rgb_image.size)
            changed = (
                previous_rendered is None
                or rendered_state != previous_rendered
                or _has_connection_event(frame)
            )
            reasons = _snapshot_reasons(
                frame,
                previous_frame,
                previous_boundary,
                boundary,
                previous_outside_tracked_count,
                index == 0,
                index == len(frames) - 1,
                changed,
            )
            saved = bool(reasons) and changed
            filename = f"threshold_{frame.threshold:03d}.png"
            if saved:
                rendered.save(run_directory / filename, format="PNG")
                snapshot_files.append(filename)
                previous_snapshot_name = filename
            elif reasons:
                filename = previous_snapshot_name or ""
                reasons.append("identical rendered state; reused prior snapshot")
            manifest_rows.append({
                "threshold": frame.threshold,
                "snapshot_saved": str(saved).lower(),
                "snapshot_file": filename,
                "reason": "; ".join(reasons) if reasons else "no rendered-state change",
                "tracked_pixel_count": frame.pixel_count,
                "selected_outside_tracked_count": (
                    frame.selected_pixel_count - frame.pixel_count
                ),
            })
            previous_frame = frame
            previous_boundary = boundary
            previous_rendered = rendered_state
            previous_outside_tracked_count = (
                frame.selected_pixel_count - frame.pixel_count
            )
            if progress_callback is not None:
                progress_callback(
                    50 + round(30 * (index + 1) / max(len(frames), 1)),
                    100,
                    f"Rendered threshold {frame.threshold}",
                )

        measurements_path = run_directory / "growth_measurements.csv"
        _write_measurements(measurements_path, frames)
        manifest_path = run_directory / "snapshot_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=(
                    "threshold",
                    "snapshot_saved",
                    "snapshot_file",
                    "reason",
                    "tracked_pixel_count",
                    "selected_outside_tracked_count",
                ),
            )
            writer.writeheader()
            writer.writerows(manifest_rows)
        settings_path = run_directory / "experiment_settings.txt"
        settings_path.write_text(
            _settings_text(config), encoding="utf-8"
        )
        if progress_callback is not None:
            progress_callback(82, 100, "Writing measurements and settings")
        if cancellation.is_set():
            raise RecordingCancelled("Recording canceled before ZIP packaging.")

        zip_path = run_directory / "results.zip"
        files = tuple(
            [run_directory / name for name in snapshot_files]
            + [measurements_path, manifest_path, settings_path]
        )
        def zip_progress(done: int, total: int, message: str) -> None:
            if progress_callback is not None:
                progress_callback(
                    82 + round(18 * done / max(total, 1)),
                    100,
                    message,
                )

        _write_zip(
            zip_path,
            files,
            cancellation,
            zip_progress,
            0,
            len(files),
        )
        if progress_callback is not None:
            progress_callback(100, 100, "Recording complete")
        return AutomaticGrowthRecordingResult(
            experiment,
            run_directory,
            tuple(snapshot_files),
        )
    except (RecordingCancelled, GrowthAnalysisCancelled) as error:
        if run_directory is not None:
            try:
                shutil.rmtree(run_directory)
            except OSError as cleanup_error:
                raise RuntimeError(
                    f"Recording canceled, but incomplete output remains at "
                    f"{run_directory}: {cleanup_error}"
                ) from cleanup_error
        raise RecordingCancelled(
            "Recording canceled; incomplete output was removed."
        ) from error
    except Exception as error:
        if run_directory is not None:
            try:
                shutil.rmtree(run_directory)
            except OSError as cleanup_error:
                raise OSError(
                    f"{error}; incomplete output could not be removed from "
                    f"{run_directory}: {cleanup_error}"
                ) from cleanup_error
        raise
