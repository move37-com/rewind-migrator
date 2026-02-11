#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import os
import plistlib
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python <3.9 not supported here.
    ZoneInfo = None  # type: ignore[assignment]

DB_PASSWORD = "soiZ58XZJhdka55hLUp18yOtTUTDXz7Diu7Z4JzuwhRwGG13N6Z9RTVU1fGiKkuF"
APP_NAME_CACHE: dict[str, Optional[str]] = {}


@dataclass(frozen=True)
class TzConfig:
    input_tz: timezone
    output_tz: timezone


def log_info(message: str) -> None:
    print(message)


def log_warn(message: str) -> None:
    print(f"WARN  | {message}")


def log_error(message: str) -> None:
    red = "\033[31m"
    reset = "\033[0m"
    print(f"{red}ERROR{reset}: {message}", file=sys.stderr)


def log_section(title: str) -> None:
    line = "-" * len(title)
    print(f"\n{title}\n{line}")


def progress_update(label: str, current: int, total: int) -> None:
    if total <= 0:
        return
    if not sys.stdout.isatty():
        if current == total:
            print(f"{label}: {current}/{total}")
        return
    ratio = current / total
    percent = int(ratio * 100)
    width = 28
    filled = int(width * ratio)
    bar = "=" * filled + "-" * (width - filled)
    print(f"\r{label}: [{bar}] {current}/{total} ({percent:3d}%)", end="", flush=True)
    if current == total:
        print()


def prompt_video_transfer_mode() -> Optional[str]:
    log_section("Video Import Options")
    print("Choose how to import video files:")
    print("  1) Move (recommended) - saves disk space by removing original Rewind.ai files after migration")
    print("  2) Copy - keeps original Rewind.ai files intact")
    if not sys.stdin.isatty():
        log_info("Non-interactive session detected. Using move mode.")
        return "move"
    while True:
        response = input("Select 1 or 2 [1]: ").strip().lower()
        if response in {"", "1", "move", "m"}:
            return "move"
        if response in {"2", "copy", "c"}:
            return "copy"
        if response in {"q", "quit", "no", "n"}:
            log_info("Migration cancelled by user.")
            return None
        print("Please enter 1 (copy) or 2 (move).")


def is_remynd_running() -> bool:
    pgrep = shutil.which("pgrep")
    if not pgrep:
        log_info("pgrep not found; falling back to osascript checks.")
        osascript = shutil.which("osascript")
        if not osascript:
            log_info("osascript not found; cannot auto-detect running apps.")
            return False
        for name in ("ReMynd", "ReMynd AI"):
            result = subprocess.run(
                [osascript, "-e", f'application "{name}" is running'],
                capture_output=True,
                text=True,
                check=False,
            )
            log_info(
                f'osascript check "{name}": exit={result.returncode} '
                f'out="{result.stdout.strip()}"'
            )
            if result.returncode == 0 and "true" in result.stdout.lower():
                return True
        return False
    result = subprocess.run(
        [pgrep, "ReMynd"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def ensure_remynd_not_running() -> bool:
    if not sys.stdin.isatty():
        log_error("This session is not interactive. Please close ReMynd AI and try again.")
        return False
    if not shutil.which("pgrep") and not shutil.which("osascript"):
        log_section("IMPORTANT: Close ReMynd AI")
        response = input(
            "Please quit ReMynd AI before continuing.\n"
            "Type YES to confirm it is closed: "
        ).strip().lower()
        return response == "yes"
    while is_remynd_running():
        log_section("IMPORTANT: Close ReMynd AI")
        print("ReMynd AI appears to be running. Please quit the app before continuing.")
        input("Press Enter once ReMynd AI is closed...")
    return True


def resolve_tz(name: str) -> timezone:
    if name.lower() in {"local", "system"}:
        return datetime.now().astimezone().tzinfo or timezone.utc
    if ZoneInfo is None:
        raise RuntimeError("zoneinfo not available; use --input-tz/--output-tz local or UTC")
    return ZoneInfo(name)


def parse_iso(ts: str, input_tz: timezone) -> datetime:
    # Rewind.ai timestamps are ISO without timezone; treat as input_tz.
    if ts.endswith("Z"):
        ts = ts[:-1]
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=input_tz)
    return dt


def format_remynd_timestamp(dt: datetime) -> str:
    ms = dt.microsecond // 1000
    return dt.strftime("%d.%m.%Y %H:%M:%S.") + f"{ms:03d}0"


def format_remynd_folder(dt: datetime, width: int, height: int) -> str:
    ms = dt.microsecond // 1000
    return dt.strftime("%Y.%m.%d-%H.%M.%S.") + f"{ms:03d}0-{width}x{height}"


def format_db_datetime(dt: datetime) -> str:
    ms = dt.microsecond // 1000
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{ms:03d}"


def resolve_app_name(bundle_id: str) -> Optional[str]:
    if not bundle_id:
        return None
    if bundle_id in APP_NAME_CACHE:
        return APP_NAME_CACHE[bundle_id]
    mdfind = shutil.which("mdfind")
    mdls = shutil.which("mdls")
    if not mdfind or not mdls:
        APP_NAME_CACHE[bundle_id] = None
        return None

    query = f'kMDItemCFBundleIdentifier == "{bundle_id}"'
    result = subprocess.run([mdfind, query], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        APP_NAME_CACHE[bundle_id] = None
        return None

    paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not paths:
        APP_NAME_CACHE[bundle_id] = None
        return None

    app_path = Path(paths[0])
    for key in ("kMDItemDisplayName", "kMDItemCFBundleName"):
        mdls_result = subprocess.run(
            [mdls, "-name", key, "-raw", str(app_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if mdls_result.returncode == 0:
            name = normalize_app_name(mdls_result.stdout.strip())
            if name and name != "(null)":
                APP_NAME_CACHE[bundle_id] = name
                return name

    plist_path = app_path / "Contents" / "Info.plist"
    if plist_path.is_file():
        try:
            with plist_path.open("rb") as fh:
                plist = plistlib.load(fh)
            for key in ("CFBundleDisplayName", "CFBundleName"):
                name = plist.get(key)
                if isinstance(name, str) and name.strip():
                    APP_NAME_CACHE[bundle_id] = name.strip()
                    normalized = normalize_app_name(name.strip())
                    APP_NAME_CACHE[bundle_id] = normalized
                    return normalized
        except Exception:
            pass

    APP_NAME_CACHE[bundle_id] = None
    return None


def normalize_app_name(name: str) -> str:
    if name.lower().endswith(".app"):
        return name[:-4].rstrip()
    return name

def encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint encoding expects non-negative integers")
    out = bytearray()
    while True:
        to_write = value & 0x7F
        value >>= 7
        if value:
            out.append(to_write | 0x80)
        else:
            out.append(to_write)
            break
    return bytes(out)


def datetime_to_ts(dt: datetime) -> int:
    return int(dt.timestamp() * 600)


def encode_value_at_bool_store(events: list[tuple[int, bool]]) -> bytes:
    out = bytearray()
    for time_val, bool_val in events:
        msg = bytearray()
        msg.extend(encode_varint(0x08))
        msg.extend(encode_varint(time_val))
        msg.extend(encode_varint(0x10))
        msg.extend(encode_varint(1 if bool_val else 0))
        out.extend(encode_varint(0x0A))
        out.extend(encode_varint(len(msg)))
        out.extend(msg)
    return bytes(out)


def build_focus_history(intervals: list[tuple[datetime, datetime]]) -> Optional[bytes]:
    if not intervals:
        return None
    intervals = sorted(intervals, key=lambda item: item[0])
    merged: list[list[datetime]] = []
    for start_dt, end_dt in intervals:
        if not merged or start_dt > merged[-1][1]:
            merged.append([start_dt, end_dt])
        else:
            if end_dt > merged[-1][1]:
                merged[-1][1] = end_dt

    events: list[tuple[int, bool]] = []
    for start_dt, end_dt in merged:
        events.append((datetime_to_ts(start_dt), True))
        events.append((datetime_to_ts(end_dt), False))

    filtered: list[tuple[int, bool]] = []
    for time_val, bool_val in events:
        if not filtered:
            filtered.append((time_val, bool_val))
            continue
        last_time, last_value = filtered[-1]
        if last_value == bool_val:
            continue
        if last_time == time_val:
            continue
        if last_time > time_val:
            continue
        filtered.append((time_val, bool_val))

    if not filtered:
        return None
    return encode_value_at_bool_store(filtered)


def compute_video_seconds(
    conn: sqlite3.Connection, video_ids: list[int], tz_cfg: TzConfig
) -> float:
    if not video_ids:
        return 0.0
    total_seconds = 0.0
    chunk_size = 900
    for idx in range(0, len(video_ids), chunk_size):
        chunk = video_ids[idx : idx + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT videoId, MIN(createdAt) AS startAt, MAX(createdAt) AS endAt
            FROM frame
            WHERE videoId IN ({placeholders})
            GROUP BY videoId
            """,
            chunk,
        ).fetchall()
        for row in rows:
            start_raw = row["startAt"]
            end_raw = row["endAt"]
            if not start_raw or not end_raw:
                continue
            start_dt = parse_iso(str(start_raw), tz_cfg.input_tz)
            end_dt = parse_iso(str(end_raw), tz_cfg.input_tz)
            delta = (end_dt - start_dt).total_seconds()
            if delta > 0:
                total_seconds += delta
    return total_seconds


def compute_window_entry_count(conn: sqlite3.Connection, tz_cfg: TzConfig) -> int:
    local_tz = resolve_tz("local")
    window_keys: set[tuple[str | None, str, datetime.date]] = set()
    rows = conn.execute(
        """
        SELECT startDate, endDate, bundleID, windowName
        FROM segment
        ORDER BY id
        """
    )
    for row in rows:
        start_raw = row["startDate"]
        end_raw = row["endDate"]
        if not start_raw or not end_raw:
            continue
        start_dt = parse_iso(str(start_raw), tz_cfg.input_tz).astimezone(tz_cfg.output_tz)
        end_dt = parse_iso(str(end_raw), tz_cfg.input_tz).astimezone(tz_cfg.output_tz)
        if end_dt <= start_dt:
            continue
        bundle_id = str(row["bundleID"]) if row["bundleID"] else None
        window_name = str(row["windowName"]).strip() if row["windowName"] else ""

        start_local = start_dt.astimezone(local_tz)
        end_local = end_dt.astimezone(local_tz)
        current_day = start_local.date()
        last_day = end_local.date()
        while current_day <= last_day:
            window_keys.add((bundle_id, window_name, current_day))
            current_day = current_day + timedelta(days=1)

    return len(window_keys)


def table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == column for row in rows)


def compute_video_bytes(
    videos: list[sqlite3.Row], rewind_chunks: Path, has_file_size: bool
) -> tuple[int, int]:
    total_bytes = 0
    missing = 0
    for video in videos:
        file_size = int(video["fileSize"]) if has_file_size and video["fileSize"] else 0
        if file_size > 0:
            total_bytes += file_size
            continue
        src_video = rewind_chunks / str(video["path"])
        if src_video.is_file():
            total_bytes += src_video.stat().st_size
        else:
            missing += 1
    return total_bytes, missing


def compute_audio_bytes(conn: sqlite3.Connection, rewind_root: Path) -> tuple[int, int]:
    total_bytes = 0
    missing = 0
    seen: set[Path] = set()
    rows = conn.execute("SELECT path, startTime FROM audio").fetchall()
    for row in rows:
        start_time = str(row["startTime"])
        audio_path = str(row["path"]) if row["path"] else ""
        source_audio = resolve_audio_path(rewind_root, audio_path, start_time)
        if source_audio in seen:
            continue
        seen.add(source_audio)
        if source_audio.is_file():
            total_bytes += source_audio.stat().st_size
        else:
            missing += 1
    return total_bytes, missing


def format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{value} B"


def get_disk_free_bytes(path: Path) -> int:
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(str(target)).free


def show_welcome() -> None:
    print(r"  _____        __  __           _       _      ___ ")
    print(r" |  __ \      |  \/  |         | |     | |    |__ \\")
    print(r" | |__) | ___ | \  / |_   _  __| | __ _| |       ) |")
    print(r" |  _  / / _ \| |\/| | | | |/ _` |/ _` | |      / / ")
    print(r" | | \ \|  __/| |  | | |_| | (_| | (_| | |     / /_ ")
    print(r" |_|  \_\\___||_|  |_|\__,_|\__,_|\__,_|_|    |____|")
    print("")
    title = "Rewind.ai -> ReMynd AI Migration Tool"
    print(title)
    print("-" * len(title))
    print("")
    print("This tool will analyze your Rewind.ai data and import it into ReMynd AI.")


def prompt_to_analyze() -> bool:
    if not sys.stdin.isatty():
        log_error("This session is not interactive. Aborting before migration.")
        return False
    response = input(
        "Continue with analysis? [y/N]: "
    ).strip().lower()
    if response not in {"y", "yes"}:
        log_info("Migration cancelled by user.")
        return False
    return True


def prompt_to_migrate(
    video_seconds: float,
    call_count: int,
    video_count: int,
    window_count: int,
    web_visit_count: int,
) -> bool:
    hours = video_seconds / 3600.0 if video_seconds else 0.0
    log_section("Import Summary")
    print(f"Videos to import: {video_count} (~{hours:.2f} hours)")
    print(f"Call recordings:  {call_count}")
    print(f"Windows to import: {window_count}")
    print(f"Web visits:        {web_visit_count}")
    print("")
    response = input("Continue with migration? [y/N]: ").strip().lower()
    if response in {"y", "yes"}:
        return True
    log_info("Migration cancelled by user.")
    return False

def iter_frames(conn: sqlite3.Connection, video_id: int) -> Iterable[tuple[int, str]]:
    cur = conn.execute(
        "SELECT videoFrameIndex, createdAt FROM frame WHERE videoId=? ORDER BY videoFrameIndex",
        (video_id,),
    )
    for row in cur:
        yield int(row[0]), str(row[1])


def copy_or_link(src: Path, dest: Path, use_link: bool) -> None:
    if use_link:
        try:
            os.link(src, dest)
            return
        except Exception:
            pass
    shutil.copy2(src, dest)


def write_frame_files(
    frames: Iterable[tuple[int, str]],
    log_path: Path,
    pts_path: Path,
    tz_cfg: TzConfig,
) -> int:
    count = 0
    with log_path.open("w", encoding="utf-8") as log_fh, pts_path.open("wb") as pts_fh:
        for frame_index, created_at in frames:
            dt_in = parse_iso(created_at, tz_cfg.input_tz)
            dt_out = dt_in.astimezone(tz_cfg.output_tz)
            log_fh.write(f"{format_remynd_timestamp(dt_out)} #{frame_index:6d}\n")
            pts_val = round(dt_out.timestamp() * 600)
            pts_fh.write(struct.pack("<Q", pts_val))
            count += 1
    return count


def map_speech_source(value: str) -> str:
    if value == "me":
        return "mic"
    return "app"


def is_terminal_word(word: str) -> bool:
    return word.endswith((".", "?", "!"))


def append_word(text: str, word: str) -> str:
    if not text:
        return word
    if word[0] in {".", ",", "?", "!", ":", ";"}:
        return text + word
    return text + " " + word


def build_transcription_segments(
    words: list[sqlite3.Row],
    call_start: datetime,
) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    current: Optional[dict[str, object]] = None

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = str(current.get("text") or "").strip()
        if text:
            segments.append(current)
        current = None

    for row in words:
        word_raw = str(row["word"] or "").strip()
        if not word_raw or word_raw in {">>", "--"}:
            continue
        source = map_speech_source(str(row["speechSource"]))
        start_ms = int(row["timeOffset"])
        end_ms = start_ms + int(row["duration"])

        if current is None or current["source"] != source:
            flush()
            current = {
                "source": source,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": word_raw,
            }
        else:
            current["text"] = append_word(str(current["text"]), word_raw)
            current["end_ms"] = end_ms

        if is_terminal_word(word_raw):
            flush()

    flush()

    results: list[dict[str, object]] = []
    for seg in segments:
        start_dt = call_start + timedelta(milliseconds=int(seg["start_ms"]))
        end_dt = call_start + timedelta(milliseconds=int(seg["end_ms"]))
        results.append(
            {
                "start": format_db_datetime(start_dt),
                "end": format_db_datetime(end_dt),
                "source": seg["source"],
                "text": seg["text"],
            }
        )
    return results


def infer_app_name(bundle_id: Optional[str], event_title: Optional[str]) -> str:
    if event_title:
        lowered = event_title.lower()
        if "zoom" in lowered:
            return "Zoom"
        if "teams" in lowered:
            return "Microsoft Teams"
        if "meet" in lowered:
            return "Google Meet"
        return event_title
    if bundle_id:
        lowered = bundle_id.lower()
        if "zoom" in lowered:
            return "Zoom"
        if "teams" in lowered:
            return "Microsoft Teams"
        if "facetime" in lowered:
            return "FaceTime"
        if "skype" in lowered:
            return "Skype"
        if "slack" in lowered:
            return "Slack"
    return "Rewind.ai"


def open_extras_db(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"extras.db not found: {path}")
    return sqlite3.connect(str(path))


def resolve_audio_path(rewind_root: Path, audio_path: str, start_time: str) -> Path:
    if audio_path:
        candidate = Path(audio_path)
        if candidate.is_file():
            return candidate
        snippet_folder = Path(audio_path).parent.name
    else:
        snippet_folder = ""
    if not snippet_folder:
        snippet_folder = start_time.split(".")[0]
    return rewind_root / "snippets" / snippet_folder / "snippet.m4a"


def migrate_calls(
    rewind_root: Path,
    remynd_root: Path,
    conn: sqlite3.Connection,
    tz_cfg: TzConfig,
    overwrite: bool,
    link: bool,
    transfer_mode: str,
) -> None:
    extras_root = remynd_root / "Recordings" / "Extras"
    call_monitor_root = extras_root / "CallMonitor"
    extras_db_path = extras_root / "extras.db"

    try:
        extras_conn: Optional[sqlite3.Connection] = open_extras_db(extras_db_path)
    except RuntimeError as exc:
        log_error(str(exc))
        return
    extras_conn.row_factory = sqlite3.Row

    try:
        calls = conn.execute(
            """
            SELECT
                a.id,
                a.segmentId,
                a.path,
                a.startTime,
                a.duration,
                s.bundleID
            FROM audio a
            LEFT JOIN segment s ON s.id = a.segmentId
            ORDER BY a.startTime, a.id
            """
        ).fetchall()

        seen_call_ids: set[int] = set()
        inserted = skipped = missing = 0

        log_section("Call Import")
        log_info(f"Found {len(calls)} call recordings to import.")
        total_calls = len(calls)
        for idx, audio in enumerate(calls, start=1):
            progress_update("Calls", idx, total_calls)
            segment_id = int(audio["segmentId"])
            start_time = str(audio["startTime"])
            duration = float(audio["duration"])
            audio_path = str(audio["path"])
            bundle_id = audio["bundleID"]

            call_start = parse_iso(start_time, tz_cfg.input_tz).astimezone(tz_cfg.output_tz)
            call_end = call_start + timedelta(seconds=duration)
            call_id = int(call_start.timestamp())

            call_dir = call_monitor_root / f"call-{call_id}"
            if call_dir.exists():
                if overwrite:
                    shutil.rmtree(call_dir)
                else:
                    skipped += 1
                    continue

            if call_id in seen_call_ids:
                skipped += 1
                continue

            if extras_conn is not None:
                existing = extras_conn.execute("SELECT 1 FROM Call WHERE id=?", (call_id,)).fetchone()
                if existing:
                    if overwrite:
                        extras_conn.execute("DELETE FROM TranscriptionSegment WHERE callID=?", (call_id,))
                        extras_conn.execute("DELETE FROM Call WHERE id=?", (call_id,))
                    else:
                        skipped += 1
                        continue

            source_audio = resolve_audio_path(rewind_root, audio_path, start_time)
            if not source_audio.is_file():
                log_warn(f"Missing audio file: {source_audio}")
                missing += 1
                continue
            seen_call_ids.add(call_id)

            event_row = conn.execute(
                "SELECT title FROM event WHERE segmentID=? LIMIT 1", (segment_id,)
            ).fetchone()
            event_title = str(event_row["title"]) if event_row and event_row["title"] else None
            app_name = infer_app_name(str(bundle_id) if bundle_id else None, event_title)

            call_dir.mkdir(parents=True, exist_ok=True)
            dest_audio = call_dir / "audio.m4a"
            if transfer_mode == "move":
                shutil.move(source_audio, dest_audio)
            else:
                copy_or_link(source_audio, dest_audio, link)

            extras_conn.execute(
                """
                INSERT INTO Call (id, appName, startDate, endDate, title, meetingId, participants, shareUrl, sharedAt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call_id,
                    app_name,
                    format_db_datetime(call_start),
                    format_db_datetime(call_end),
                    event_title,
                    None,
                    None,
                    None,
                    None,
                ),
            )

            words = conn.execute(
                """
                SELECT speechSource, word, timeOffset, duration
                FROM transcript_word
                WHERE segmentId=?
                ORDER BY id
                """,
                (segment_id,),
            ).fetchall()

            segments = build_transcription_segments(words, call_start)
            for seg in segments:
                extras_conn.execute(
                    """
                    INSERT INTO TranscriptionSegment (callID, startTimestamp, endTimestamp, source, text, speaker)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        call_id,
                        seg["start"],
                        seg["end"],
                        seg["source"],
                        seg["text"],
                        None,
                    ),
                )

            inserted += 1

        log_info(
            f"Call import complete. Imported={inserted}, skipped={skipped}, missing audio={missing}."
        )
    finally:
        if extras_conn is not None:
            extras_conn.commit()
            extras_conn.close()


def migrate_focused_windows(
    remynd_root: Path,
    conn: sqlite3.Connection,
    tz_cfg: TzConfig,
    overwrite: bool,
) -> None:
    app_db_path = remynd_root / "Recordings" / "app.db"
    if not app_db_path.is_file():
        log_error(f"Could not find ReMynd AI database: {app_db_path}")
        return

    remynd_conn = sqlite3.connect(str(app_db_path))
    remynd_conn.row_factory = sqlite3.Row
    try:
        log_section("Window Data Import")
        log_info("Importing window data...")
        has_table = remynd_conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='FocusedWindow'"
        ).fetchone()
        if not has_table:
            log_error("Required window data table not found in app.db.")
            return
        has_window_table = remynd_conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ApplicationWindow'"
        ).fetchone()
        if not has_window_table:
            log_error("Required window data table not found in app.db.")
            return

        total = remynd_conn.execute("SELECT COUNT(*) FROM FocusedWindow").fetchone()[0]
        if total and overwrite:
            log_warn("Overwrite enabled: existing window data may be replaced.")

        local_tz = resolve_tz("local")
        app_rows = conn.execute(
            """
            SELECT bundleID, MIN(startDate) AS firstStart, MAX(endDate) AS lastEnd
            FROM segment
            WHERE bundleID IS NOT NULL AND bundleID != ''
            GROUP BY bundleID
            """
        ).fetchall()

        segment_rows = conn.execute(
            """
            SELECT startDate, endDate, bundleID, windowName, browserUrl
            FROM segment
            ORDER BY id
            """
        )
        segment_total = 0
        window_entries: dict[tuple[str | None, str | None, datetime.date], dict[str, object]] = {}
        for row in segment_rows:
            segment_total += 1
            start_raw = row["startDate"]
            end_raw = row["endDate"]
            if not start_raw or not end_raw:
                continue
            start_dt = parse_iso(str(start_raw), tz_cfg.input_tz).astimezone(tz_cfg.output_tz)
            end_dt = parse_iso(str(end_raw), tz_cfg.input_tz).astimezone(tz_cfg.output_tz)
            if end_dt <= start_dt:
                continue

            bundle_id = str(row["bundleID"]) if row["bundleID"] else None
            window_name = str(row["windowName"]).strip() if row["windowName"] else ""
            window_doc = str(row["browserUrl"]) if row["browserUrl"] else None

            start_local = start_dt.astimezone(local_tz)
            end_local = end_dt.astimezone(local_tz)
            current_day = start_local.date()
            last_day = end_local.date()
            while current_day <= last_day:
                day_start_local = datetime(
                    current_day.year,
                    current_day.month,
                    current_day.day,
                    tzinfo=local_tz,
                )
                day_end_local = day_start_local + timedelta(days=1)
                day_start = day_start_local.astimezone(tz_cfg.output_tz)
                day_end = day_end_local.astimezone(tz_cfg.output_tz)
                seg_start = max(start_dt, day_start)
                seg_end = min(end_dt, day_end)
                if seg_end <= seg_start:
                    current_day = (current_day + timedelta(days=1))
                    continue
                day_key = current_day
                key = (bundle_id, window_name, day_key)
                entry = window_entries.get(key)
                if entry is None:
                    window_entries[key] = {
                        "bundle_id": bundle_id,
                        "window_name": window_name,
                        "document": window_doc,
                        "first": seg_start,
                        "last": seg_end,
                        "intervals": [(seg_start, seg_end)],
                    }
                else:
                    entry["first"] = min(entry["first"], seg_start)
                    entry["last"] = max(entry["last"], seg_end)
                    if window_doc:
                        entry["document"] = window_doc
                    entry["intervals"].append((seg_start, seg_end))
                current_day = (current_day + timedelta(days=1))

        if segment_total == 0:
            log_info("No window data found.")
            return

        total_steps = segment_total + len(window_entries) + (len(app_rows) * 2)
        progress_current = 0
        log_info(f"Found {len(window_entries)} windows to import.")

        app_inserted = app_existing = 0
        run_inserted = run_skipped = run_deleted = 0
        app_id_by_bundle: dict[str, int] = {}
        app_run_stats: dict[str, tuple[datetime, datetime]] = {}

        existing_apps: dict[str, int] = {
            str(row["bundleIdentifier"]): int(row["id"])
            for row in remynd_conn.execute("SELECT id, bundleIdentifier FROM Application")
        }

        for row in app_rows:
            progress_current += 1
            progress_update("Window data", progress_current, total_steps)
            bundle_id = str(row["bundleID"])
            first_raw = str(row["firstStart"])
            last_raw = str(row["lastEnd"])
            first_dt = parse_iso(first_raw, tz_cfg.input_tz).astimezone(tz_cfg.output_tz)
            last_dt = parse_iso(last_raw, tz_cfg.input_tz).astimezone(tz_cfg.output_tz)
            app_run_stats[bundle_id] = (first_dt, last_dt)

            if bundle_id in existing_apps:
                app_id_by_bundle[bundle_id] = existing_apps[bundle_id]
                app_existing += 1
                continue

            localized_name = resolve_app_name(bundle_id) or bundle_id
            remynd_conn.execute(
                """
                INSERT INTO Application (localizedName, bundleIdentifier, createdAt)
                VALUES (?, ?, ?)
                """,
                (localized_name, bundle_id, format_db_datetime(first_dt)),
            )
            app_id = int(remynd_conn.execute("SELECT last_insert_rowid()").fetchone()[0])
            app_id_by_bundle[bundle_id] = app_id
            existing_apps[bundle_id] = app_id
            app_inserted += 1

        existing_runs: set[int] = {
            int(row["applicationId"])
            for row in remynd_conn.execute("SELECT applicationId FROM ApplicationRun")
            if row["applicationId"] is not None
        }

        for bundle_id, (first_dt, last_dt) in app_run_stats.items():
            progress_current += 1
            progress_update("Window data", progress_current, total_steps)
            app_id = app_id_by_bundle.get(bundle_id)
            if app_id is None:
                continue

            if app_id in existing_runs:
                if overwrite:
                    remynd_conn.execute(
                        "DELETE FROM ApplicationRun WHERE applicationId=?",
                        (app_id,),
                    )
                    run_deleted += 1
                if not overwrite:
                    run_skipped += 1
                    continue

            remynd_conn.execute(
                """
                INSERT INTO ApplicationRun (
                    applicationId,
                    processIdentifier,
                    launchedAt,
                    terminatedAt,
                    reveivedDidTerminate,
                    createdAt
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    app_id,
                    None,
                    format_db_datetime(first_dt),
                    format_db_datetime(last_dt),
                    None,
                    format_db_datetime(first_dt),
                ),
            )
            run_inserted += 1

        remynd_conn.commit()

        inserted = skipped = invalid = 0
        cursor = conn.execute(
            """
            SELECT startDate, endDate, bundleID, windowName, browserUrl
            FROM segment
            ORDER BY id
            """
        )
        for row in cursor:
            progress_current += 1
            progress_update("Window data", progress_current, total_steps)
            start_raw = row["startDate"]
            end_raw = row["endDate"]
            if not start_raw or not end_raw:
                log_warn("Missing start/end timestamp; skipping this segment.")
                invalid += 1
                continue

            start_dt = parse_iso(str(start_raw), tz_cfg.input_tz).astimezone(tz_cfg.output_tz)
            end_dt = parse_iso(str(end_raw), tz_cfg.input_tz).astimezone(tz_cfg.output_tz)
            started_at = format_db_datetime(start_dt)
            ended_at = format_db_datetime(end_dt)

            bundle_id = row["bundleID"]
            app_name = str(bundle_id) if bundle_id else None
            window_title = row["windowName"]
            window_document = row["browserUrl"]

            existing = remynd_conn.execute(
                """
                SELECT 1
                FROM FocusedWindow
                WHERE startedAt=?
                  AND endedAt=?
                  AND applicationName IS ?
                  AND windowTitle IS ?
                  AND windowDocument IS ?
                LIMIT 1
                """,
                (started_at, ended_at, app_name, window_title, window_document),
            ).fetchone()
            if existing and not overwrite:
                skipped += 1
                continue
            if existing and overwrite:
                remynd_conn.execute(
                    """
                    DELETE FROM FocusedWindow
                    WHERE startedAt=?
                      AND endedAt=?
                      AND applicationName IS ?
                      AND windowTitle IS ?
                      AND windowDocument IS ?
                    """,
                    (started_at, ended_at, app_name, window_title, window_document),
                )

            remynd_conn.execute(
                """
                INSERT INTO FocusedWindow (
                    metaSpaceName,
                    metaSpaceId,
                    applicationName,
                    applicationId,
                    applicationRunId,
                    windowTitle,
                    windowDocument,
                    applicationWindowId,
                    startedAt,
                    endedAt
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    None,
                    None,
                    app_name,
                    None,
                    None,
                    window_title,
                    window_document,
                    None,
                    started_at,
                    ended_at,
                ),
            )
            inserted += 1

        remynd_conn.commit()

        run_rows = remynd_conn.execute(
            """
            SELECT ar.id, ar.applicationId, ar.launchedAt, app.bundleIdentifier
            FROM ApplicationRun ar
            JOIN Application app ON app.id = ar.applicationId
            ORDER BY app.bundleIdentifier, ar.launchedAt, ar.id
            """
        ).fetchall()
        run_id_by_bundle: dict[str, int] = {}
        for row in run_rows:
            bundle_id = str(row["bundleIdentifier"])
            if bundle_id not in run_id_by_bundle:
                run_id_by_bundle[bundle_id] = int(row["id"])

        video_dims = conn.execute(
            "SELECT MAX(width) AS max_w, MAX(height) AS max_h FROM video"
        ).fetchone()
        max_width = int(video_dims["max_w"]) if video_dims and video_dims["max_w"] else 0
        max_height = int(video_dims["max_h"]) if video_dims and video_dims["max_h"] else 0
        if max_width == 0 or max_height == 0:
            log_warn("Could not determine screen size; bounds will be 0x0.")
        bounds_str = f"[[0,0],[{max_width // 2},{max_height // 2}]]"

        inserted = skipped = deleted = 0
        window_id_by_key: dict[tuple[str | None, str | None, datetime.date], int] = {}
        next_window_number = 100
        for key, entry in window_entries.items():
            progress_current += 1
            progress_update("Window data", progress_current, total_steps)
            bundle_id = entry["bundle_id"]
            app_run_id = run_id_by_bundle.get(bundle_id) if bundle_id else None
            first_dt = entry["first"]
            last_dt = entry["last"]
            focus_blob = build_focus_history(entry["intervals"])

            existing = remynd_conn.execute(
                """
                SELECT id
                FROM ApplicationWindow
                WHERE applicationRunId IS ?
                  AND name=?
                  AND document IS ?
                  AND firstSeenAt=?
                  AND lastSeenAt=?
                LIMIT 1
                """,
                (
                    app_run_id,
                    entry["window_name"],
                    entry["document"],
                    format_db_datetime(first_dt),
                    format_db_datetime(last_dt),
                ),
                ).fetchone()
            if existing and not overwrite:
                skipped += 1
                window_id_by_key[key] = int(existing["id"])
                continue
            if existing and overwrite:
                remynd_conn.execute(
                    "DELETE FROM ApplicationWindow WHERE id=?",
                    (int(existing["id"]),),
                )
                deleted += 1

            remynd_conn.execute(
                """
                INSERT INTO ApplicationWindow (
                    applicationRunId,
                    number,
                    firstSeenAt,
                    createdAt,
                    lastSeenAt,
                    isOnscreen,
                    "order",
                    name,
                    bounds,
                    document,
                    focusHistory,
                    state,
                    focus
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    app_run_id,
                    next_window_number,
                    format_db_datetime(first_dt),
                    format_db_datetime(first_dt),
                    format_db_datetime(last_dt),
                    0,
                    0,
                    entry["window_name"],
                    bounds_str,
                    entry["document"],
                    sqlite3.Binary(focus_blob) if focus_blob else None,
                    2,
                    0,
                ),
            )
            window_id = int(remynd_conn.execute("SELECT last_insert_rowid()").fetchone()[0])
            window_id_by_key[key] = window_id
            inserted += 1
            next_window_number += 1

        remynd_conn.commit()
        log_info("Window data import complete.")

        browser_rows = conn.execute(
            """
            SELECT id, startDate, endDate, bundleID, windowName, browserUrl, browserProfile
            FROM segment
            WHERE browserUrl IS NOT NULL AND browserUrl != ''
            ORDER BY id
            """
        )

        session_stats: dict[str, dict[str, object]] = {}
        url_first_seen: dict[str, datetime] = {}
        for row in browser_rows:
            url = str(row["browserUrl"]).strip()
            if not url:
                continue
            start_dt = parse_iso(str(row["startDate"]), tz_cfg.input_tz).astimezone(tz_cfg.output_tz)
            end_dt = parse_iso(str(row["endDate"]), tz_cfg.input_tz).astimezone(tz_cfg.output_tz)
            url_first_seen[url] = min(url_first_seen.get(url, start_dt), start_dt)

            bundle_id = str(row["bundleID"]) if row["bundleID"] else ""
            if not bundle_id:
                continue
            stats = session_stats.get(bundle_id)
            if stats is None:
                stats = {
                    "first": start_dt,
                    "last": end_dt,
                    "profile": str(row["browserProfile"]) if row["browserProfile"] else None,
                }
                session_stats[bundle_id] = stats
            else:
                stats["first"] = min(stats["first"], start_dt)
                stats["last"] = max(stats["last"], end_dt)
                if not stats.get("profile") and row["browserProfile"]:
                    stats["profile"] = str(row["browserProfile"])

        log_section("Web History Import")
        web_session_by_bundle: dict[str, int] = {
            str(row["bundleIdentifier"]): int(row["id"])
            for row in remynd_conn.execute("SELECT id, bundleIdentifier FROM WebSession")
            if row["bundleIdentifier"]
        }
        web_url_by_url: dict[str, int] = {
            str(row["url"]): int(row["id"])
            for row in remynd_conn.execute("SELECT id, url FROM WebUrl")
            if row["url"]
        }

        session_inserted = session_existing = 0
        for bundle_id, stats in session_stats.items():
            if bundle_id in web_session_by_bundle:
                session_existing += 1
                continue
            first_dt = stats["first"]
            last_dt = stats["last"]
            profile = stats.get("profile")
            guid = str(uuid.uuid4())
            remynd_conn.execute(
                """
                INSERT INTO WebSession (
                    startedAt,
                    createdAt,
                    lastSeenAt,
                    guid,
                    userAgent,
                    portName,
                    extensionId,
                    extensionVersion,
                    bundleIdentifier,
                    bundleURL,
                    profileName
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    format_db_datetime(first_dt),
                    format_db_datetime(first_dt),
                    format_db_datetime(last_dt),
                    guid,
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 "
                    "Safari/537.36",
                    "ws://localhost:46382/",
                    "migrated-from-rewind.ai",
                    "1.0.0",
                    bundle_id,
                    "",
                    profile,
                ),
            )
            session_id = int(remynd_conn.execute("SELECT last_insert_rowid()").fetchone()[0])
            web_session_by_bundle[bundle_id] = session_id
            session_inserted += 1

        url_inserted = url_existing = 0
        for url, first_dt in url_first_seen.items():
            if url in web_url_by_url:
                url_existing += 1
                continue
            remynd_conn.execute(
                "INSERT INTO WebUrl (url, createdAt) VALUES (?, ?)",
                (url, format_db_datetime(first_dt)),
            )
            url_id = int(remynd_conn.execute("SELECT last_insert_rowid()").fetchone()[0])
            web_url_by_url[url] = url_id
            url_inserted += 1

        visit_total = conn.execute(
            """
            SELECT COUNT(*)
            FROM segment
            WHERE browserUrl IS NOT NULL AND browserUrl != ''
            """
        ).fetchone()[0]
        log_info(f"Found {visit_total} web visits to import.")
        visit_rows = conn.execute(
            """
            SELECT id, startDate, endDate, bundleID, windowName, browserUrl
            FROM segment
            WHERE browserUrl IS NOT NULL AND browserUrl != ''
            ORDER BY id
            """
        )
        visit_inserted = visit_skipped = visit_missing = 0
        for idx, row in enumerate(visit_rows, start=1):
            progress_update("Web history", idx, visit_total)
            url = str(row["browserUrl"]).strip()
            if not url:
                visit_missing += 1
                continue
            bundle_id = str(row["bundleID"]) if row["bundleID"] else ""
            if not bundle_id:
                visit_missing += 1
                continue
            session_id = web_session_by_bundle.get(bundle_id)
            if session_id is None:
                visit_missing += 1
                continue
            url_id = web_url_by_url.get(url)
            if url_id is None:
                visit_missing += 1
                continue

            start_dt = parse_iso(str(row["startDate"]), tz_cfg.input_tz).astimezone(tz_cfg.output_tz)
            end_dt = parse_iso(str(row["endDate"]), tz_cfg.input_tz).astimezone(tz_cfg.output_tz)
            window_title = str(row["windowName"]).strip() if row["windowName"] else ""
            day_key = start_dt.astimezone(local_tz).date()
            app_window_id = window_id_by_key.get((bundle_id, window_title, day_key))

            existing = remynd_conn.execute(
                """
                SELECT 1
                FROM WebURLVisit
                WHERE webSessionId=?
                  AND webURLId=?
                  AND createdAt=?
                  AND endedAt=?
                  AND title=?
                  AND applicationWindowId IS ?
                LIMIT 1
                """,
                (
                    session_id,
                    url_id,
                    format_db_datetime(start_dt),
                    format_db_datetime(end_dt),
                    window_title,
                    app_window_id,
                ),
            ).fetchone()
            if existing and not overwrite:
                visit_skipped += 1
                continue
            if existing and overwrite:
                remynd_conn.execute(
                    """
                    DELETE FROM WebURLVisit
                    WHERE webSessionId=?
                      AND webURLId=?
                      AND createdAt=?
                      AND endedAt=?
                      AND title=?
                      AND applicationWindowId IS ?
                    """,
                    (
                        session_id,
                        url_id,
                        format_db_datetime(start_dt),
                        format_db_datetime(end_dt),
                        window_title,
                        app_window_id,
                    ),
                )

            remynd_conn.execute(
                """
                INSERT INTO WebURLVisit (
                    webSessionId,
                    webURLId,
                    tabId,
                    createdAt,
                    navigatedToAt,
                    endedAt,
                    closeReason,
                    title,
                    "index",
                    windowId,
                    applicationWindowId
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    url_id,
                    0,
                    format_db_datetime(start_dt),
                    format_db_datetime(start_dt),
                    format_db_datetime(end_dt),
                    "close",
                    window_title,
                    0,
                    None,
                    app_window_id,
                ),
            )
            visit_inserted += 1

        remynd_conn.commit()
        log_info(
            f"Web history complete. inserted={visit_inserted}, skipped={visit_skipped}, missing={visit_missing}, total={visit_total}."
        )
    finally:
        remynd_conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate Rewind.ai video files, frame metadata, calls, and focused windows "
            "into ReMynd AI databases."
        ),
    )
    parser.add_argument(
        "--rewind-root",
        default="~/Library/Application Support/com.memoryvault.MemoryVault",
        help="Path to Rewind.ai Application Support folder (com.memoryvault.MemoryVault)",
    )
    parser.add_argument(
        "--remynd-root",
        default="~/Library/Application Support/Move37/ReMynd",
        help="Path to ReMynd AI Application Support root (Move37/ReMynd)",
    )
    parser.add_argument("--input-tz", default="UTC", help="Timezone to interpret Rewind.ai timestamps")
    parser.add_argument("--output-tz", default="UTC", help="Timezone to write ReMynd AI timestamps")
    parser.add_argument("--limit", type=int, default=0, help="Max videos to process (0 = all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing folders")
    parser.add_argument("--link", action="store_true", help="Hardlink video files instead of copying")
    return parser.parse_args()


def decrypt_database(source_db: Path, dest_db: Path) -> None:
    sqlcipher = shutil.which("sqlcipher")
    if not sqlcipher:
        raise RuntimeError("sqlcipher not found; install with: brew install sqlcipher")

    script = (
        f"PRAGMA key = '{DB_PASSWORD}';\n"
        "PRAGMA cipher_compatibility = 4;\n"
        f"ATTACH DATABASE '{dest_db}' AS plaintext KEY '';\n"
        "SELECT sqlcipher_export('plaintext');\n"
        "DETACH DATABASE plaintext;\n"
    )

    result = subprocess.run(
        [sqlcipher, str(source_db)],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "sqlcipher export failed:\n"
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    if not dest_db.exists():
        raise RuntimeError("sqlcipher export failed: no output database created")


def main() -> int:
    args = parse_args()

    show_welcome()

    rewind_root = Path(args.rewind_root).expanduser()
    rewind_db_enc = rewind_root / "db-enc.sqlite3"
    rewind_chunks = rewind_root / "chunks"

    remynd_root = Path(args.remynd_root).expanduser()
    output_root = remynd_root / "Recordings"
    extras_root = output_root / "Extras"
    call_monitor_root = extras_root / "CallMonitor"
    extras_db_path = extras_root / "extras.db"
    app_db_path = output_root / "app.db"

    if not rewind_chunks.is_dir():
        log_error(f"Couldn't find Rewind.ai chunks folder: {rewind_chunks}")
        return 2

    tz_cfg = TzConfig(
        input_tz=resolve_tz(args.input_tz),
        output_tz=resolve_tz(args.output_tz),
    )

    if not rewind_db_enc.is_file():
        log_error(f"Couldn't find encrypted Rewind.ai database: {rewind_db_enc}")
        return 2

    if (
        not remynd_root.is_dir()
        or not output_root.is_dir()
        or not extras_root.is_dir()
        or not call_monitor_root.is_dir()
        or not app_db_path.is_file()
        or not extras_db_path.is_file()
    ):
        print() # Empty line
        log_error(
            "An existing ReMynd AI installation was not found!\n"
            "Please launch ReMynd AI first and create an account, "
            "then run this script again to migrate your Rewind.ai data.\n"
        )
        return 2

    if not ensure_remynd_not_running():
        return 0

    if not prompt_to_analyze():
        return 0

    tmpdir_ctx = tempfile.TemporaryDirectory(prefix="rewind_migrate_")
    tmpdir = Path(tmpdir_ctx.name)
    rewind_db = tmpdir / "rewind.sqlite3"
    try:
        log_section("Preparing Database")
        log_info("Decrypting your Rewind.ai database...")
        decrypt_database(rewind_db_enc, rewind_db)
    except Exception as exc:
        log_error(str(exc))
        tmpdir_ctx.cleanup()
        return 2

    conn = sqlite3.connect(str(rewind_db))
    conn.row_factory = sqlite3.Row
    try:
        log_section("Analyzing Data")
        log_info("Calculating how much data will be imported...")
        limit_clause = f"LIMIT {args.limit}" if args.limit and args.limit > 0 else ""
        has_file_size = table_has_column(conn, "video", "fileSize")
        select_fields = "id, path, width, height"
        if has_file_size:
            select_fields += ", fileSize"
        videos = conn.execute(
            f"SELECT {select_fields} FROM video ORDER BY id {limit_clause}"
        ).fetchall()

        total = len(videos)
        video_ids = [int(row["id"]) for row in videos]
        call_count = conn.execute(
            "SELECT COUNT(DISTINCT startTime) FROM audio"
        ).fetchone()[0]
        window_count = compute_window_entry_count(conn, tz_cfg)
        web_visit_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM segment
            WHERE browserUrl IS NOT NULL AND browserUrl != ''
            """
        ).fetchone()[0]
        total_seconds = compute_video_seconds(conn, video_ids, tz_cfg)
        video_bytes, missing_video_files = compute_video_bytes(
            videos, rewind_chunks, has_file_size
        )
        audio_bytes, missing_audio_files = compute_audio_bytes(conn, rewind_root)
        required_bytes = (10 * 1024**3) + video_bytes + audio_bytes
        free_bytes = get_disk_free_bytes(output_root)

        log_section("Disk Space Check")
        print(
            "Estimated media size: "
            f"{format_bytes(video_bytes + audio_bytes)} "
            f"(video {format_bytes(video_bytes)}, audio {format_bytes(audio_bytes)})"
        )
        print(f"Required free space:  {format_bytes(required_bytes)}")
        print(f"Available space:      {format_bytes(free_bytes)}")
        if missing_video_files or missing_audio_files:
            log_info(
                "Note: Some media files were missing; the estimate may be low."
            )
        if free_bytes < required_bytes:
            log_error("Not enough free space to proceed with migration.")
            return 2

        if not prompt_to_migrate(
            total_seconds,
            int(call_count),
            total,
            int(window_count),
            int(web_visit_count),
        ):
            return 0

        transfer_mode = prompt_video_transfer_mode()
        if transfer_mode is None:
            return 0

        processed = skipped = missing = 0

        log_section("Video Import")
        log_info(f"Found {total} video recordings to import.")
        for idx, video in enumerate(videos, start=1):
            video_id = int(video["id"])
            rel_path = str(video["path"])
            width = int(video["width"])
            height = int(video["height"])
            progress_update("Videos", idx, total)

            frames_iter = iter_frames(conn, video_id)
            try:
                first_frame = next(frames_iter)
            except StopIteration:
                skipped += 1
                continue

            first_dt = parse_iso(first_frame[1], tz_cfg.input_tz).astimezone(tz_cfg.output_tz)
            folder_name = format_remynd_folder(first_dt, width, height)
            dest_dir = output_root / folder_name

            if dest_dir.exists():
                if args.overwrite:
                    shutil.rmtree(dest_dir)
                else:
                    log_info(f"Already exists, skipping: {dest_dir.name}")
                    skipped += 1
                    continue

            src_video = rewind_chunks / rel_path
            if not src_video.is_file():
                log_warn(f"Missing source video file: {src_video}")
                missing += 1
                continue

            dest_movie = dest_dir / f"movie-high-{width}x{height}.mov"
            frames_log = dest_dir / "frames.log"
            frames_pts = dest_dir / "frames.pts"
            dest_dir.mkdir(parents=True, exist_ok=True)
            if transfer_mode == "move":
                shutil.move(src_video, dest_movie)
            else:
                copy_or_link(src_video, dest_movie, args.link)

            write_frame_files(
                frames=itertools.chain([first_frame], frames_iter),
                log_path=frames_log,
                pts_path=frames_pts,
                tz_cfg=tz_cfg,
            )

            checked_path = dest_dir / "checked"
            checked_path.write_text("OK", encoding="utf-8")

            processed += 1
        migrate_calls(
            rewind_root=rewind_root,
            remynd_root=remynd_root,
            conn=conn,
            tz_cfg=tz_cfg,
            overwrite=args.overwrite,
            link=args.link,
            transfer_mode=transfer_mode,
        )
        migrate_focused_windows(
            remynd_root=remynd_root,
            conn=conn,
            tz_cfg=tz_cfg,
            overwrite=args.overwrite,
        )
    finally:
        conn.close()
        if tmpdir_ctx is not None:
            tmpdir_ctx.cleanup()

    print("\n+--------------------+")
    print("| Migration Complete |")
    print("+--------------------+")
    print("You can now launch ReMynd AI to see your imported data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
