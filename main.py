#!/usr/bin/env python3
"""QR Image Forge - create valid QR codes whose modules approximate an image.

Copyright (C) 2026 clearact290
SPDX-License-Identifier: GPL-3.0-only
"""

from __future__ import annotations

import importlib
import os
import queue
import random
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

APP_NAME = "QR Image Forge"
VERSION = "1.0.0"

ECC_NAMES = {
    "L - 7% (most data)": "L",
    "M - 15%": "M",
    "Q - 25%": "Q",
    "H - 30% (strongest)": "H",
}
ECC_LABELS = {value: key for key, value in ECC_NAMES.items()}
DITHER_METHODS = ("Floyd-Steinberg", "Atkinson", "Bayer 4x4", "Threshold")
EFFORTS = {
    "Quick": (24, 4),
    "Balanced": (70, 6),
    "High": (180, 8),
    "Custom": (0, 8),
}


def _lazy_dependencies():
    """Load heavy/base dependencies only after the window has appeared."""
    np = importlib.import_module("numpy")
    Image = importlib.import_module("PIL.Image")
    ImageOps = importlib.import_module("PIL.ImageOps")
    qrcode = importlib.import_module("qrcode")
    constants = importlib.import_module("qrcode.constants")
    base = importlib.import_module("qrcode.base")
    util = importlib.import_module("qrcode.util")
    return np, Image, ImageOps, qrcode, constants, base, util


def detect_compute_backend(np, target):
    """Prefer CuPy or PyTorch GPU scoring; otherwise score on the CPU.

    Imports of GPU frameworks are intentionally delayed until generation starts.
    QR encoding itself remains on the CPU (python-qrcode); the GPU accelerates
    batched image comparisons used to rank candidate QR matrices.
    """
    try:
        cp = importlib.import_module("cupy")
        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count > 0:
            try:
                hip_flag = getattr(cp.cuda.runtime, "is_hip", False)
                is_hip = bool(hip_flag() if callable(hip_flag) else hip_flag)
            except Exception:
                is_hip = False
            label = "CuPy / AMD ROCm" if is_hip else "CuPy / CUDA"
            return _CuPyBackend(cp, target, label)
    except Exception:
        pass

    try:
        torch = importlib.import_module("torch")
        if bool(torch.cuda.is_available()):
            is_hip = bool(getattr(torch.version, "hip", None))
            label = "PyTorch / AMD ROCm" if is_hip else "PyTorch / CUDA"
            return _TorchBackend(torch, target, label)
    except Exception:
        pass

    return _NumpyBackend(np, target, "CPU / NumPy")


class _NumpyBackend:
    def __init__(self, np, target, label):
        self.np = np
        self.target = target.astype(np.uint8, copy=False)
        self.label = label

    def score_batch(self, matrices):
        batch = self.np.asarray(matrices, dtype=self.np.uint8)
        return self.np.count_nonzero(batch == self.target[None, :, :], axis=(1, 2)).astype(self.np.int64)


class _CuPyBackend:
    def __init__(self, cp, target, label):
        self.cp = cp
        self.target = cp.asarray(target, dtype=cp.uint8)
        self.label = label

    def score_batch(self, matrices):
        batch = self.cp.asarray(matrices, dtype=self.cp.uint8)
        scores = self.cp.count_nonzero(batch == self.target[None, :, :], axis=(1, 2))
        return self.cp.asnumpy(scores).astype("int64", copy=False)


class _TorchBackend:
    def __init__(self, torch, target, label):
        self.torch = torch
        self.device = torch.device("cuda")
        self.target = torch.as_tensor(target, dtype=torch.uint8, device=self.device)
        self.label = label

    def score_batch(self, matrices):
        tensor = self.torch.as_tensor(matrices, dtype=self.torch.uint8, device=self.device)
        scores = (tensor == self.target.unsqueeze(0)).sum(dim=(1, 2))
        return scores.detach().cpu().numpy().astype("int64", copy=False)


def _dither_image(path, matrix_size, method, invert, np, Image, ImageOps):
    image = Image.open(path).convert("L")
    fitted = ImageOps.contain(image, (matrix_size, matrix_size), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (matrix_size, matrix_size), 255)
    canvas.paste(fitted, ((matrix_size - fitted.width) // 2, (matrix_size - fitted.height) // 2))
    pixels = np.asarray(canvas, dtype=np.float32).copy()
    if invert:
        pixels = 255.0 - pixels

    if method == "Floyd-Steinberg":
        height, width = pixels.shape
        for y in range(height):
            for x in range(width):
                old = float(pixels[y, x])
                new = 255.0 if old >= 127.5 else 0.0
                error = old - new
                pixels[y, x] = new
                if x + 1 < width:
                    pixels[y, x + 1] += error * (7.0 / 16.0)
                if y + 1 < height:
                    if x > 0:
                        pixels[y + 1, x - 1] += error * (3.0 / 16.0)
                    pixels[y + 1, x] += error * (5.0 / 16.0)
                    if x + 1 < width:
                        pixels[y + 1, x + 1] += error * (1.0 / 16.0)
        return (pixels >= 127.5).astype(np.uint8)

    if method == "Atkinson":
        height, width = pixels.shape
        for y in range(height):
            for x in range(width):
                old = float(pixels[y, x])
                new = 255.0 if old >= 127.5 else 0.0
                error = (old - new) / 8.0
                pixels[y, x] = new
                for dx, dy in ((1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (0, 2)):
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < width and yy < height:
                        pixels[yy, xx] += error
        return (pixels >= 127.5).astype(np.uint8)

    if method == "Bayer 4x4":
        bayer = np.array(
            [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
            dtype=np.float32,
        )
        thresholds = (bayer + 0.5) * (255.0 / 16.0)
        tiled = np.tile(thresholds, (matrix_size // 4 + 1, matrix_size // 4 + 1))[:matrix_size, :matrix_size]
        return (pixels >= tiled).astype(np.uint8)

    return (pixels >= 127.5).astype(np.uint8)


class _PositionRecorder:
    """Record QR data-module traversal in exactly python-qrcode's bit order."""

    def __init__(self, qrcode_module):
        self.qrcode = qrcode_module

    def get_positions(self, version, ecc, constants, util):
        qrcode_module = self.qrcode

        class RecorderQRCode(qrcode_module.QRCode):
            def map_data(self, data, mask_pattern):
                positions = []
                inc = -1
                row = self.modules_count - 1
                for col in range(self.modules_count - 1, 0, -2):
                    if col <= 6:
                        col -= 1
                    while True:
                        for c in (col, col - 1):
                            if self.modules[row][c] is None:
                                positions.append((row, c))
                        row += inc
                        if row < 0 or self.modules_count <= row:
                            row -= inc
                            inc = -inc
                            break
                self._recorded_positions = positions
                super().map_data(data, mask_pattern)

        qr = RecorderQRCode(version=version, error_correction=ecc, box_size=1, border=0)
        qr.add_data(util.QRData(b"", mode=util.MODE_8BIT_BYTE), optimize=0)
        qr.makeImpl(True, 0)
        return qr._recorded_positions


def _payload_length(version, ecc, base, util):
    blocks = base.rs_blocks(version, ecc)
    bit_limit = sum(block.data_count * 8 for block in blocks)
    header_bits = 4 + util.length_in_bits(util.MODE_8BIT_BYTE, version)
    count = (bit_limit - header_bits) // 8
    if count < 1:
        raise ValueError("This QR version / error-correction setting has no byte-mode payload space.")
    return count


def _data_to_interleaved_map(version, ecc, base):
    blocks = base.rs_blocks(version, ecc)
    max_data = max(block.data_count for block in blocks)
    offsets = []
    offset = 0
    for block in blocks:
        offsets.append(offset)
        offset += block.data_count

    mapping = {}
    interleaved_index = 0
    for column in range(max_data):
        for block_index, block in enumerate(blocks):
            if column < block.data_count:
                mapping[offsets[block_index] + column] = interleaved_index
                interleaved_index += 1
    return mapping


def _seed_payload(target, positions, version, ecc, mask_pattern, payload_length, base, util):
    """Invert the QR mask for the direct payload bits to create an image-led seed."""
    data_to_interleaved = _data_to_interleaved_map(version, ecc, base)
    mask_function = util.mask_func(mask_pattern)
    header_bits = 4 + util.length_in_bits(util.MODE_8BIT_BYTE, version)
    payload = bytearray(payload_length)

    for payload_index in range(payload_length):
        for bit_in_payload_byte in range(8):
            stream_bit = header_bits + payload_index * 8 + bit_in_payload_byte
            data_word = stream_bit // 8
            bit_within_data_word = stream_bit % 8
            interleaved_word = data_to_interleaved[data_word]
            module_index = interleaved_word * 8 + bit_within_data_word
            row, col = positions[module_index]
            wanted_module = bool(target[row, col])
            wanted_data_bit = wanted_module ^ bool(mask_function(row, col))
            if wanted_data_bit:
                payload[payload_index] |= 1 << (7 - bit_in_payload_byte)
    return bytes(payload)


def _make_matrix(payload, version, ecc, mask_pattern, qrcode_module, util, np):
    qr = qrcode_module.QRCode(
        version=version,
        error_correction=ecc,
        box_size=1,
        border=0,
        mask_pattern=mask_pattern,
    )
    qr.add_data(util.QRData(payload, mode=util.MODE_8BIT_BYTE), optimize=0)
    qr.make(fit=False)
    return np.asarray(qr.modules, dtype=np.uint8)


def generate_qr_image(
    image_path,
    version,
    ecc_letter,
    dither_method,
    invert,
    effort_name,
    custom_steps,
    seed_value,
    cancelled,
    progress_callback,
):
    np, Image, ImageOps, qrcode_module, constants, base, util = _lazy_dependencies()
    ecc = {
        "L": constants.ERROR_CORRECT_L,
        "M": constants.ERROR_CORRECT_M,
        "Q": constants.ERROR_CORRECT_Q,
        "H": constants.ERROR_CORRECT_H,
    }[ecc_letter]
    matrix_size = 17 + 4 * version
    target = _dither_image(image_path, matrix_size, dither_method, invert, np, Image, ImageOps)
    backend = detect_compute_backend(np, target)
    payload_length = _payload_length(version, ecc, base, util)
    positions = _PositionRecorder(qrcode_module).get_positions(version, ecc, constants, util)
    if len(positions) < (sum(b.total_count for b in base.rs_blocks(version, ecc)) * 8):
        raise RuntimeError("Could not map the QR data modules for this version.")

    rng = random.Random(seed_value) if seed_value is not None else random.Random()
    initial_payloads = []
    initial_matrices = []
    for mask_pattern in range(8):
        if cancelled():
            raise InterruptedError("Generation cancelled.")
        payload = _seed_payload(target, positions, version, ecc, mask_pattern, payload_length, base, util)
        matrix = _make_matrix(payload, version, ecc, mask_pattern, qrcode_module, util, np)
        initial_payloads.append((payload, mask_pattern))
        initial_matrices.append(matrix)

    initial_scores = backend.score_batch(initial_matrices)
    best_index = int(np.argmax(initial_scores))
    best_payload, chosen_mask = initial_payloads[best_index]
    current_payload = best_payload
    current_score = int(initial_scores[best_index])
    best_score = current_score
    best_matrix = initial_matrices[best_index]

    if effort_name == "Custom":
        steps, population = max(0, min(600, int(custom_steps))), 8
    else:
        steps, population = EFFORTS[effort_name]

    total_modules = matrix_size * matrix_size
    for step in range(steps):
        if cancelled():
            raise InterruptedError("Generation cancelled.")

        # Early steps use smaller mutations; later steps broaden the search.
        progress = step / max(1, steps)
        max_flips = 1 + int(progress * 5)
        candidate_payloads = []
        candidate_matrices = []
        for _ in range(population):
            mutated = bytearray(current_payload)
            flips = rng.randint(1, max_flips)
            chosen_bits = rng.sample(range(payload_length * 8), min(flips, payload_length * 8))
            for bit_index in chosen_bits:
                mutated[bit_index // 8] ^= 1 << (bit_index % 8)
            mutated_bytes = bytes(mutated)
            candidate_payloads.append(mutated_bytes)
            candidate_matrices.append(_make_matrix(mutated_bytes, version, ecc, chosen_mask, qrcode_module, util, np))

        scores = backend.score_batch(candidate_matrices)
        candidate_order = np.argsort(scores)[::-1]
        top_index = int(candidate_order[0])
        top_score = int(scores[top_index])
        top_payload = candidate_payloads[top_index]
        top_matrix = candidate_matrices[top_index]

        # Small simulated-annealing allowance helps escape a local plateau.
        if top_score >= current_score:
            current_payload, current_score = top_payload, top_score
        else:
            temperature = max(0.7, 7.0 * (1.0 - progress))
            delta = top_score - current_score
            if rng.random() < min(0.35, float(np.exp(delta / temperature))):
                current_payload, current_score = top_payload, top_score

        if top_score > best_score:
            best_score = top_score
            best_payload = top_payload
            best_matrix = top_matrix

        if step % 2 == 0 or step + 1 == steps:
            progress_callback(step + 1, steps, best_score / total_modules * 100.0, backend.label)

    # Rebuild from the winning payload to guarantee the saved matrix matches it.
    best_matrix = _make_matrix(best_payload, version, ecc, chosen_mask, qrcode_module, util, np)
    score_percent = float(np.count_nonzero(best_matrix == target)) / total_modules * 100.0
    return {
        "matrix": best_matrix,
        "payload": best_payload,
        "version": version,
        "ecc": ecc_letter,
        "mask": chosen_mask,
        "score": score_percent,
        "target": target,
        "backend": backend.label,
        "payload_length": payload_length,
        "steps": steps,
    }


class QRImageForgeApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME} {VERSION}")
        self.root.minsize(780, 570)
        self.root.geometry("940x690")

        self.image_path = tk.StringVar()
        self.version_var = tk.StringVar(value="10")
        self.ecc_var = tk.StringVar(value="L - 7% (most data)")
        self.dither_var = tk.StringVar(value="Floyd-Steinberg")
        self.effort_var = tk.StringVar(value="Balanced")
        self.custom_steps_var = tk.StringVar(value="120")
        self.scale_var = tk.StringVar(value="12")
        self.invert_var = tk.BooleanVar(value=False)
        self.seed_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Choose an image to begin.")
        self.backend_var = tk.StringVar(value="Accelerator: detected only when Generate is clicked")
        self.progress_var = tk.DoubleVar(value=0)

        self.result = None
        self.preview_photo = None
        self.worker = None
        self.cancel_event = threading.Event()
        self.events = queue.Queue()

        self._build_ui()
        self.root.after(100, self._poll_events)

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("vista" if os.name == "nt" else "clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 17, "bold"))
        style.configure("Subtle.TLabel", foreground="#666666")
        style.configure("Action.TButton", padding=(12, 7))

        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="QR Image Forge", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Turn an image into a visual target, then optimize a real, decodable QR payload.",
            style="Subtle.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        image_frame = ttk.LabelFrame(outer, text="Source image", padding=10)
        image_frame.pack(fill="x", pady=(0, 10))
        image_row = ttk.Frame(image_frame)
        image_row.pack(fill="x")
        ttk.Entry(image_row, textvariable=self.image_path).pack(side="left", fill="x", expand=True)
        ttk.Button(image_row, text="Browse…", command=self._choose_image).pack(side="left", padx=(8, 0))

        controls = ttk.LabelFrame(outer, text="QR settings", padding=10)
        controls.pack(fill="x", pady=(0, 10))
        for column in (1, 3, 5):
            controls.columnconfigure(column, weight=1)

        ttk.Label(controls, text="Version / resolution").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=5)
        version_box = ttk.Combobox(controls, textvariable=self.version_var, values=[str(i) for i in range(1, 41)], width=7, state="readonly")
        version_box.grid(row=0, column=1, sticky="ew", padx=(0, 14), pady=5)
        ttk.Label(controls, text="Error correction").grid(row=0, column=2, sticky="w", padx=(0, 6), pady=5)
        ttk.Combobox(controls, textvariable=self.ecc_var, values=list(ECC_NAMES), width=22, state="readonly").grid(row=0, column=3, sticky="ew", padx=(0, 14), pady=5)
        ttk.Label(controls, text="Dither").grid(row=0, column=4, sticky="w", padx=(0, 6), pady=5)
        ttk.Combobox(controls, textvariable=self.dither_var, values=DITHER_METHODS, width=19, state="readonly").grid(row=0, column=5, sticky="ew", pady=5)

        ttk.Label(controls, text="Search effort").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=5)
        effort_box = ttk.Combobox(controls, textvariable=self.effort_var, values=list(EFFORTS), width=10, state="readonly")
        effort_box.grid(row=1, column=1, sticky="ew", padx=(0, 14), pady=5)
        effort_box.bind("<<ComboboxSelected>>", self._effort_changed)
        ttk.Label(controls, text="Custom steps").grid(row=1, column=2, sticky="w", padx=(0, 6), pady=5)
        self.custom_steps_entry = ttk.Entry(controls, textvariable=self.custom_steps_var, width=9)
        self.custom_steps_entry.grid(row=1, column=3, sticky="ew", padx=(0, 14), pady=5)
        ttk.Label(controls, text="PNG module scale").grid(row=1, column=4, sticky="w", padx=(0, 6), pady=5)
        ttk.Combobox(controls, textvariable=self.scale_var, values=("6", "8", "10", "12", "16", "20", "24"), width=8, state="readonly").grid(row=1, column=5, sticky="ew", pady=5)

        options_row = ttk.Frame(controls)
        options_row.grid(row=2, column=0, columnspan=6, sticky="ew", pady=(5, 0))
        ttk.Checkbutton(options_row, text="Invert source image", variable=self.invert_var).pack(side="left")
        ttk.Label(options_row, text="Optional random seed:").pack(side="left", padx=(18, 6))
        ttk.Entry(options_row, textvariable=self.seed_var, width=18).pack(side="left")
        ttk.Label(options_row, text="Leave blank for a random run", style="Subtle.TLabel").pack(side="left", padx=(8, 0))

        work = ttk.Frame(outer)
        work.pack(fill="both", expand=True)
        preview_frame = ttk.LabelFrame(work, text="Generated QR preview", padding=10)
        preview_frame.pack(side="left", fill="both", expand=True)
        self.preview_label = ttk.Label(preview_frame, text="No QR generated yet", anchor="center")
        self.preview_label.pack(fill="both", expand=True)

        info_frame = ttk.LabelFrame(work, text="Generation details", padding=10)
        info_frame.pack(side="left", fill="y", padx=(10, 0))
        info_frame.configure(width=250)
        info_frame.pack_propagate(False)
        self.details_text = tk.Text(info_frame, width=28, height=12, wrap="word", relief="flat", background=self.root.cget("background"), font=("Segoe UI", 9))
        self.details_text.pack(fill="both", expand=True)
        self.details_text.insert("1.0", "The QR matrix is generated from a binary image target.\n\nThe payload is arbitrary binary data, not a URL or an encoded copy of your image.")
        self.details_text.configure(state="disabled")

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(12, 0))
        self.progress = ttk.Progressbar(footer, variable=self.progress_var, maximum=100)
        self.progress.pack(fill="x", pady=(0, 6))
        status_row = ttk.Frame(footer)
        status_row.pack(fill="x")
        ttk.Label(status_row, textvariable=self.status_var).pack(side="left", fill="x", expand=True)
        ttk.Label(status_row, textvariable=self.backend_var, style="Subtle.TLabel").pack(side="right")

        buttons = ttk.Frame(footer)
        buttons.pack(fill="x", pady=(8, 0))
        self.generate_button = ttk.Button(buttons, text="Generate QR", style="Action.TButton", command=self._start_generation)
        self.generate_button.pack(side="left")
        self.cancel_button = ttk.Button(buttons, text="Cancel", command=self._cancel_generation, state="disabled")
        self.cancel_button.pack(side="left", padx=(8, 0))
        self.save_button = ttk.Button(buttons, text="Export PNG…", style="Action.TButton", command=self._save_png, state="disabled")
        self.save_button.pack(side="right")

        ttk.Label(outer, text="QR version 1 = 21×21 modules · version 40 = 177×177 modules · PNG includes a 4-module quiet zone", style="Subtle.TLabel").pack(anchor="w", pady=(8, 0))
        self._effort_changed()

    def _effort_changed(self, _event=None):
        self.custom_steps_entry.configure(state="normal" if self.effort_var.get() == "Custom" else "disabled")

    def _choose_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.image_path.set(path)
            self.status_var.set(f"Selected: {Path(path).name}")

    def _start_generation(self):
        image_path = self.image_path.get().strip()
        if not image_path or not Path(image_path).is_file():
            messagebox.showerror(APP_NAME, "Choose a valid source image first.")
            return
        try:
            version = int(self.version_var.get())
            if not 1 <= version <= 40:
                raise ValueError
            custom_steps = int(self.custom_steps_var.get() or "0")
            if self.effort_var.get() == "Custom" and not 0 <= custom_steps <= 600:
                raise ValueError
            seed = int(self.seed_var.get()) if self.seed_var.get().strip() else None
            if seed is not None and not -(2**63) <= seed < 2**64:
                raise ValueError
        except ValueError:
            messagebox.showerror(APP_NAME, "Check the QR version, custom step count (0–600), and optional seed.")
            return

        ecc_letter = ECC_NAMES[self.ecc_var.get()]
        self.result = None
        self.save_button.configure(state="disabled")
        self.generate_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress_var.set(0)
        self.status_var.set("Preparing the image and QR matrix…")
        self.backend_var.set("Detecting optional GPU backend…")
        self.preview_label.configure(image="", text="Generating…")
        self.cancel_event.clear()

        args = (
            image_path,
            version,
            ecc_letter,
            self.dither_var.get(),
            self.invert_var.get(),
            self.effort_var.get(),
            custom_steps,
            seed,
        )
        self.worker = threading.Thread(target=self._worker_generate, args=args, daemon=True)
        self.worker.start()

    def _worker_generate(self, *args):
        try:
            def progress_callback(step, total, score, backend):
                self.events.put(("progress", step, total, score, backend))

            result = generate_qr_image(
                *args,
                cancelled=self.cancel_event.is_set,
                progress_callback=progress_callback,
            )
            self.events.put(("done", result))
        except InterruptedError:
            self.events.put(("cancelled",))
        except Exception as exc:
            self.events.put(("error", str(exc), traceback.format_exc()))

    def _cancel_generation(self):
        self.cancel_event.set()
        self.cancel_button.configure(state="disabled")
        self.status_var.set("Cancelling after the current QR operation…")

    def _poll_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "progress":
                    _, step, total, score, backend = event
                    self.progress_var.set((step / max(1, total)) * 100.0)
                    self.status_var.set(f"Search {step}/{total} · best match {score:.2f}%")
                    self.backend_var.set(f"Scoring: {backend}")
                elif kind == "done":
                    self._generation_done(event[1])
                elif kind == "cancelled":
                    self._generation_cancelled()
                elif kind == "error":
                    self._generation_error(event[1], event[2])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _generation_done(self, result):
        self.result = result
        self.generate_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.save_button.configure(state="normal")
        self.progress_var.set(100)
        n = result["matrix"].shape[0]
        self.status_var.set(f"Complete · {n}×{n} modules · {result['score']:.2f}% visual match")
        self.backend_var.set(f"Scoring: {result['backend']}")
        self._show_preview(result["matrix"])

        details = (
            f"Version: {result['version']}\n"
            f"Matrix: {n} × {n}\n"
            f"Error correction: {result['ecc']}\n"
            f"Mask pattern: {result['mask']}\n"
            f"Binary payload: {result['payload_length']} bytes\n"
            f"Visual agreement: {result['score']:.2f}%\n"
            f"Search steps: {result['steps']}\n"
            f"Scoring backend: {result['backend']}\n\n"
            "This is a standards-based QR symbol with a byte-mode payload. The payload is intentionally arbitrary and may not display as readable text in QR scanner apps."
        )
        self.details_text.configure(state="normal")
        self.details_text.delete("1.0", "end")
        self.details_text.insert("1.0", details)
        self.details_text.configure(state="disabled")

    def _show_preview(self, matrix):
        try:
            Image = importlib.import_module("PIL.Image")
            ImageOps = importlib.import_module("PIL.ImageOps")
            ImageTk = importlib.import_module("PIL.ImageTk")
            image = Image.fromarray((1 - matrix.astype("uint8")) * 255, mode="L")
            image = ImageOps.expand(image, border=4, fill=255)
            image.thumbnail((360, 360), Image.Resampling.NEAREST)
            self.preview_photo = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.preview_photo, text="")
        except Exception as exc:
            self.preview_label.configure(image="", text=f"Preview unavailable:\n{exc}")

    def _save_png(self):
        if not self.result:
            return
        output = filedialog.asksaveasfilename(
            title="Export QR code as PNG",
            defaultextension=".png",
            initialfile=f"qr_image_v{self.result['version']}.png",
            filetypes=[("PNG image", "*.png")],
        )
        if not output:
            return
        try:
            Image = importlib.import_module("PIL.Image")
            ImageOps = importlib.import_module("PIL.ImageOps")
            np = importlib.import_module("numpy")
            matrix = self.result["matrix"]
            image = Image.fromarray((1 - matrix.astype(np.uint8)) * 255, mode="L")
            image = ImageOps.expand(image, border=4, fill=255)
            scale = max(1, min(64, int(self.scale_var.get())))
            image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
            image.save(output, format="PNG", optimize=False)
            self.status_var.set(f"Saved: {output}")
            messagebox.showinfo(APP_NAME, f"PNG exported successfully.\n\n{output}")
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"Could not save PNG:\n{exc}")

    def _generation_cancelled(self):
        self.generate_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.progress_var.set(0)
        self.status_var.set("Generation cancelled.")
        self.backend_var.set("No active scoring backend")
        self.preview_label.configure(image="", text="Generation cancelled")

    def _generation_error(self, message, details):
        self.generate_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.progress_var.set(0)
        self.status_var.set("Generation failed. See error details.")
        self.backend_var.set("No active scoring backend")
        self.preview_label.configure(image="", text="Generation failed")
        messagebox.showerror(APP_NAME, f"{message}\n\nTechnical details:\n{details[-1800:]}")


def main():
    root = tk.Tk()
    QRImageForgeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
