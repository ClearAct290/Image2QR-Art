# QR Image Forge
<br>
<img width="4440" height="4440" alt="qr_image_v40" src="https://github.com/user-attachments/assets/8e8d02d3-4a0b-4d1d-b21a-f6fe8123b770" width="360" height="360" />
<br>
**A valid-QR image approximation experiment** by **clearact290**.

QR Image Forge converts a selected image into a black-and-white dithered target, then builds and optimizes a real QR Code whose black/white modules resemble that target as closely as possible.

The QR's payload is arbitrary binary data. It is not the source image or a URL, and it is not guaranteed to display as readable text when scanned. The symbol is encoded as an ordinary QR Code in byte mode, with normal format information, finder patterns, timing patterns, masking, and Reed–Solomon error correction. QR readers differ in how they expose arbitrary non-text bytes; some may display replacements or decline to show the payload even though the QR symbol itself is structurally valid.

## Features

- QR versions **1–40**, from **21×21** to **177×177** modules.
- Error correction levels **L, M, Q, and H**.
- Floyd–Steinberg, Atkinson, Bayer 4×4, and threshold dithering.
- An image-led initial payload: the app maps target pixels into QR data modules before searching.
- Payload mutation and simulated-annealing search to improve the visual match without invalidating the QR encoding.
- PNG export with a four-module white quiet zone and configurable pixel scale.
- Optional CUDA / ROCm acceleration for batched candidate scoring through CuPy or PyTorch. CPU / NumPy is used automatically when no supported GPU backend is available.
- A small Tkinter interface. GPU frameworks are imported only after you press **Generate QR**, so they do not slow down application startup.

## Requirements

- Python 3.10 or newer.
- Tkinter (included with most Windows Python installations; Linux distributions may package it separately).
- The packages in `requirements.txt`.

## Run from source

Open a terminal in this folder and run:

### Windows

```powershell
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
py main.py
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

## How to use

1. Click **Browse…** and select an image.
2. Select a QR version. Version 1 is the smallest symbol; version 40 is the largest.
3. Choose the error-correction level. `L` leaves the most room for payload data, which is usually best for visual matching. Higher correction levels make the symbol more robust but leave less room for the image-led payload.
4. Choose a dithering method. Floyd–Steinberg is the recommended starting point for photographs.
5. Choose **Quick**, **Balanced**, or **High** search effort. **Custom** allows 0–600 search steps. More search means more candidate QR codes are encoded and tested.
6. Click **Generate QR**. Generation runs in a worker thread; the window remains responsive, and **Cancel** requests a stop.
7. Click **Export PNG…** to save the final symbol.

The match percentage is the percentage of QR modules that agree with the dithered image target. QR function patterns (such as the three finder squares) cannot be changed, so a 100% match is generally impossible. Larger versions have more visual detail, but take longer to search.

## GPU acceleration

The base installation does **not** install a GPU framework. The application checks for one lazily when generation begins:

1. CuPy with a working CUDA or ROCm/HIP runtime.
2. Otherwise, PyTorch with `torch.cuda.is_available()` reporting a working CUDA or ROCm/HIP device.
3. Otherwise, CPU / NumPy.

GPU acceleration is used for **batched candidate-matrix scoring**. QR Code construction and Reed–Solomon encoding are still performed by `python-qrcode` on the CPU. For small QR versions, GPU transfer overhead can outweigh the scoring savings, so GPU mode is not guaranteed to be faster in every case.

Install a GPU build that matches your system by following the official installation instructions for [CuPy](https://docs.cupy.dev/en/stable/install.html) or [PyTorch](https://pytorch.org/get-started/locally/). For AMD ROCm support, follow the current [AMD ROCm PyTorch instructions](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html). The app automatically falls back to CPU if a GPU package cannot initialize a supported device.

## Build a Windows executable with PyInstaller

```powershell
py -m pip install -r requirements.txt
py -m pip install pyinstaller
py -m PyInstaller --noconfirm --clean --onefile --windowed --name QRImageForge --collect-all qrcode --collect-all PIL --collect-all numpy main.py
```

The executable will be created at:

```text
dist\QRImageForge.exe
```

Or run `build_windows.bat` to install the base requirements and build it.

### Packaging optional GPU support

The default PyInstaller command deliberately excludes optional GPU frameworks so the executable stays smaller and starts more quickly. If you want GPU support inside a packaged build, install a compatible GPU framework in the build environment and add its collection option, for example `--collect-all torch` or `--collect-all cupy`. GPU runtime libraries and drivers may still need to be installed on the target computer. A PyInstaller build containing PyTorch or CuPy can be very large.

## Notes and limitations

- QR version determines module resolution: `17 + 4 × version` modules per side.
- The PNG has a standard four-module quiet zone around the QR symbol.
- A QR scanner can decode the binary payload, but applications may display its bytes differently. The generated data is not intended to be human-readable.
- The source image is only a visual target. It cannot be reproduced perfectly because QR function patterns, mask rules, codeword structure, and error-correction bytes constrain the final pattern.
- This program generates one QR code at a time; it does not split an image into multiple linked QR codes.

## License

This project is distributed under the **GNU General Public License version 3.0 only**. See [`LICENSE`](LICENSE).
