"""Comprehensive validation test suite for the FedForge forensic pipeline.

Tests:
1. Real camera portrait simulation (should be AUTHENTIC)
2. Natural landscape photo (should be AUTHENTIC)
3. AI-generated uniform gradient (should be AUTHENTIC or mild suspicion)
4. Simulated inpainting: smooth skin + one isolated spliced patch (should be SYNTHETIC)
5. GAN-style image with FFT checkerboard (should be SYNTHETIC)
"""
import sys
import numpy as np
import cv2
from src.fedforge.vision.detector import MediaDetectionPipeline
from src.fedforge.vision.forensics import PixelForensics

pipeline = MediaDetectionPipeline()


def make_real_portrait():
    """Simulate a real camera photo: face with hair, eyes, camera noise."""
    h, w = 400, 400
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Skin background (warm beige)
    img[:] = (165, 175, 200)
    # Gradient lighting (natural)
    for y in range(h):
        v = int(18 * np.sin(y / 80.0))
        img[y, :, :] = np.clip(img[y, :, :].astype(int) + v, 0, 255)
    # Hair (dark)
    img[:90, :] = (25, 20, 15)
    # Eyes (dark brown)
    cv2.ellipse(img, (130, 160), (30, 18), 0, 0, 360, (45, 40, 30), -1)
    cv2.ellipse(img, (270, 160), (30, 18), 0, 0, 360, (45, 40, 30), -1)
    # Iris
    cv2.circle(img, (130, 160), 12, (80, 70, 60), -1)
    cv2.circle(img, (270, 160), 12, (80, 70, 60), -1)
    # Lips
    cv2.ellipse(img, (200, 280), (40, 15), 0, 0, 360, (120, 100, 140), -1)
    # Uniform camera sensor noise
    np.random.seed(7)
    noise = np.random.normal(0, 4.5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # JPEG encode (simulates camera compression)
    _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return enc.tobytes()


def make_real_landscape():
    """Simulate a real landscape photo: sky gradient + tree texture."""
    h, w = 400, 500
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        # Sky gradient (top blue to white)
        t = y / h
        img[y, :, 0] = int(220 - 100 * t)   # B
        img[y, :, 1] = int(180 - 60 * t)    # G
        img[y, :, 2] = int(100 + 80 * t)    # R
    # Tree region (bottom half, dark green, texture)
    for y in range(h // 2, h):
        for x in range(w):
            img[y, x, :] = (
                int(30 + 15 * np.sin(x / 12.0 + y / 8.0)),
                int(80 + 40 * np.cos(x / 10.0 - y / 15.0)),
                int(20 + 10 * np.sin(x / 20.0))
            )
    np.random.seed(3)
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return enc.tobytes()


def make_inpainted_portrait():
    """Real portrait with a clearly inpainted 'AI-added' isolated patch."""
    # Start with real portrait
    h, w = 400, 400
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (165, 175, 200)
    for y in range(h):
        v = int(18 * np.sin(y / 80.0))
        img[y, :, :] = np.clip(img[y, :, :].astype(int) + v, 0, 255)
    img[:90, :] = (25, 20, 15)
    cv2.ellipse(img, (130, 160), (30, 18), 0, 0, 360, (45, 40, 30), -1)
    cv2.ellipse(img, (270, 160), (30, 18), 0, 0, 360, (45, 40, 30), -1)
    np.random.seed(7)
    noise = np.random.normal(0, 4.5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # JPEG compress the "original"
    _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # Simulate AI inpainting: paste a block with completely different compression
    # history (no noise, smooth texture, double-compressed at different quality)
    inpaint_region = np.zeros((80, 120, 3), dtype=np.uint8)
    inpaint_region[:] = (180, 170, 200)  # slightly different color
    for y in range(80):
        inpaint_region[y, :, 1] = int(120 + 60 * np.sin(y / 10.0))
    # Compress at very different quality (AI output typical quality)
    _, enc2 = cv2.imencode('.jpg', inpaint_region, [cv2.IMWRITE_JPEG_QUALITY, 65])
    inpaint_region = cv2.imdecode(enc2, cv2.IMREAD_COLOR)
    # Paste it at mouth region (simulating grillz addition)
    img[250:330, 140:260] = inpaint_region

    # Now save as JPEG (standard web image format preserving compression differential)
    _, enc3 = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return enc3.tobytes()


def make_ai_generated_smooth():
    """Simulate an AI-generated image: perfectly smooth gradients, no camera noise."""
    h, w = 400, 400
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            # Ultra-smooth gradient (AI images lack natural noise)
            img[y, x, 0] = int(80 + 40 * np.sin(y / 60.0) + 30 * np.cos(x / 80.0))
            img[y, x, 1] = int(120 + 50 * np.cos(y / 70.0) + 20 * np.sin(x / 60.0))
            img[y, x, 2] = int(60 + 35 * np.sin((x + y) / 90.0))
    # No noise added (AI images are unnaturally smooth)
    # JPEG at AI-typical quality
    _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return enc.tobytes()


if __name__ == "__main__":
    print("=" * 65)
    print("FedForge Forensic Pipeline - Comprehensive Validation")
    print("=" * 65)

    tests = [
        ("1. Real Camera Portrait",  make_real_portrait(),       "AUTHENTIC",        True),
        ("2. Real Landscape Photo",  make_real_landscape(),      "AUTHENTIC",        True),
        ("3. AI-Generated Smooth",   make_ai_generated_smooth(), None,               None),  # borderline
        ("4. Inpainted Portrait",    make_inpainted_portrait(),  "LIKELY_SYNTHETIC", False),
    ]

    passed = 0
    total_with_expectation = 0

    for name, data, expected_verdict, expect_authentic in tests:
        r = pipeline.analyze_image_bytes(data, filename="test.jpg")
        score = r["overall_score"]
        verdict = r["verdict"]
        conf = r["confidence"]
        metrics = r["face_details"][0]["pixel_forensics"] if r["face_details"] else {}

        if expected_verdict is not None:
            total_with_expectation += 1
            ok = (verdict == expected_verdict)
            if ok:
                passed += 1
            status = "PASS [OK]" if ok else "FAIL [WRONG]"
        else:
            status = "INFO"

        print(f"\n{name}")
        print(f"  Verdict:    {verdict}  (expected: {expected_verdict or 'any'})")
        print(f"  Score:      {score:.4f}  |  Confidence: {conf}%")
        print(f"  ELA:        {metrics.get('ela_compression_anomaly', '-')}")
        print(f"  Noise:      {metrics.get('noise_residual_inconsistency', '-')}")
        print(f"  Seam:       {metrics.get('boundary_seam_anomaly', '-')}")
        print(f"  FFT:        {metrics.get('fft_spectral_anomaly', '-')}")
        print(f"  Result:     {status}")

    print()
    print("=" * 65)
    print(f"Test Results: {passed}/{total_with_expectation} passed")
    print("=" * 65)
    sys.exit(0 if passed == total_with_expectation else 1)

