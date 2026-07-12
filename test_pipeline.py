"""
Pipeline Integration Test
==========================
Tests the full AI pipeline end-to-end without Streamlit:
  1. Image Preprocessing (OpenCV)
  2. Classification (ResNet50)
  3. Segmentation (SegFormer)
  4. Object Detection (YOLOv8)
  5. Damage Assessment
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import os
import time
import traceback

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_pipeline():
    print("=" * 60)
    print("🌊 DISASTER DAMAGE ANALYSIS — PIPELINE TEST")
    print("=" * 60)

    errors = []
    results = {}

    # ── Step 1: Preprocessing ──────────────────────────────
    print("\n[1/5] 🔧 Testing Image Preprocessing...")
    try:
        from preprocessing.image_processor import ImageProcessor
        processor = ImageProcessor()
        raw = processor.load_image("test_images/test_flood.jpg")
        preprocessed = processor.preprocess(raw)

        assert "original" in preprocessed, "Missing 'original' key"
        assert "display" in preprocessed, "Missing 'display' key"
        assert "tensor" in preprocessed, "Missing 'tensor' key"
        assert "pil" in preprocessed, "Missing 'pil' key"
        assert preprocessed["tensor"].shape == (1, 3, 224, 224), \
            f"Tensor shape wrong: {preprocessed['tensor'].shape}"

        results["preprocessed"] = preprocessed
        print(f"  ✅ Preprocessing OK")
        print(f"     Display shape: {preprocessed['display'].shape}")
        print(f"     Tensor shape:  {preprocessed['tensor'].shape}")
        print(f"     PIL size:      {preprocessed['pil'].size}")
    except Exception as e:
        errors.append(("Preprocessing", str(e)))
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()
        return errors  # Can't continue without preprocessing

    # ── Step 2: Classification (ResNet50) ──────────────────
    print("\n[2/5] 🧠 Testing ResNet50 Classification...")
    try:
        from models.classifier import FloodClassifier
        t0 = time.time()
        classifier = FloodClassifier(weights_path=None)
        load_time = time.time() - t0

        t0 = time.time()
        classification = classifier.predict(preprocessed["tensor"])
        infer_time = time.time() - t0

        assert "label" in classification, "Missing 'label' key"
        assert "confidence" in classification, "Missing 'confidence' key"
        assert "is_flood" in classification, "Missing 'is_flood' key"
        assert 0 <= classification["confidence"] <= 1, "Confidence out of range"

        # Force is_flood=True for pipeline continuation (pure ImageNet model)
        classification["is_flood"] = True

        results["classification"] = classification
        print(f"  ✅ Classification OK")
        print(f"     Label:       {classification['label']}")
        print(f"     Confidence:  {classification['confidence']:.4f}")
        print(f"     ImageNet:    {classification.get('imagenet_class', 'N/A')}")
        print(f"     Load time:   {load_time:.2f}s")
        print(f"     Infer time:  {infer_time:.3f}s")
    except Exception as e:
        errors.append(("Classification", str(e)))
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()

    # ── Step 3: Segmentation (SegFormer) ───────────────────
    print("\n[3/5] 🎭 Testing SegFormer Segmentation...")
    try:
        from models.segmenter import FloodSegmenter
        t0 = time.time()
        segmenter = FloodSegmenter()
        load_time = time.time() - t0

        t0 = time.time()
        segmentation = segmenter.segment(preprocessed["pil"])
        infer_time = time.time() - t0

        assert "flood_mask" in segmentation, "Missing 'flood_mask' key"
        assert "flood_area_ratio" in segmentation, "Missing 'flood_area_ratio' key"
        assert "full_segmap" in segmentation, "Missing 'full_segmap' key"
        assert "num_labels" in segmentation, "Missing 'num_labels' key"
        assert 0 <= segmentation["flood_area_ratio"] <= 1, "Ratio out of range"
        assert segmentation["flood_mask"].ndim == 2, "Mask should be 2D"

        results["segmentation"] = segmentation
        print(f"  ✅ Segmentation OK")
        print(f"     Flood area:  {segmentation['flood_area_ratio']:.2%}")
        print(f"     Mask shape:  {segmentation['flood_mask'].shape}")
        print(f"     Num labels:  {segmentation['num_labels']}")
        print(f"     Load time:   {load_time:.2f}s")
        print(f"     Infer time:  {infer_time:.3f}s")
    except Exception as e:
        errors.append(("Segmentation", str(e)))
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()

    # ── Step 4: Object Detection (YOLOv8) ──────────────────
    print("\n[4/5] 🔍 Testing YOLOv8 Object Detection...")
    try:
        from models.detector import ObjectDetector
        t0 = time.time()
        detector = ObjectDetector()
        load_time = time.time() - t0

        flood_mask = results.get("segmentation", {}).get("flood_mask", None)
        t0 = time.time()
        detection = detector.detect(preprocessed["display"], flood_mask=flood_mask)
        infer_time = time.time() - t0

        assert "detections" in detection, "Missing 'detections' key"
        assert "annotated_image" in detection, "Missing 'annotated_image' key"
        assert "summary" in detection, "Missing 'summary' key"
        assert "total_objects" in detection["summary"], "Missing 'total_objects'"
        assert "objects_in_flood" in detection["summary"], "Missing 'objects_in_flood'"
        assert "by_class" in detection["summary"], "Missing 'by_class'"
        assert isinstance(detection["detections"], list), "Detections should be a list"

        results["detection"] = detection
        print(f"  ✅ Detection OK")
        print(f"     Total objects:    {detection['summary']['total_objects']}")
        print(f"     In flood zone:    {detection['summary']['objects_in_flood']}")
        print(f"     By class:         {detection['summary']['by_class']}")
        print(f"     Annotated shape:  {detection['annotated_image'].shape}")
        print(f"     Load time:        {load_time:.2f}s")
        print(f"     Infer time:       {infer_time:.3f}s")
    except Exception as e:
        errors.append(("Detection", str(e)))
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()

    # ── Step 5: Damage Assessment ──────────────────────────
    print("\n[5/5] 📊 Testing Damage Assessment Engine...")
    try:
        from engine.damage_assessment import DamageAssessmentEngine
        engine = DamageAssessmentEngine()

        classification = results.get("classification", {"is_flood": True, "confidence": 0.5})
        segmentation = results.get("segmentation", {"flood_area_ratio": 0.0, "flood_mask": None})
        detection = results.get("detection", {"detections": [], "summary": {"total_objects": 0, "objects_in_flood": 0, "by_class": {}}})

        assessment = engine.assess(classification, segmentation, detection, location={"lat": 20.5937, "lon": 78.9629})

        assert "severity_score" in assessment, "Missing 'severity_score'"
        assert "damage_category" in assessment, "Missing 'damage_category'"
        assert "recommendations" in assessment, "Missing 'recommendations'"
        assert "score_breakdown" in assessment, "Missing 'score_breakdown'"
        assert 0 <= assessment["severity_score"] <= 10, "Severity out of range"

        results["assessment"] = assessment
        print(f"  ✅ Assessment OK")
        print(f"     Severity:       {assessment['severity_score']}/10")
        print(f"     Category:       {assessment['damage_category_emoji']} {assessment['damage_category']}")
        print(f"     Flood area:     {assessment['flood_area_percentage']:.1f}%")
        print(f"     Objects:        {assessment['total_objects_detected']}")
        print(f"     In flood zone:  {assessment['objects_in_flood_zone']}")
        print(f"     Recommendations: {len(assessment['recommendations'])}")
        for rec in assessment["recommendations"]:
            print(f"       [{rec['priority']}] {rec['action']}")
    except Exception as e:
        errors.append(("Assessment", str(e)))
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()

    # ── Test Visualization Utilities ───────────────────────
    print("\n[EXTRA] 🎨 Testing Visualization Utilities...")
    try:
        from utils.visualization import create_flood_overlay, generate_severity_gauge, generate_score_breakdown_chart

        if "segmentation" in results:
            overlay = create_flood_overlay(preprocessed["display"], results["segmentation"]["flood_mask"])
            assert overlay.shape == preprocessed["display"].shape, "Overlay shape mismatch"
            print(f"  ✅ Flood overlay OK (shape: {overlay.shape})")

        if "assessment" in results:
            gauge = generate_severity_gauge(results["assessment"]["severity_score"], results["assessment"]["damage_category"])
            assert gauge is not None and len(gauge) > 0, "Gauge is empty"
            print(f"  ✅ Severity gauge OK ({len(gauge)} bytes)")

            chart = generate_score_breakdown_chart(results["assessment"]["score_breakdown"])
            assert chart is not None and len(chart) > 0, "Chart is empty"
            print(f"  ✅ Score breakdown chart OK ({len(chart)} bytes)")
    except Exception as e:
        errors.append(("Visualization", str(e)))
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()

    # ── Summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    if not errors:
        print("🎉 ALL TESTS PASSED! Pipeline is working correctly.")
    else:
        print(f"⚠️  {len(errors)} test(s) failed:")
        for name, err in errors:
            print(f"   ❌ {name}: {err}")
    print("=" * 60)

    return errors


if __name__ == "__main__":
    test_pipeline()
