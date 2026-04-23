import os
import time
import glob
import csv
import json
import torch
import cv2
import logging
import warnings
import numpy as np
from pathlib import Path
from PIL import Image

from model_adapter import get_model_adapter
from benchmark_utils import (
    get_benchmark_config,
    get_duration_ms,
    get_utc_timestamps,
    calculate_stats,
    process_segmentation,
    process_classification,
    process_detection,
    COCO_CLASSES,
    COLORS,
    analyze_segmentation_mask,
    start_docker_monitor,
    stop_docker_monitor,
    start_system_monitor,
    stop_system_monitor,
    stabilize_torch_compile,
    load_ground_truth,
    calculate_iou
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    # 1. LOAD CONFIGURATION FROM UTILS
    config = get_benchmark_config()

    if "remote" in config["backend"]:
        print(f"   🔎 VACCEL_RPC_ADDRESS={os.environ.get('VACCEL_RPC_ADDRESS')}")

    print(f"\n🚀 STARTING MODEL BENCHMARK (With Accuracy Check)")
    print(f"   Backend: {config['backend']}")
    print(f"   Host:    {config['host']}")
    print(f"   Model:   {config['core_model_name']}")
    print(f"   Type:    {config['model_type']}")
    print(f"   Device:  {config['target_device']}")
    print(f"   Monitor: {'ON' if config['monitor_resources'] else 'OFF'}")

    CURRENT_MODEL_DIR = Path("models") / config['model_arch']
    print(f"   Loading: {CURRENT_MODEL_DIR}")

    if config['torch_device'].type == "cuda" and not torch.cuda.is_available():
        print("   ❌ GPU was selected but no GPU is available.")
        return

    try:
        adapter = get_model_adapter(config['model_arch'], config['backend'], config['device'])
        adapter.load_model(CURRENT_MODEL_DIR)
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return

    # 2. LOAD GROUND TRUTH & SELECT FILES
    gt_data = load_ground_truth(config['model_type'])
    files_to_process = []
    is_processing_video_files = config['is_video_model']

    if gt_data:
        if config['model_type'] == "semantic_segmentation":
            base_path = Path("data/coco_segmentation/images")
            limit = config['num_images']
        elif config['model_type'] == "object_detection":
            base_path = Path("data/coco_detection/images")
            limit = config['num_images']
        elif config['model_type'] == "video_classification":
            base_path = Path("data/kinetics/videos")
            limit = config['num_videos']
        else: # image_classification
            base_path = Path("data/imagenet/images")
            limit = config['num_images']

        files_to_process = [str(base_path / item['file_name']) for item in gt_data[:limit]]
        print(f"   ✅ Loaded {len(files_to_process)} samples from Manifest (Ground Truth available).")
    else:
        print(f"   ⚠️  No manifest found for {config['model_type']}. Falling back to folder scan (No Accuracy).")
        DATA_DIRS = [Path("data/images"), Path("data/videos")]
        for d in DATA_DIRS:
            if d.exists():
                if config['is_video_model']:
                    files_to_process.extend(sorted(glob.glob(str(d / "*.mp4")))[:config['num_videos']])
                else:
                    files_to_process.extend(sorted(glob.glob(str(d / "*.jpg")))[:config['num_images']])

    if not files_to_process:
        print("   ❌ No data found to process.")
        return

    # --- torch.compile stabilization (ptc only) ---
    if config["backend"] == "ptc" and files_to_process:
        stabilize_torch_compile(
            adapter=adapter,
            sample_input=files_to_process[0],
            torch_device=config["torch_device"],
            iters=int(os.environ.get("PTC_STABILIZE_ITERS", "10")),
            do_postprocess=False,
        )

    # 4. PREPARE OUTPUT ID & DIRECTORIES
    prefix = config['run_tag'] if config['run_tag'] else time.strftime("%d-%m-%Y_%H-%M-%S")
    local_mode = "gpu" if (config['torch_device'].type == "cuda") else "cpu"

    is_vaccel_remote_run = (config['host'] == "robot" and "remote" in config['backend'])
    if is_vaccel_remote_run:
        run_id = f"{prefix}_{config['core_model_name']}_{config['backend']}_{config['host']}_{local_mode}_target-{config['target_device']}"
    else:
        run_id = f"{prefix}_{config['core_model_name']}_{config['backend']}_{config['host']}_{local_mode}"

    run_dir = config['base_results_dir'] / config['host'] / run_id
    img_out_dir = run_dir / "output_images"

    if config['export_results']:
        run_dir.mkdir(parents=True, exist_ok=True)
        if config['export_output_images']:
            img_out_dir.mkdir(exist_ok=True)
        print(f"   📂 Output Directory: {run_dir}")

    # 5. WARMUP
    print("   🔥 Warming up...")
    for i in range(min(5, len(files_to_process))):
        try:
            dummy_input = adapter.preprocess(files_to_process[i])
            with torch.inference_mode():
                raw_output = adapter.infer(dummy_input)
                _ = adapter.postprocess(raw_output)
                if config['torch_device'].type == 'cuda': torch.cuda.synchronize()
        except Exception as e:
            print(f"      Warmup failed on {os.path.basename(files_to_process[i])}: {e}")

    # --- 6. RESOURCE MONITORING (START) ---
    if config['monitor_resources']:
        docker_csv_dir = str(Path(config['docker_csv_base']) / config['host'])
        system_csv_dir = str(Path(config['system_csv_base']) / config['host'])

        start_docker_monitor(run_id, config['docker_endpoint'], docker_csv_dir, "torchvision-app", "torchvision-app_")
        start_system_monitor(run_id, config['system_endpoint'], system_csv_dir, [local_mode])

        if is_vaccel_remote_run:
            vaccel_remote_run_id = f"{prefix}_{config['core_model_name']}_{config['backend']}_edge-asus_{config['target_device']}"
            rem_docker_csv_dir = str(Path(config['docker_csv_base']) / "edge-asus")
            rem_system_csv_dir = str(Path(config['system_csv_base']) / "edge-asus")
            remote_mode = "gpu" if config['target_device'].lower() in ["cuda", "gpu"] else "cpu"

            start_docker_monitor(vaccel_remote_run_id, config['remote_docker_endpoint'], rem_docker_csv_dir, "torchvision-app-agent", "torchvision-app-agent_")
            start_system_monitor(vaccel_remote_run_id, config['remote_system_endpoint'], rem_system_csv_dir, [remote_mode])

        time.sleep(1.2)

    # Capture Start Time
    t_start_dt, t_start_iso = get_utc_timestamps()
    frames_per_sample = 16 if config['is_video_model'] else 1

    # 7. RUN LOOP
    print("   ⏱️  Running Inference...")
    inference_latencies_ms, preprocessing_latencies_ms = [], []
    postprocessing_latencies_ms, total_system_latencies_ms = [], []
    confidence_scores_list, detailed_sample_records = [], []
    accuracy_metrics = []

    try:
        for i, file_path in enumerate(files_to_process):
            file_name = os.path.basename(file_path)
            stem_name = os.path.splitext(file_name)[0]

            # --- A. SYSTEM START ---
            system_start_time_ns = time.perf_counter_ns()

            # --- B. PREPROCESSING ---
            if config['torch_device'].type == 'cuda': torch.cuda.synchronize()
            preprocessing_start_time_ns = time.perf_counter_ns()
            input_tensor = adapter.preprocess(file_path)
            if config['torch_device'].type == 'cuda': torch.cuda.synchronize()
            preprocessing_end_time_ns = time.perf_counter_ns()

            # --- C. INFERENCE ---
            if config['torch_device'].type == 'cuda': torch.cuda.synchronize()
            inference_start_time_ns = time.perf_counter_ns()
            with torch.inference_mode():
                raw_output = adapter.infer(input_tensor)
            if config['torch_device'].type == 'cuda': torch.cuda.synchronize()
            inference_end_time_ns = time.perf_counter_ns()

            # --- D. POSTPROCESSING ---
            if config['torch_device'].type == 'cuda': torch.cuda.synchronize()
            postprocessing_start_time_ns = time.perf_counter_ns()

            class_id, confidence_score, detected_info = -1, 0.0, ""
            pred_mask = None

            try:
                result = adapter.postprocess(raw_output)

                if config['model_type'] == "semantic_segmentation":
                    pred_mask, confidence_score, detected_info = process_segmentation(
                        result, img_out_dir, stem_name, i, config['export_output_images'], COLORS, analyze_segmentation_mask
                    )
                elif config['model_type'] in ["image_classification", "video_classification"]:
                    class_id, confidence_score, detected_info = process_classification(
                        result, file_path, img_out_dir, stem_name, i, config['export_output_images'], adapter, is_processing_video_files
                    )
                elif config['model_type'] == "object_detection":
                    class_id, confidence_score, detected_info = process_detection(
                        result, file_path, img_out_dir, stem_name, i, config['export_output_images'], COCO_CLASSES
                    )
            except Exception as e:
                print(f"Error post-processing {file_name}: {e}")

            if config['torch_device'].type == 'cuda': torch.cuda.synchronize()
            postprocessing_end_time_ns = time.perf_counter_ns()
            system_end_time_ns = time.perf_counter_ns()

            # --- E. ACCURACY LOGIC ---
            acc_score = 0.0
            acc_info = ""
            if gt_data:
                target = gt_data[i]
                if "classification" in config['model_type']:
                    is_correct = (int(class_id) == int(target['ground_truth_id']))
                    acc_score = 1.0 if is_correct else 0.0
                    acc_info = f"| GT: {target['ground_truth_id']} {'✅' if is_correct else '❌'}"
                    
                elif config['model_type'] == "semantic_segmentation":
                    gt_mask_path = Path("data/coco_segmentation/masks") / target['mask_name']
                    if gt_mask_path.exists() and pred_mask is not None:
                        gt_mask = np.array(Image.open(gt_mask_path))
                        original_dims = (gt_mask.shape[1], gt_mask.shape[0]) 
                        pred_mask_resized = cv2.resize(pred_mask, original_dims, interpolation=cv2.INTER_NEAREST)
                        acc_score = calculate_iou(pred_mask_resized, gt_mask)
                        acc_info = f"| IoU: {acc_score:.2%}"

                elif config['model_type'] == "object_detection":
                    gt_boxes = target['objects']['bboxes']
                    gt_classes = target['objects']['categories']
                    if 'boxes' in result:
                        pred_boxes = result["boxes"]
                        pred_scores = result["scores"]
                        pred_classes = result["classes"]
                        
                        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                            max_idx = torch.argmax(pred_scores).item()
                            best_pred_box = pred_boxes[max_idx].cpu().numpy()
                            best_pred_class = int(pred_classes[max_idx].item())
                            
                            best_iou = 0.0
                            for gt_idx, gt_box_coco in enumerate(gt_boxes):
                                gt_class = int(gt_classes[gt_idx])
                                if best_pred_class == gt_class:
                                    gt_x1, gt_y1, w, h = gt_box_coco
                                    gt_x2, gt_y2 = gt_x1 + w, gt_y1 + h
                                    xA = max(best_pred_box[0], gt_x1)
                                    yA = max(best_pred_box[1], gt_y1)
                                    xB = min(best_pred_box[2], gt_x2)
                                    yB = min(best_pred_box[3], gt_y2)
                                    
                                    interArea = max(0, xB - xA) * max(0, yB - yA)
                                    boxAArea = (best_pred_box[2] - best_pred_box[0]) * (best_pred_box[3] - best_pred_box[1])
                                    boxBArea = w * h
                                    iou = interArea / float(boxAArea + boxBArea - interArea)
                                    if iou > best_iou: best_iou = iou
                            
                            if best_iou >= 0.5:
                                acc_score = 1.0
                                acc_info = f"| mAP@0.5: ✅ (IoU: {best_iou:.2f})"
                            else:
                                acc_info = f"| mAP@0.5: ❌ (Best IoU: {best_iou:.2f})"
                        else:
                            acc_info = "| mAP@0.5: ❌ (No detections/GT)"

            accuracy_metrics.append(acc_score)

            # --- F. STORE RAW DATA ---
            current_preprocessing_ms = get_duration_ms(preprocessing_start_time_ns, preprocessing_end_time_ns)
            current_inference_ms = get_duration_ms(inference_start_time_ns, inference_end_time_ns)
            current_postprocessing_ms = get_duration_ms(postprocessing_start_time_ns, postprocessing_end_time_ns)
            current_system_ms = get_duration_ms(system_start_time_ns, system_end_time_ns)

            preprocessing_latencies_ms.append(current_preprocessing_ms)
            inference_latencies_ms.append(current_inference_ms)
            postprocessing_latencies_ms.append(current_postprocessing_ms)
            total_system_latencies_ms.append(current_system_ms)
            if confidence_score > 0:
                confidence_scores_list.append(confidence_score)

            print(f"   [{i+1}/{len(files_to_process)}] {file_name} "
                  f"| Inf: {current_inference_ms:.1f}ms "
                  f"{acc_info}")

            detailed_sample_records.append({
                "filename": file_name,
                "preprocessing_ms": round(current_preprocessing_ms, 4),
                "inference_ms": round(current_inference_ms, 4),
                "postprocessing_ms": round(current_postprocessing_ms, 4),
                "e2e_ms": round(current_system_ms, 4),
                "class_id": class_id,
                "confidence": round(confidence_score, 2),
                "accuracy": round(acc_score, 4),
                "info": str(detected_info)
            })

    finally:
        # Capture Stop Time
        t_stop_dt, t_stop_iso = get_utc_timestamps()

        # --- 8. RESOURCE MONITORING (STOP) ---
        if config['monitor_resources']:
            if config['torch_device'].type == "cuda":
                torch.cuda.synchronize()
            time.sleep(0.2) 

            stop_docker_monitor(config['docker_endpoint'])
            stop_system_monitor(config['system_endpoint'])

            if is_vaccel_remote_run:
                stop_docker_monitor(config['remote_docker_endpoint'])
                stop_system_monitor(config['remote_system_endpoint'])

    # 9. FINAL CALCULATIONS & EXPORT
    if not inference_latencies_ms:
        print("❌ No successful inferences recorded.")
        return

    stats_preprocessing = calculate_stats(preprocessing_latencies_ms)
    stats_inference = calculate_stats(inference_latencies_ms)
    stats_postprocessing = calculate_stats(postprocessing_latencies_ms)
    stats_system = calculate_stats(total_system_latencies_ms)
    stats_confidence = calculate_stats(confidence_scores_list)

    avg_inf, avg_sys = stats_inference["mean"], stats_system["mean"]
    inference_fps = (1000.0 / avg_inf) if avg_inf > 0 else 0
    system_fps = (1000.0 / avg_sys) if avg_sys > 0 else 0

    system_fps_list = [(1000.0 / t) for t in total_system_latencies_ms if t > 0]
    inference_fps_list = [(1000.0 / t) for t in inference_latencies_ms if t > 0]

    sys_fps_std = float(np.std(system_fps_list)) if system_fps_list else 0.0
    inf_fps_std = float(np.std(inference_fps_list)) if inference_fps_list else 0.0

    avg_acc = np.mean(accuracy_metrics) if accuracy_metrics else 0
    metric_label = "mIoU" if config['model_type'] == "semantic_segmentation" else ("mAP@0.5" if config['model_type'] == "object_detection" else "Top-1 Acc")

    print(f"\n📊 BENCHMARK SUMMARY ({config['core_model_name']})")
    print(f"   ---------------------------------------------")
    print(f"   Avg Preprocessing:  {stats_preprocessing['mean']:.2f} ms")
    print(f"   Avg Inference:      {stats_inference['mean']:.2f} ms (P90: {stats_inference['p90']:.2f})")
    print(f"   Avg Postprocessing: {stats_postprocessing['mean']:.2f} ms")
    print(f"   Avg System E2E:     {stats_system['mean']:.2f} ms")
    print(f"   ---------------------------------------------")
    print(f"   Inference FPS:      {inference_fps:.2f}")
    print(f"   System FPS:         {system_fps:.2f}")
    print(f"   {metric_label}:         {avg_acc:.2%}")
    print(f"   ---------------------------------------------")

    if config['export_results']:
        # Export JSON
        json_output_path = run_dir / "benchmark_summary.json"
        final_output_data = {
            "run_id": run_id,
            "backend": config['backend'],
            "host": config['host'],
            "model": config['core_model_name'],
            "model_type": config['model_type'],
            "device": config['target_device'],
            "num_samples": len(inference_latencies_ms),
            "time_window": {
                "start": t_start_iso,
                "stop": t_stop_iso,
                "duration_sec": (t_stop_dt - t_start_dt).total_seconds()
            },
            "frames_per_sample": frames_per_sample,
            "fps": {
                "inference": round(inference_fps, 2),
                "inference_std": round(inf_fps_std, 2),
                "system": round(system_fps, 2),
                "system_std": round(sys_fps_std, 2)
            },
            "accuracy": {
                "metric": metric_label,
                "score": round(avg_acc, 4)
            },
            "preprocessing_ms": stats_preprocessing,
            "inference_ms": stats_inference,
            "postprocessing_ms": stats_postprocessing,
            "system_ms": stats_system,
            "confidence_score": stats_confidence
        }
        
        if "sol" in config['backend']:
            final_output_data["sol_run_mode"] = int(os.environ.get("SOL_RUN_MODE", "2").strip())

        with open(json_output_path, 'w') as json_file:
            json.dump(final_output_data, json_file, indent=4)
        print(f"   ✅ JSON Summary saved to {json_output_path}")

        # Export CSV
        csv_output_path = run_dir / "benchmark_data.csv"
        if detailed_sample_records:
            with open(csv_output_path, 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=detailed_sample_records[0].keys())
                writer.writeheader()
                writer.writerows(detailed_sample_records)
            print(f"   ✅ CSV Data saved to {csv_output_path}")

if __name__ == "__main__":
    main()