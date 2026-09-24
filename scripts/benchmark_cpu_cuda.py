"""Compare the configured Ultralytics detector on one image; never send alerts."""
import argparse
import gc
import json
import statistics
import time
import warnings
from pathlib import Path

import cv2
import torch
import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--image', default='/dev/shm/preview.jpg')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--output', default='cpu-cuda-results.json')
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 1 or args.threads < 1:
        parser.error('runs, warmup and threads must be positive')
    warnings.simplefilter('default')  # Keep CUDA compatibility warnings visible.
    config = yaml.safe_load(Path(args.config).read_text())
    candidates = [(key, cfg) for key, cfg in config['models'].items()
                  if cfg.get('enabled', True) and cfg.get('type') == 'ultralytics']
    if len(candidates) != 1:
        parser.error('Expected exactly one enabled Ultralytics model')
    key, cfg = candidates[0]
    frame = cv2.imread(args.image)
    if frame is None:
        parser.error('Could not read benchmark image')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing to substitute CPU')
    torch.set_num_threads(args.threads)
    report = dict(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0),
                  compiled_architectures=torch.cuda.get_arch_list(),
                  model_key=key, weights=cfg['name'], image_shape=list(frame.shape),
                  threads=args.threads, runs=args.runs, warmup=args.warmup,
                  note='Production service remains running; timings include contention.', results={})
    print(json.dumps(report, indent=2), flush=True)
    for device in ('cpu', 'cuda:0'):
        model = YOLO(cfg['name'])
        model.to(device)
        # Warm up first: Ultralytics may fuse layers and initialize its predictor.
        def predict():
            return model.predict(frame, device=device, verbose=False,
                                 conf=cfg.get('confidence_threshold', .6))
        for _ in range(args.warmup):
            prediction = predict()
        parameter_device = str(next(model.model.parameters()).device)
        observed_inputs = set()
        def observe(module, inputs):
            observed_inputs.update(str(x.device) for x in inputs if isinstance(x, torch.Tensor))
        hook = model.model.register_forward_pre_hook(observe)
        times = []
        try:
            for _ in range(args.runs):
                if device.startswith('cuda'):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                prediction = predict()
                if device.startswith('cuda'):
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
        finally:
            hook.remove()
        output_device = str(prediction[0].boxes.data.device)
        assert parameter_device == device and output_device == device, 'Unexpected model/output device'
        assert observed_inputs == {device}, f'Unexpected forward input devices: {observed_inputs}'
        result = dict(median_ms=round(statistics.median(times), 2),
                      samples_ms=[round(t, 2) for t in times],
                      parameter_device=parameter_device, forward_input_devices=sorted(observed_inputs),
                      output_device=output_device, boxes=len(prediction[0].boxes))
        report['results'][device] = result
        print(device, json.dumps(result), flush=True)
        del model, prediction
        gc.collect()
        torch.cuda.empty_cache()
    report['cuda_speedup'] = round(report['results']['cpu']['median_ms'] /
                                   report['results']['cuda:0']['median_ms'], 2)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print('CUDA speedup:', report['cuda_speedup'], flush=True)


if __name__ == '__main__':
    main()
