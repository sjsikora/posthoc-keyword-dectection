#!/bin/bash
set -e

DATA_DIR="grader/data"

echo "=== Cleaning old data ==="
rm -f "$DATA_DIR"/transcriptions_*.jsonl
rm -f "$DATA_DIR"/thresholds.json
rm -f "$DATA_DIR"/threshold_curves.json
rm -f "$DATA_DIR"/results*.json
rm -f "$DATA_DIR"/phoneme_confusion.json

#echo ""
#echo "=== Step 1: Build confusion matrix (train split) ==="
#uv run python -m grader.build_confusion --samples 100

echo ""
echo "=== Step 2: Transcribe train split (clean) ==="
uv run python -m grader.transcribe --split train --samples 50 --unknown 600

echo ""
echo "=== Step 3: Tune thresholds ==="
uv run python -m grader.tune

echo ""
echo "=== Step 4: Transcribe validation — all 4 conditions in parallel ==="
uv run python -m grader.transcribe --split validation --samples 50 --unknown 600 &
PID_CLEAN=$!

uv run python -m grader.transcribe --split validation --samples 50 --unknown 600 --snr 20 &
PID_SNR20=$!

uv run python -m grader.transcribe --split validation --samples 50 --unknown 600 --snr 10 &
PID_SNR10=$!

uv run python -m grader.transcribe --split validation --samples 50 --unknown 600 --snr 0 &
PID_SNR0=$!

echo "  Waiting for all transcription jobs..."
wait $PID_CLEAN  && echo "  [done] clean"
wait $PID_SNR20  && echo "  [done] snr20"
wait $PID_SNR10  && echo "  [done] snr10"
wait $PID_SNR0   && echo "  [done] snr0"

echo ""
echo "=== Step 5: Evaluate all conditions ==="
uv run python -m grader.evaluate &
uv run python -m grader.evaluate \
    --transcriptions "$DATA_DIR/transcriptions_validation_snr20.jsonl" \
    --results        "$DATA_DIR/results_snr20.json" &
uv run python -m grader.evaluate \
    --transcriptions "$DATA_DIR/transcriptions_validation_snr10.jsonl" \
    --results        "$DATA_DIR/results_snr10.json" &
uv run python -m grader.evaluate \
    --transcriptions "$DATA_DIR/transcriptions_validation_snr0.jsonl" \
    --results        "$DATA_DIR/results_snr0.json" &

wait
echo ""
echo "=== All done ==="
