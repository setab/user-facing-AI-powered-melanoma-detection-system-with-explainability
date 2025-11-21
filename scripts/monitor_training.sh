#!/bin/bash
# Monitor training progress with live updates

LOG_FILE=$(ls -t training_full_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "No training log file found!"
    exit 1
fi

echo "=================================================="
echo "🎯 MELANOMA DETECTION - TRAINING MONITOR"
echo "=================================================="
echo "Log file: $LOG_FILE"
echo "Started: $(date)"
echo ""

# Function to extract progress
extract_progress() {
    if [ -f "$LOG_FILE" ]; then
        # Get current architecture
        CURRENT_ARCH=$(grep "Training [A-Z]" "$LOG_FILE" | tail -1 | sed 's/.*Training \([A-Z_0-9]*\).*/\1/')
        
        # Count completed epochs
        EPOCHS_DONE=$(grep -c "✓ Saved checkpoint" "$LOG_FILE")
        
        # Get latest metrics
        LATEST_TRAIN_LOSS=$(grep "Train Loss:" "$LOG_FILE" | tail -1 | sed 's/.*Train Loss: \([0-9.]*\).*/\1/')
        LATEST_TRAIN_ACC=$(grep "Train Acc:" "$LOG_FILE" | tail -1 | sed 's/.*Train Acc: \([0-9.]*\).*/\1/')
        LATEST_VAL_LOSS=$(grep "Val Loss:" "$LOG_FILE" | tail -1 | sed 's/.*Val Loss: \([0-9.]*\).*/\1/')
        LATEST_VAL_ACC=$(grep "Val Acc:" "$LOG_FILE" | tail -1 | sed 's/.*Val Acc: \([0-9.]*\).*/\1/')
        
        echo "$CURRENT_ARCH|$EPOCHS_DONE|$LATEST_TRAIN_LOSS|$LATEST_TRAIN_ACC|$LATEST_VAL_LOSS|$LATEST_VAL_ACC"
    fi
}

# Monitor loop
while true; do
    clear
    echo "=================================================="
    echo "🎯 MELANOMA DETECTION - TRAINING MONITOR"
    echo "=================================================="
    echo "📅 $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Check if training process is running
    if pgrep -f "compare_models.py" > /dev/null; then
        echo "✅ Training Status: RUNNING"
    else
        echo "❌ Training Status: NOT RUNNING"
        echo ""
        echo "Training may have completed or crashed."
        echo "Check log file: $LOG_FILE"
        break
    fi
    
    echo ""
    echo "=================================================="
    echo "📊 TRAINING PROGRESS"
    echo "=================================================="
    
    PROGRESS=$(extract_progress)
    IFS='|' read -r ARCH EPOCHS_DONE TRAIN_LOSS TRAIN_ACC VAL_LOSS VAL_ACC <<< "$PROGRESS"
    
    if [ -n "$ARCH" ]; then
        echo "Current Model: $ARCH"
        echo "Epochs Completed: $EPOCHS_DONE / 80 total (4 models × 20 epochs)"
        
        # Progress bar
        PERCENT=$((EPOCHS_DONE * 100 / 80))
        FILLED=$((PERCENT / 2))
        EMPTY=$((50 - FILLED))
        printf "Progress: ["
        printf "%${FILLED}s" | tr ' ' '█'
        printf "%${EMPTY}s" | tr ' ' '░'
        printf "] %d%%\n" "$PERCENT"
        
        echo ""
        if [ -n "$TRAIN_LOSS" ]; then
            echo "Latest Metrics:"
            echo "  Train Loss: $TRAIN_LOSS | Train Acc: $TRAIN_ACC"
            echo "  Val Loss:   $VAL_LOSS   | Val Acc:   $VAL_ACC"
        fi
    fi
    
    echo ""
    echo "=================================================="
    echo "🖥️  GPU USAGE"
    echo "=================================================="
    
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | while IFS=',' read -r idx name temp gpu_util mem_util mem_used mem_total; do
            echo "GPU $idx: $name"
            echo "  Temperature: ${temp}°C"
            echo "  GPU Utilization: ${gpu_util}%"
            echo "  Memory: ${mem_used}MB / ${mem_total}MB (${mem_util}% used)"
            
            # GPU utilization bar
            GPU_FILLED=$((gpu_util / 2))
            GPU_EMPTY=$((50 - GPU_FILLED))
            printf "  GPU: ["
            printf "%${GPU_FILLED}s" | tr ' ' '█'
            printf "%${GPU_EMPTY}s" | tr ' ' '░'
            printf "] %d%%\n" "$gpu_util"
            
            # Memory bar
            MEM_FILLED=$((mem_util / 2))
            MEM_EMPTY=$((50 - MEM_FILLED))
            printf "  MEM: ["
            printf "%${MEM_FILLED}s" | tr ' ' '█'
            printf "%${MEM_EMPTY}s" | tr ' ' '░'
            printf "] %d%%\n" "$mem_util"
        done
    else
        echo "nvidia-smi not found. GPU monitoring unavailable."
    fi
    
    echo ""
    echo "=================================================="
    echo "📝 RECENT LOG OUTPUT (Last 5 lines)"
    echo "=================================================="
    tail -5 "$LOG_FILE" | sed 's/^/  /'
    
    echo ""
    echo "Press Ctrl+C to exit monitoring"
    echo "=================================================="
    
    sleep 5
done
