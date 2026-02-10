#!/bin/bash
# 查看指定目录下各子文件夹占用空间，按从大到小排序
# 用法: ./check_data_usage.sh [目录]，默认 /data

DATA_DIR="${1:-/data}"

if [[ ! -d "$DATA_DIR" ]]; then
    echo "错误: 目录不存在: $DATA_DIR"
    exit 1
fi

echo "目录占用统计: $DATA_DIR (按大小降序)"
echo "----------------------------------------"

# --max-depth=1 只统计直接子目录；-h 人类可读；2>/dev/null 忽略无权限等错误
du -h --max-depth=1 "$DATA_DIR" 2>/dev/null | sort -hr

echo "----------------------------------------"
echo "总计:"
du -sh "$DATA_DIR" 2>/dev/null
