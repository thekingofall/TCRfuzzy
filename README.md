

![image](https://github.com/user-attachments/assets/d965eecb-cce9-4001-acd8-6e5eacaf4e19)

# TCRfuzzy - TCR序列相似度比较工具

TCRfuzzy 是一个用于比较 TCR（T细胞受体）序列相似度的工具集，包含两个版本的实现：

- `tcr_fuzzy.py`: 基础版本，使用多线程在内存中处理
- `tcr_fuzzy_batch.py`: 批处理版本，使用进程池处理大规模数据

## 功能特点

### 共同特点
- 支持多种相似度计算方法
- 自动安装依赖包
- 详细的日志记录
- 进度显示
- 支持阈值过滤
- 内存优化

### tcr_fuzzy.py 特点
- 单进程多线程处理
- 适合中小规模数据
- 内存中处理，速度较快

### tcr_fuzzy_batch.py 特点
- 多进程并行处理
- 支持大规模数据
- 批量处理，内存占用小
- 支持断点续传
- 更安全的文件处理

## 安装

### 系统要求
- Python 3.6+
- pip (Python包管理器)

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/TCRfuzzy.git
cd TCRfuzzy
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

或直接运行脚本，它会自动安装所需依赖。

### 可选依赖（批处理版本）
- GNU parallel
- xargs (通常预装在Unix系统中)

## 使用方法

### 基础版本 (tcr_fuzzy.py)

基本用法：
```bash
python tcr_fuzzy.py input.csv
```

完整参数：
```bash
python tcr_fuzzy.py [-h] [-c1 COLUMN1] [-c2 COLUMN2] [-o OUTPUT]
                    [-d DELIMITER] [--encoding ENCODING]
                    [--skip-rows SKIP_ROWS] [--threshold THRESHOLD]
                    csv_file
```

### 批处理版本 (tcr_fuzzy_batch.py)

基本用法：
```bash
python tcr_fuzzy_batch.py input.csv
```

完整参数：
```bash
python tcr_fuzzy_batch.py [-h] [-c1 COLUMN1] [-c2 COLUMN2] [-o OUTPUT]
                         [-d DELIMITER] [--encoding ENCODING]
                         [--skip-rows SKIP_ROWS] [--threshold THRESHOLD]
                         [-p PROCESSES] [-b BATCH_SIZE]
                         csv_file
```

### 参数说明

- `csv_file`: 输入CSV文件路径（必需）
- `-c1, --column1`: 第一个比较列的索引，从0开始（默认：0）
- `-c2, --column2`: 第二个比较列的索引，从0开始（默认：1）
- `-o, --output`: 输出文件路径
- `-d, --delimiter`: CSV文件分隔符（默认：","）
- `--encoding`: 文件编码（默认：utf-8）
- `--skip-rows`: 跳过的起始行数（默认：0）
- `--threshold`: 相似度阈值，0-100（默认：0）

批处理版本额外参数：
- `-p, --processes`: 最大并行进程数（默认：4）
- `-b, --batch-size`: 每批处理的比较对数量（默认：50）

## 使用示例

### 基本比较
```bash
# 基础版本
python tcr_fuzzy.py data.csv

# 批处理版本
python tcr_fuzzy_batch.py data.csv
```

### 指定列和阈值
```bash
# 基础版本
python tcr_fuzzy.py data.csv -c1 0 -c2 1 --threshold 80

# 批处理版本
python tcr_fuzzy_batch.py data.csv -c1 0 -c2 1 --threshold 80
```

### 自定义输出和分隔符
```bash
# 基础版本
python tcr_fuzzy.py data.csv -o results.csv -d ";" --encoding "utf-8"

# 批处理版本
python tcr_fuzzy_batch.py data.csv -o results.csv -d ";" --encoding "utf-8"
```

### 批处理版本特有功能
```bash
# 指定进程数和批次大小
python tcr_fuzzy_batch.py data.csv -p 8 -b 100

# 大文件处理
python tcr_fuzzy_batch.py big_data.csv -p 16 -b 1000
```

## 输出格式

输出的CSV文件包含以下列：
- 字符串1：第一个序列
- 字符串2：第二个序列
- 相似度(ratio)：基本相似度分数
- 部分相似度(partial_ratio)：部分匹配相似度
- 排序标记相似度(token_sort_ratio)：词序重排后的相似度
- 集合标记相似度(token_set_ratio)：集合形式的相似度

## 选择建议

- 对于小型数据集（<10万对比较）：使用基础版本（tcr_fuzzy.py）
- 对于大型数据集（>10万对比较）：使用批处理版本（tcr_fuzzy_batch.py）
- 内存受限系统：使用批处理版本，适当调小批次大小
- 多核心系统：使用批处理版本，增加进程数

## 注意事项

1. 内存使用
   - 基础版本：所有数据都在内存中处理
   - 批处理版本：通过批处理控制内存使用

2. 性能优化
   - 适当的批次大小可以优化性能
   - 进程数建议设置为CPU核心数的1-2倍
   - 对于SSD系统，可以增加批次大小

3. 日志文件
   - 程序运行时会生成日志文件：`TCR_comparison_YYYYMMDD_HHMMSS.log`
   - 记录详细的运行信息和错误报告

