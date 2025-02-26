# TCRfuzzy

一个专为TCR序列设计的高效并行模糊字符串匹配工具，用于计算和分析T细胞受体序列相似度。

## 简介

TCRfuzzy利用FuzzyWuzzy库和Levenshtein距离算法，专门用于比较TCR（T细胞受体）序列的相似度。该工具能够自动安装依赖，并利用多核CPU进行并行处理，显著提高大量TCR序列数据的处理速度。

## 特性

- **TCR序列比较**：专为比较TCR序列对设计
- **自动依赖管理**：自动检测并安装必要的Python库
- **多核并行处理**：利用所有可用CPU核心加速计算
- **批量处理**：支持大型TCR数据集的内存高效处理
- **多种相似度指标**：计算四种不同的相似度分数，适合不同类型的TCR序列分析
- **灵活的参数配置**：支持自定义列索引、分隔符、编码等
- **进度可视化**：实时显示处理进度
- **相似度过滤**：可设置阈值只保留高相似度的TCR序列匹配结果

## 安装

只需克隆此仓库：

```bash
git clone https://github.com/thekingofall/TCRfuzzy.git
cd TCRfuzzy
```

需要Python 3.6或更高版本。脚本会自动安装所需的依赖包。

## 使用方法

### 基本用法

```bash
python tcr_fuzzy.py your_tcr_file.csv
```

这将比较CSV文件中第一列和第二列的所有TCR序列可能组合，并将结果保存到同目录下的`your_tcr_file_similarity_results.csv`。

### 查看帮助

```bash
python tcr_fuzzy.py --help
```

### 参数说明

| 参数 | 短格式 | 说明 | 默认值 |
|------|------|------|------|
| `csv_file` | - | 包含TCR序列的CSV文件路径 | (必需) |
| `--column1` | `-c1` | 第一个TCR序列列的索引（从0开始） | 0 |
| `--column2` | `-c2` | 第二个TCR序列列的索引（从0开始） | 1 |
| `--output` | `-o` | 输出CSV文件的路径 | 原文件名_similarity_results.csv |
| `--delimiter` | `-d` | CSV文件的分隔符 | , |
| `--encoding` | - | CSV文件的编码 | utf-8 |
| `--skip-rows` | - | 读取CSV时跳过的行数 | 0 |
| `--processes` | `-p` | 并行处理的进程数 | 可用CPU核心数 |
| `--batch-size` | `-b` | 每批处理的TCR序列对数 | 10000 |
| `--threshold` | - | 仅保存相似度高于此阈值的结果 | 0 |

### 示例

比较文件中特定列的TCR序列，使用4个进程，并只保存相似度大于70的结果：
```bash
python tcr_fuzzy.py tcr_data.csv -c1 3 -c2 4 -p 4 --threshold 70
```

处理分号分隔的TCR数据文件，指定输出路径：
```bash
python tcr_fuzzy.py tcr_data.csv -d ";" -o tcr_similarity_results.csv
```

## 输出说明

输出的CSV文件包含以下列：

- `TCR序列1`: 第一列的TCR序列
- `TCR序列2`: 第二列的TCR序列
- `相似度(ratio)`: 基本序列相似度 (0-100)
- `部分相似度(partial_ratio)`: 部分序列匹配相似度，适合CDR3区分析 (0-100)
- `排序标记相似度(token_sort_ratio)`: 标记化并排序后的相似度，适合氨基酸顺序变异分析 (0-100)
- `集合标记相似度(token_set_ratio)`: 标记化并作为集合比较的相似度，适合氨基酸组成分析 (0-100)

## 应用场景

- **TCR多样性分析**：评估T细胞库的克隆多样性
- **CDR3区比较**：分析抗原特异性T细胞的CDR3区相似度
- **TCR克隆型分类**：基于序列相似度对TCR克隆型进行分类
- **序列聚类**：对大量TCR序列进行聚类分析
- **交叉反应性预测**：预测不同TCR与抗原的潜在交叉反应性

## 性能优化

- 对于非常大的TCR数据集，建议增加批处理大小 (`--batch-size`)
- 对于内存有限的系统，可以减小批处理大小
- 设置适当的阈值 (`--threshold`) 可以减少输出文件大小
- 进程数 (`--processes`) 设置为CPU核心数通常是最优的，但IO密集型系统可能需要调整

## 依赖库

- pandas: 数据处理
- fuzzywuzzy: 模糊字符串匹配
- python-Levenshtein: 加速Levenshtein距离计算
- tqdm: 进度条显示
- argparse: 命令行参数处理





