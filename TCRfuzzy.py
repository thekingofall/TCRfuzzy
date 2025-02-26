import sys
import subprocess
import importlib.util
import os
import multiprocessing

# 检查和安装必要的库
required_packages = ['pandas', 'fuzzywuzzy', 'python-Levenshtein', 'tqdm', 'argparse']
missing_packages = []

for package in required_packages:
    # 特殊处理python-Levenshtein，因为它的导入名与包名不同
    if package == 'python-Levenshtein':
        import_name = 'Levenshtein'
    else:
        import_name = package
    
    # 检查是否已安装
    if importlib.util.find_spec(import_name) is None:
        missing_packages.append(package)

# 如果有缺失的包，安装它们
if missing_packages:
    print(f"正在安装缺失的包: {', '.join(missing_packages)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("所有必要的包已成功安装")
    except Exception as e:
        print(f"安装包时出错: {e}")
        sys.exit(1)

# 现在可以安全地导入所需库
import pandas as pd
from fuzzywuzzy import fuzz
import itertools
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import numpy as np

# 定义计算相似度的函数（用于并行处理）
def calculate_similarity(pair_with_index):
    index, (str1, str2) = pair_with_index
    ratio = fuzz.ratio(str1, str2)
    partial_ratio = fuzz.partial_ratio(str1, str2)
    token_sort_ratio = fuzz.token_sort_ratio(str1, str2)
    token_set_ratio = fuzz.token_set_ratio(str1, str2)
    
    return {
        'idx': index,
        '字符串1': str1,
        '字符串2': str2,
        '相似度(ratio)': ratio,
        '部分相似度(partial_ratio)': partial_ratio,
        '排序标记相似度(token_sort_ratio)': token_sort_ratio,
        '集合标记相似度(token_set_ratio)': token_set_ratio
    }

def main():
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(
        description="FuzzyWuzzy CSV并行比较工具 - 计算CSV文件中两列之间的字符串相似度",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 添加参数
    parser.add_argument("csv_file", help="要处理的CSV文件路径")
    parser.add_argument("-c1", "--column1", type=int, default=0, 
                        help="第一个要比较的列索引（从0开始）")
    parser.add_argument("-c2", "--column2", type=int, default=1, 
                        help="第二个要比较的列索引（从0开始）")
    parser.add_argument("-o", "--output", 
                        help="输出CSV文件的路径（默认为原文件名加_similarity_results后缀）")
    parser.add_argument("-d", "--delimiter", default=",", 
                        help="CSV文件的分隔符")
    parser.add_argument("--encoding", default="utf-8", 
                        help="CSV文件的编码")
    parser.add_argument("--skip-rows", type=int, default=0, 
                        help="读取CSV时跳过的行数")
    parser.add_argument("-p", "--processes", type=int, default=0,
                        help="并行处理的进程数（默认为CPU核心数）")
    parser.add_argument("-b", "--batch-size", type=int, default=10000,
                        help="每批处理的对比对数（用于大型文件的内存管理）")
    parser.add_argument("--threshold", type=int, default=0,
                        help="仅保存相似度分数高于此阈值的结果（0-100）")
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置进程数
    processes = args.processes if args.processes > 0 else cpu_count()
    print(f"将使用 {processes} 个进程进行并行处理")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(args.csv_file, delimiter=args.delimiter, 
                         encoding=args.encoding, skiprows=args.skip_rows)
        print(f"成功读取CSV文件: {args.csv_file}")
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        sys.exit(1)
    
    # 确保指定的列存在
    if len(df.columns) <= max(args.column1, args.column2):
        print(f"错误：CSV文件只有{len(df.columns)}列，但指定了列{max(args.column1, args.column2)}")
        sys.exit(1)
    
    # 获取指定列数据
    col1 = df.iloc[:, args.column1].astype(str).tolist()
    col2 = df.iloc[:, args.column2].astype(str).tolist()
    
    # 计算所有可能的配对组合
    pairs = list(itertools.product(col1, col2))
    total_comparisons = len(pairs)
    
    if total_comparisons == 0:
        print("没有可比较的对，请检查CSV文件和列索引")
        sys.exit(0)
        
    print(f"将比较 {len(col1)} 个第一列项目和 {len(col2)} 个第二列项目，共 {total_comparisons} 个组合")
    
    # 准备批处理
    batch_size = min(args.batch_size, total_comparisons)
    num_batches = (total_comparisons + batch_size - 1) // batch_size
    
    # 创建结果列表
    all_results = []
    
    # 使用multiprocessing.Pool进行并行处理
    with Pool(processes=processes) as pool:
        # 批量处理以管理内存
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_comparisons)
            
            # 为每个对创建索引
            batch_pairs = [(i, pairs[i]) for i in range(start_idx, end_idx)]
            
            # 创建一个tqdm进度条来跟踪这个批次
            batch_results = list(tqdm(
                pool.imap(calculate_similarity, batch_pairs),
                total=end_idx - start_idx,
                desc=f"批次 {batch_idx+1}/{num_batches}",
                unit="对"
            ))
            
            # 如果设置了阈值，过滤结果
            if args.threshold > 0:
                batch_results = [r for r in batch_results if 
                                max(r['相似度(ratio)'], 
                                    r['部分相似度(partial_ratio)'], 
                                    r['排序标记相似度(token_sort_ratio)'], 
                                    r['集合标记相似度(token_set_ratio)']) >= args.threshold]
            
            all_results.extend(batch_results)
            
            # 显示当前进度
            print(f"已完成 {end_idx}/{total_comparisons} 对比较 ({end_idx/total_comparisons*100:.1f}%)")
    
    # 根据原始索引排序结果
    all_results.sort(key=lambda x: x['idx'])
    
    # 移除idx字段
    for result in all_results:
        del result['idx']
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 确定输出文件路径
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.csv_file)[0]
        output_path = f"{base_name}_similarity_results.csv"
    
    # 保存结果到新的CSV文件
    results_df.to_csv(output_path, index=False)
    print(f"\n处理完成！共计算 {total_comparisons} 对，保存了 {len(results_df)} 条结果")
    print(f"结果已保存到: {output_path}")

if __name__ == "__main__":
    main()
