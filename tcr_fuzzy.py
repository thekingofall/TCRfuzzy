import sys
import subprocess
import importlib.util
import os
import multiprocessing
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime
import gc
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'string_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

def check_and_install_packages(required_packages: List[str]) -> None:
    """检查并安装所需包"""
    missing_packages = []
    
    for package in required_packages:
        import_name = 'Levenshtein' if package == 'python-Levenshtein' else package
        if importlib.util.find_spec(import_name) is None:
            missing_packages.append(package)
    
    if missing_packages:
        logging.info(f"正在安装缺失的包: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + missing_packages)
            logging.info("所有必要的包已成功安装")
        except Exception as e:
            logging.error(f"安装包时出错: {e}")
            sys.exit(1)

# 安装必要的包
required_packages = ['pandas', 'fuzzywuzzy', 'python-Levenshtein', 'tqdm', 'argparse']
check_and_install_packages(required_packages)

# 导入所需库
import pandas as pd
from fuzzywuzzy import fuzz
import itertools
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import numpy as np

def get_optimal_process_count() -> int:
    """计算最优进程数"""
    cpu_cores = cpu_count()
    MAX_PROCESSES = 8
    
    optimal_processes = min(cpu_cores, MAX_PROCESSES)
    
    if cpu_cores > 8:  # 如果是多核系统，减少使用的核心数
        optimal_processes = optimal_processes // 2
    
    return max(1, min(optimal_processes, 4))  # 确保至少1个进程，最多4个进程

def preprocess_string(s: str) -> str:
    """预处理字符串"""
    if not isinstance(s, str):
        s = str(s)
    return s.strip().lower()

def calculate_similarity(args: Tuple[int, Tuple[str, str]]) -> Optional[Dict[str, Any]]:
    """计算字符串相似度"""
    try:
        index, (str1, str2) = args
        
        # 预处理字符串
        str1_processed = preprocess_string(str1)
        str2_processed = preprocess_string(str2)
        
        # 如果字符串完全相同，直接返回100分
        if str1_processed == str2_processed:
            return {
                'idx': index,
                '字符串1': str1,
                '字符串2': str2,
                '相似度(ratio)': 100,
                '部分相似度(partial_ratio)': 100,
                '排序标记相似度(token_sort_ratio)': 100,
                '集合标记相似度(token_set_ratio)': 100
            }
        
        # 空字符串处理
        if not str1_processed or not str2_processed:
            return {
                'idx': index,
                '字符串1': str1,
                '字符串2': str2,
                '相似度(ratio)': 0,
                '部分相似度(partial_ratio)': 0,
                '排序标记相似度(token_sort_ratio)': 0,
                '集合标记相似度(token_set_ratio)': 0
            }
        
        # 计算各种相似度
        ratio = fuzz.ratio(str1_processed, str2_processed)
        partial_ratio = fuzz.partial_ratio(str1_processed, str2_processed)
        token_sort_ratio = fuzz.token_sort_ratio(str1_processed, str2_processed)
        token_set_ratio = fuzz.token_set_ratio(str1_processed, str2_processed)
        
        return {
            'idx': index,
            '字符串1': str1,
            '字符串2': str2,
            '相似度(ratio)': ratio,
            '部分相似度(partial_ratio)': partial_ratio,
            '排序标记相似度(token_sort_ratio)': token_sort_ratio,
            '集合标记相似度(token_set_ratio)': token_set_ratio
        }
    except Exception as e:
        logging.error(f"计算相似度时出错 (索引 {index}): {e}")
        return None

def process_batch(batch_data: List[Tuple[int, Tuple[str, str]]], 
                 pool: Pool, 
                 threshold: int) -> List[Dict[str, Any]]:
    """处理一批数据"""
    try:
        # 使用更小的chunksize来避免内存问题
        batch_results = list(tqdm(
            pool.imap_unordered(calculate_similarity, batch_data, chunksize=10),
            total=len(batch_data),
            desc="处理进度",
            unit="对"
        ))
        
        # 立即清理无效结果
        valid_results = [r for r in batch_results if r is not None]
        del batch_results
        
        if threshold > 0:
            valid_results = [r for r in valid_results if 
                           max(r['相似度(ratio)'],
                               r['部分相似度(partial_ratio)'],
                               r['排序标记相似度(token_sort_ratio)'],
                               r['集合标记相似度(token_set_ratio)']) >= threshold]
        
        return valid_results
    except Exception as e:
        logging.error(f"批处理时出错: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(
        description="FuzzyWuzzy CSV并行比较工具 - 计算CSV文件中两列之间的字符串相似度",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("csv_file", help="要处理的CSV文件路径")
    parser.add_argument("-c1", "--column1", type=int, default=0, 
                        help="第一个要比较的列索引（从0开始）")
    parser.add_argument("-c2", "--column2", type=int, default=1, 
                        help="第二个要比较的列索引（从0开始）")
    parser.add_argument("-o", "--output", help="输出CSV文件的路径")
    parser.add_argument("-d", "--delimiter", default=",", help="CSV文件的分隔符")
    parser.add_argument("--encoding", default="utf-8", help="CSV文件的编码")
    parser.add_argument("--skip-rows", type=int, default=0, help="读取CSV时跳过的行数")
    parser.add_argument("-p", "--processes", type=int, default=0,
                        help="并行处理的进程数（默认为自动优化）")
    parser.add_argument("-b", "--batch-size", type=int, default=50,
                        help="每批处理的对比对数")
    parser.add_argument("--threshold", type=int, default=0,
                        help="仅保存相似度分数高于此阈值的结果（0-100）")
    
    args = parser.parse_args()
    
    # 验证参数
    if not os.path.exists(args.csv_file):
        logging.error(f"文件不存在: {args.csv_file}")
        sys.exit(1)
    
    if not 0 <= args.threshold <= 100:
        logging.error("阈值必须在0到100之间")
        sys.exit(1)
    
    # 设置最优进程数
    processes = args.processes if args.processes > 0 else get_optimal_process_count()
    logging.info(f"将使用 {processes} 个进程进行并行处理")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(args.csv_file, delimiter=args.delimiter, 
                        encoding=args.encoding, skiprows=args.skip_rows)
        
        # 验证列索引
        if len(df.columns) <= max(args.column1, args.column2):
            logging.error(f"CSV文件只有{len(df.columns)}列，但指定了列{max(args.column1, args.column2)}")
            sys.exit(1)
        
        # 获取列数据并预处理
        col1 = df.iloc[:, args.column1].astype(str).tolist()
        col2 = df.iloc[:, args.column2].astype(str).tolist()
        
        # 清理DataFrame
        del df
        gc.collect()
        
        # 生成对比对
        pairs = list(itertools.product(col1, col2))
        total_comparisons = len(pairs)
        
        if total_comparisons == 0:
            logging.error("没有可比较的对")
            sys.exit(0)
            
        logging.info(f"将比较 {len(col1)} x {len(col2)} = {total_comparisons} 个组合")
        
        # 调整批处理大小
        batch_size = min(args.batch_size, max(50, total_comparisons // (processes * 8)))
        num_batches = (total_comparisons + batch_size - 1) // batch_size
        
        all_results = []
        
        # 使用进程池进行并行处理
        with Pool(processes=processes) as pool:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_comparisons)
                
                batch_pairs = [(i, pairs[i]) for i in range(start_idx, end_idx)]
                
                try:
                    batch_results = process_batch(batch_pairs, pool, args.threshold)
                    all_results.extend(batch_results)
                    logging.info(f"批次 {batch_idx + 1}/{num_batches} 完成")
                except Exception as e:
                    logging.error(f"处理批次 {batch_idx + 1} 时出错: {e}")
                    continue
                
                # 清理内存
                del batch_pairs
                gc.collect()
        
        # 清理pairs
        del pairs
        gc.collect()
        
        # 处理结果
        if not all_results:
            logging.warning("没有找到满足条件的结果")
            sys.exit(0)
        
        # 排序并移除索引
        all_results.sort(key=lambda x: x['idx'])
        for result in all_results:
            del result['idx']
        
        # 创建结果DataFrame并保存
        try:
            results_df = pd.DataFrame(all_results)
            output_path = args.output or f"{os.path.splitext(args.csv_file)[0]}_similarity_results.csv"
            results_df.to_csv(output_path, index=False)
            logging.info(f"处理完成！共计算 {total_comparisons} 对，保存了 {len(results_df)} 条结果")
            logging.info(f"结果已保存到: {output_path}")
        except Exception as e:
            logging.error(f"保存结果时出错: {e}")
            sys.exit(1)
        
    except Exception as e:
        logging.error(f"处理过程中出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # 设置更保守的线程数
    os.environ["NUMEXPR_MAX_THREADS"] = "4"
    # 设置进程启动方法
    multiprocessing.set_start_method('spawn', force=True)
    main()
