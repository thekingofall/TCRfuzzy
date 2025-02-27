import sys
import subprocess
import importlib.util
import os
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime
import gc
import warnings
import argparse
import multiprocessing as mp
import csv
import signal
import time
warnings.filterwarnings('ignore')

def init_worker():
    """初始化工作进程，忽略中断信号"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

class TCRfuzzy:
    def __init__(self):
        """初始化TCRfuzzy类"""
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'TCR_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        
        # 设置必要的包
        self.required_packages = ['pandas', 'fuzzywuzzy', 'python-Levenshtein', 'tqdm', 'argparse', 'psutil']
        self.check_and_install_packages()
        
        # 导入所需库（仅在主进程中导入）
        import pandas as pd
        from tqdm import tqdm
        import psutil
        
        self.pd = pd
        self.tqdm = tqdm
        self.psutil = psutil

    def check_and_install_packages(self) -> None:
        """检查并安装所需包"""
        missing_packages = []
        
        for package in self.required_packages:
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

    @staticmethod
    def preprocess_string(s: str) -> str:
        """预处理字符串"""
        if not isinstance(s, str):
            s = str(s)
        return s.strip().lower()

    @staticmethod
    def calculate_similarity_worker(task_data):
        """工作进程的静态方法，计算相似度"""
        try:
            from fuzzywuzzy import fuzz
            
            task_id, str1, str2, threshold = task_data
            
            # 预处理字符串
            str1_processed = TCRfuzzy.preprocess_string(str1)
            str2_processed = TCRfuzzy.preprocess_string(str2)
            
            # 如果字符串完全相同，直接返回100分
            if str1_processed == str2_processed:
                result = {
                    '字符串1': str1,
                    '字符串2': str2,
                    '相似度(ratio)': 100,
                    '部分相似度(partial_ratio)': 100,
                    '排序标记相似度(token_sort_ratio)': 100,
                    '集合标记相似度(token_set_ratio)': 100
                }
                return task_id, result
            
            # 空字符串处理
            if not str1_processed or not str2_processed:
                result = {
                    '字符串1': str1,
                    '字符串2': str2,
                    '相似度(ratio)': 0,
                    '部分相似度(partial_ratio)': 0,
                    '排序标记相似度(token_sort_ratio)': 0,
                    '集合标记相似度(token_set_ratio)': 0
                }
                return task_id, result
            
            # 计算各种相似度
            ratio = fuzz.ratio(str1_processed, str2_processed)
            partial_ratio = fuzz.partial_ratio(str1_processed, str2_processed)
            token_sort_ratio = fuzz.token_sort_ratio(str1_processed, str2_processed)
            token_set_ratio = fuzz.token_set_ratio(str1_processed, str2_processed)
            
            # 检查是否满足阈值
            if threshold > 0 and max(ratio, partial_ratio, token_sort_ratio, token_set_ratio) < threshold:
                return task_id, None
            
            result = {
                '字符串1': str1,
                '字符串2': str2,
                '相似度(ratio)': ratio,
                '部分相似度(partial_ratio)': partial_ratio,
                '排序标记相似度(token_sort_ratio)': token_sort_ratio,
                '集合标记相似度(token_set_ratio)': token_set_ratio
            }
            return task_id, result
            
        except Exception as e:
            return task_id, f"计算相似度时出错: {e}"

    def check_memory_usage(self, threshold_percent=80):
        """检查内存使用情况，如果超过阈值返回True"""
        memory_info = self.psutil.virtual_memory()
        return memory_info.percent > threshold_percent

    def save_batch_results(self, results, output_path, is_first_batch):
        """保存一批结果到CSV文件"""
        mode = 'w' if is_first_batch else 'a'
        header = is_first_batch
        
        if not results:
            return

        # 直接使用csv模块写入，更加内存高效
        try:
            with open(output_path, mode, newline='', encoding='utf-8') as f:
                if results and isinstance(results[0], dict):
                    # 确保我们有字段名
                    fieldnames = list(results[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    
                    if header:
                        writer.writeheader()
                    
                    for result in results:
                        writer.writerow(result)
            return True
        except Exception as e:
            logging.error(f"保存结果时出错: {e}")
            return False

    def compare_tcrs(self, csv_file: str, column1: int = 0, column2: int = 1,
                    output: str = None, delimiter: str = ",", encoding: str = "utf-8",
                    skip_rows: int = 0, threshold: int = 0, num_processes: int = None,
                    batch_size: int = 1000) -> None:
        """比较TCR序列"""
        try:
            # 设置进程数，默认为可用CPU的一半（避免系统过载）
            if num_processes is None:
                num_processes = max(1, os.cpu_count() // 2)
            
            logging.info(f"将使用 {num_processes} 个进程进行处理")
            
            # 读取CSV文件数据
            logging.info(f"正在读取CSV文件: {csv_file}")
            df = self.pd.read_csv(csv_file, delimiter=delimiter, 
                                encoding=encoding, skiprows=skip_rows)
            
            # 验证列索引
            if len(df.columns) <= max(column1, column2):
                logging.error(f"CSV文件只有{len(df.columns)}列，但指定了列{max(column1, column2)}")
                return
            
            # 获取列数据并去除重复和空值
            col1 = df.iloc[:, column1].dropna().drop_duplicates().astype(str).tolist()
            col2 = df.iloc[:, column2].dropna().drop_duplicates().astype(str).tolist()
            
            # 清理DataFrame
            del df
            gc.collect()
            
            # 计算总对比数
            len1 = len(col1)
            len2 = len(col2)
            if column1 == column2:
                total_comparisons = (len1 * (len1 - 1)) // 2
            else:
                total_comparisons = len1 * len2
                
            logging.info(f"将比较第{column1}列({len1}个序列) x 第{column2}列({len2}个序列) = {total_comparisons} 个组合")
            
            # 设置输出路径
            output_path = output or f"{os.path.splitext(csv_file)[0]}_TCR_similarity_results.csv"
            
            # 准备任务列表（分批以减少内存使用）
            all_tasks = []
            task_id = 0
            
            if column1 == column2:
                # 如果是同一列的比较，只比较上三角矩阵
                for i, str1 in enumerate(col1):
                    for j, str2 in enumerate(col2[i+1:], start=i+1):
                        all_tasks.append((task_id, str1, str2, threshold))
                        task_id += 1
            else:
                # 不同列的比较，进行全比较
                for i, str1 in enumerate(col1):
                    for j, str2 in enumerate(col2):
                        all_tasks.append((task_id, str1, str2, threshold))
                        task_id += 1
            
            # 分批处理任务
            num_batches = (len(all_tasks) + batch_size - 1) // batch_size
            logging.info(f"任务将分为 {num_batches} 批进行处理，每批 {batch_size} 个")
            
            # 创建进度条
            with self.tqdm(total=total_comparisons, desc="处理进度", unit="对") as pbar:
                is_first_batch = True
                processed_tasks = 0
                
                # 分批处理任务
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(all_tasks))
                    current_batch = all_tasks[start_idx:end_idx]
                    
                    batch_results = []
                    
                    # 使用进程池处理当前批次
                    with mp.Pool(processes=num_processes, initializer=init_worker) as pool:
                        # 更安全的方式来处理结果
                        for task_id, result in pool.imap_unordered(
                                TCRfuzzy.calculate_similarity_worker, 
                                current_batch,
                                chunksize=max(1, len(current_batch) // (num_processes * 4))):
                            
                            # 只有当结果不为None时才添加
                            if result is not None and isinstance(result, dict):
                                batch_results.append(result)
                            
                            # 更新进度条
                            pbar.update(1)
                            processed_tasks += 1
                    
                    # 保存批次结果
                    if batch_results:
                        logging.info(f"保存第 {batch_idx+1}/{num_batches} 批结果，共 {len(batch_results)} 条")
                        self.save_batch_results(batch_results, output_path, is_first_batch)
                        is_first_batch = False
                    
                    # 清理内存
                    del batch_results
                    gc.collect()
            
            logging.info(f"处理完成！共处理 {processed_tasks} 对序列")
            logging.info(f"结果已保存到: {output_path}")
                
        except Exception as e:
            logging.error(f"处理过程中出错: {e}")
        
        finally:
            # 清理所有变量
            try:
                del col1
                del col2
                del all_tasks
            except:
                pass
            
            # 强制进行垃圾回收
            gc.collect()

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="TCRfuzzy - TCR序列相似度比较工具",
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
    parser.add_argument("--threshold", type=int, default=0,
                        help="仅保存相似度分数高于此阈值的结果（0-100）")
    parser.add_argument("--processes", type=int, default=None,
                        help="使用的进程数（默认为CPU核心数的一半）")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="每批处理的任务数量")
    
    args = parser.parse_args()
    
    # 验证文件存在
    if not os.path.exists(args.csv_file):
        logging.error(f"文件不存在: {args.csv_file}")
        return 1
    
    # 验证阈值范围
    if not 0 <= args.threshold <= 100:
        logging.error("阈值必须在0到100之间")
        return 1
    
    # 创建TCRfuzzy实例并运行比较
    tcr_fuzzy = TCRfuzzy()
    tcr_fuzzy.compare_tcrs(
        csv_file=args.csv_file,
        column1=args.column1,
        column2=args.column2,
        output=args.output,
        delimiter=args.delimiter,
        encoding=args.encoding,
        skip_rows=args.skip_rows,
        threshold=args.threshold,
        num_processes=args.processes,
        batch_size=args.batch_size
    )
    
    return 0

if __name__ == "__main__":
    try:
        # 设置NumExpr线程数
        os.environ["NUMEXPR_MAX_THREADS"] = "4"
        
        # 设置文件描述符限制
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
        except (ImportError, ValueError):
            pass
        
        # 运行主程序
        result = main()
        
        # 清理资源
        gc.collect()
        
        # 关闭日志
        for handler in logging.root.handlers[:]:
            handler.flush()
            handler.close()
            logging.root.removeHandler(handler)
        
        # 正常退出
        os._exit(result)
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        os._exit(1)
    except Exception as e:
        logging.error(f"程序执行出错: {e}")
        os._exit(1)
