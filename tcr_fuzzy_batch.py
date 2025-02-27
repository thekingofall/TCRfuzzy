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
import concurrent.futures
import threading
import queue
import time
warnings.filterwarnings('ignore')

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
        
        # 导入所需库
        import pandas as pd
        from fuzzywuzzy import fuzz
        from tqdm import tqdm
        import psutil
        
        self.pd = pd
        self.fuzz = fuzz
        self.tqdm = tqdm
        self.psutil = psutil
        
        # 设置线程安全锁和结果队列
        self.lock = threading.Lock()
        self.result_queue = queue.Queue()
        self.active_workers = 0
        self.worker_lock = threading.Lock()

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

    def calculate_similarity(self, str1: str, str2: str) -> Optional[Dict[str, Any]]:
        """计算字符串相似度"""
        try:
            # 预处理字符串
            str1_processed = self.preprocess_string(str1)
            str2_processed = self.preprocess_string(str2)
            
            # 如果字符串完全相同，直接返回100分
            if str1_processed == str2_processed:
                return {
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
                    '字符串1': str1,
                    '字符串2': str2,
                    '相似度(ratio)': 0,
                    '部分相似度(partial_ratio)': 0,
                    '排序标记相似度(token_sort_ratio)': 0,
                    '集合标记相似度(token_set_ratio)': 0
                }
            
            # 计算各种相似度
            ratio = self.fuzz.ratio(str1_processed, str2_processed)
            partial_ratio = self.fuzz.partial_ratio(str1_processed, str2_processed)
            token_sort_ratio = self.fuzz.token_sort_ratio(str1_processed, str2_processed)
            token_set_ratio = self.fuzz.token_set_ratio(str1_processed, str2_processed)
            
            return {
                '字符串1': str1,
                '字符串2': str2,
                '相似度(ratio)': ratio,
                '部分相似度(partial_ratio)': partial_ratio,
                '排序标记相似度(token_sort_ratio)': token_sort_ratio,
                '集合标记相似度(token_set_ratio)': token_set_ratio
            }
        except Exception as e:
            logging.error(f"计算相似度时出错: {e}")
            return None

    def worker(self, work_queue, threshold, pbar, batch_size=100):
        """工作线程处理函数"""
        try:
            with self.worker_lock:
                self.active_workers += 1
                
            local_results = []
            
            while True:
                try:
                    # 从工作队列中获取任务
                    task = work_queue.get(block=False)
                    if task is None:  # 哨兵值，表示结束
                        break
                        
                    str1, str2 = task
                    result = self.calculate_similarity(str1, str2)
                    
                    if result is not None:
                        if threshold == 0 or max(
                            result['相似度(ratio)'],
                            result['部分相似度(partial_ratio)'],
                            result['排序标记相似度(token_sort_ratio)'],
                            result['集合标记相似度(token_set_ratio)']
                        ) >= threshold:
                            local_results.append(result)
                    
                    # 更新进度条
                    with self.lock:
                        pbar.update(1)
                    
                    # 如果本地结果达到批处理大小，放入队列
                    if len(local_results) >= batch_size:
                        self.result_queue.put(local_results)
                        local_results = []
                        
                except queue.Empty:
                    break
                    
            # 提交剩余的结果
            if local_results:
                self.result_queue.put(local_results)
                
        except Exception as e:
            logging.error(f"工作线程发生错误: {e}")
        finally:
            with self.worker_lock:
                self.active_workers -= 1

    def check_memory_usage(self, threshold_percent=80):
        """检查内存使用情况，如果超过阈值返回True"""
        memory_info = self.psutil.virtual_memory()
        return memory_info.percent > threshold_percent

    def save_results_to_file(self, results, output_path, append=False):
        """保存结果到文件"""
        try:
            df = self.pd.DataFrame(results)
            mode = 'a' if append else 'w'
            header = not append
            
            df.to_csv(output_path, index=False, mode=mode, header=header)
            return True
        except Exception as e:
            logging.error(f"保存结果时出错: {e}")
            return False

    def collect_and_save_results(self, output_path, timeout=1):
        """收集并保存队列中的结果"""
        all_results = []
        first_save = True
        
        while True:
            try:
                batch_results = self.result_queue.get(timeout=timeout)
                all_results.extend(batch_results)
                
                # 当内存使用过高或结果数量足够多时，保存到文件并清空
                if self.check_memory_usage(threshold_percent=70) or len(all_results) > 100000:
                    logging.info(f"保存中间结果，共 {len(all_results)} 条")
                    self.save_results_to_file(all_results, output_path, append=not first_save)
                    first_save = False
                    all_results = []  # 清空内存
                    gc.collect()  # 触发垃圾回收
            except queue.Empty:
                # 如果没有活动的工作线程且队列为空，退出循环
                with self.worker_lock:
                    if self.active_workers == 0 and self.result_queue.empty():
                        break
                continue
        
        # 保存剩余结果
        if all_results:
            self.save_results_to_file(all_results, output_path, append=not first_save)
            
        return first_save  # 返回是否为第一次保存（即是否有结果）

    def compare_tcrs(self, csv_file: str, column1: int = 0, column2: int = 1,
                    output: str = None, delimiter: str = ",", encoding: str = "utf-8",
                    skip_rows: int = 0, threshold: int = 0, num_threads: int = None,
                    chunk_size: int = 1000) -> None:
        """比较TCR序列"""
        try:
            # 设置线程数
            if num_threads is None:
                num_threads = max(1, os.cpu_count() - 1)  # 默认使用CPU核心数减1
            
            logging.info(f"将使用 {num_threads} 个线程进行处理")
            
            # 读取CSV文件
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
            
            # 创建工作队列
            work_queue = queue.Queue()
            
            # 填充任务队列
            if column1 == column2:
                # 如果是同一列的比较，只比较上三角矩阵
                for i, str1 in enumerate(col1):
                    for str2 in col2[i+1:]:
                        work_queue.put((str1, str2))
            else:
                # 不同列的比较，进行全比较
                for str1 in col1:
                    for str2 in col2:
                        work_queue.put((str1, str2))
            
            # 为每个工作线程添加结束标志
            for _ in range(num_threads):
                work_queue.put(None)
            
            # 设置输出路径
            output_path = output or f"{os.path.splitext(csv_file)[0]}_TCR_similarity_results.csv"
            
            # 启动结果收集和保存线程
            collector_thread = threading.Thread(
                target=lambda: self.collect_and_save_results(output_path),
                daemon=True
            )
            collector_thread.start()
            
            # 使用tqdm显示进度
            with self.tqdm(total=total_comparisons, desc="处理进度", unit="对") as pbar:
                # 启动工作线程池
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = []
                    for _ in range(num_threads):
                        future = executor.submit(self.worker, work_queue, threshold, pbar, chunk_size)
                        futures.append(future)
                    
                    # 等待所有线程完成
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            future.result()  # 获取线程的返回值，捕获可能的异常
                        except Exception as e:
                            logging.error(f"线程执行出错: {e}")
            
            # 等待收集器线程完成
            collector_thread.join()
            
            logging.info(f"处理完成！结果已保存到: {output_path}")
                
        except Exception as e:
            logging.error(f"处理过程中出错: {e}")
        
        finally:
            # 清理所有变量
            try:
                del col1
                del col2
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
    parser.add_argument("--threads", type=int, default=None,
                        help="使用的线程数（默认为CPU核心数-1）")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="每批保存的结果数量")
    
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
        num_threads=args.threads,
        chunk_size=args.chunk_size
    )
    
    return 0

if __name__ == "__main__":
    try:
        # 设置NumExpr线程数
        os.environ["NUMEXPR_MAX_THREADS"] = "4"
        
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
        
    except Exception as e:
        logging.error(f"程序执行出错: {e}")
        os._exit(1)
