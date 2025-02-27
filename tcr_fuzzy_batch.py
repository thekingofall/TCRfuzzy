#!/usr/bin/env python3
# tcr_fuzzy_safe.py - 安全版TCR序列相似度比较工具

import sys
import os
import subprocess
import importlib.util
import logging
from datetime import datetime
import argparse
import gc
import warnings
import tempfile
import shutil
from pathlib import Path
import csv
warnings.filterwarnings('ignore')

class TCRFuzzySafe:
    def __init__(self):
        """初始化TCRFuzzySafe类"""
        # 设置日志
        log_file = f'TCR_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        self.log_file = log_file
        
        # 设置必要的包
        self.required_packages = ['pandas', 'fuzzywuzzy', 'python-Levenshtein', 'tqdm']
        self.check_and_install_packages()
        
        # 导入所需库
        import pandas as pd
        from tqdm import tqdm
        
        self.pd = pd
        self.tqdm = tqdm
        
        # 创建临时工作目录
        self.work_dir = tempfile.mkdtemp(prefix="tcr_fuzzy_")
        logging.info(f"创建临时工作目录: {self.work_dir}")

    def __del__(self):
        """清理临时目录"""
        try:
            shutil.rmtree(self.work_dir)
            logging.info(f"已删除临时工作目录: {self.work_dir}")
        except Exception as e:
            logging.warning(f"清理临时目录时出错: {e}")

    def check_and_install_packages(self):
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

    def create_similarity_calculator(self):
        """创建相似度计算器脚本"""
        calculator_path = os.path.join(self.work_dir, "calculate_similarity.py")
        
        with open(calculator_path, 'w') as f:
            f.write("""#!/usr/bin/env python3
import sys
import csv
from fuzzywuzzy import fuzz

def preprocess_string(s):
    if not isinstance(s, str):
        s = str(s)
    return s.strip().lower()

def calculate_similarity(str1, str2, threshold=0):
    # 预处理字符串
    str1_processed = preprocess_string(str1)
    str2_processed = preprocess_string(str2)
    
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
        return result
    
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
        return result
    
    # 计算各种相似度
    ratio = fuzz.ratio(str1_processed, str2_processed)
    partial_ratio = fuzz.partial_ratio(str1_processed, str2_processed)
    token_sort_ratio = fuzz.token_sort_ratio(str1_processed, str2_processed)
    token_set_ratio = fuzz.token_set_ratio(str1_processed, str2_processed)
    
    # 检查是否满足阈值
    max_score = max(ratio, partial_ratio, token_sort_ratio, token_set_ratio)
    if threshold > 0 and max_score < threshold:
        return None
    
    result = {
        '字符串1': str1,
        '字符串2': str2,
        '相似度(ratio)': ratio,
        '部分相似度(partial_ratio)': partial_ratio,
        '排序标记相似度(token_sort_ratio)': token_sort_ratio,
        '集合标记相似度(token_set_ratio)': token_set_ratio
    }
    return result

def main():
    if len(sys.argv) != 4:
        print("用法: python calculate_similarity.py <输入文件> <输出文件> <阈值>")
        return 1
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    threshold = int(sys.argv[3])
    
    try:
        # 读取输入文件
        pairs = []
        with open(input_file, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    pairs.append((row[0], row[1]))
        
        # 跳过标题行
        if pairs and pairs[0][0] == '字符串1':
            pairs = pairs[1:]
        
        # 处理每一对
        results = []
        for str1, str2 in pairs:
            result = calculate_similarity(str1, str2, threshold)
            if result:
                results.append(result)
        
        # 写入结果
        if results:
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)
        else:
            # 创建空文件
            with open(output_file, 'w') as f:
                pass
                
        return 0
    
    except Exception as e:
        print(f"错误: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
""")
        
        # 设置为可执行
        os.chmod(calculator_path, 0o755)
        return calculator_path

    def compare_tcrs(self, csv_file, column1=0, column2=1, output=None, delimiter=",", 
                     encoding="utf-8", skip_rows=0, threshold=0, max_processes=4, batch_size=50):
        """比较TCR序列"""
        try:
            # 读取CSV文件
            logging.info(f"正在读取CSV文件: {csv_file}")
            df = self.pd.read_csv(csv_file, delimiter=delimiter, 
                                  encoding=encoding, skiprows=skip_rows)
            
            # 验证列索引
            if len(df.columns) <= max(column1, column2):
                logging.error(f"CSV文件只有{len(df.columns)}列，但指定了列{max(column1, column2)}")
                return False
            
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
                # 同一列只比较上三角矩阵
                pairs = [(col1[i], col2[j]) for i in range(len1) for j in range(i+1, len2)]
                total_comparisons = len(pairs)
            else:
                # 不同列全比较
                pairs = [(str1, str2) for str1 in col1 for str2 in col2]
                total_comparisons = len1 * len2
            
            logging.info(f"将比较第{column1}列({len1}个序列) x 第{column2}列({len2}个序列) = {total_comparisons} 个组合")
            
            # 输出文件
            output_file = output or f"{os.path.splitext(csv_file)[0]}_TCR_similarity_results.csv"
            
            # 创建相似度计算器脚本
            calculator_script = self.create_similarity_calculator()
            
            # 分批处理
            num_batches = (total_comparisons + batch_size - 1) // batch_size
            logging.info(f"任务将分为 {num_batches} 批进行处理，每批最多 {batch_size} 个比较对")
            
            # 生成批处理任务
            batch_files = []
            result_files = []
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_comparisons)
                batch_pairs = pairs[start_idx:end_idx]
                
                # 创建批处理输入文件
                batch_file = os.path.join(self.work_dir, f"batch_{batch_idx+1}.csv")
                with open(batch_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['字符串1', '字符串2'])
                    writer.writerows(batch_pairs)
                
                batch_files.append(batch_file)
                
                # 设置结果文件
                result_file = os.path.join(self.work_dir, f"result_{batch_idx+1}.csv")
                result_files.append(result_file)
            
            # 生成并执行命令
            commands = [
                f"{sys.executable} {calculator_script} {batch_file} {result_file} {threshold}"
                for batch_file, result_file in zip(batch_files, result_files)
            ]
            
            # 创建命令批处理文件
            cmd_file = os.path.join(self.work_dir, "commands.sh")
            with open(cmd_file, 'w') as f:
                for cmd in commands:
                    f.write(cmd + "\n")
            
            # 并行执行命令
            logging.info(f"开始并行处理 {num_batches} 个批次，最大进程数: {max_processes}")
            
            # 选择最适合当前系统的并行执行方法
            if self.check_command_exists("parallel"):
                # 使用GNU parallel
                parallel_cmd = f"parallel -j {max_processes} < {cmd_file}"
                logging.info(f"使用GNU parallel执行命令: {parallel_cmd}")
                proc = subprocess.Popen(parallel_cmd, shell=True)
                proc.wait()
            elif self.check_command_exists("xargs"):
                # 使用xargs
                xargs_cmd = f"cat {cmd_file} | xargs -P {max_processes} -I{{}} bash -c 'echo 执行: {{}}; {{}}'"
                logging.info(f"使用xargs执行命令: {xargs_cmd}")
                proc = subprocess.Popen(xargs_cmd, shell=True)
                proc.wait()
            else:
                # 使用Python的subprocess
                logging.info("未找到parallel或xargs，使用Python subprocess逐个执行命令")
                for i, cmd in enumerate(commands, 1):
                    logging.info(f"执行批次 {i}/{num_batches}: {cmd}")
                    proc = subprocess.Popen(cmd, shell=True)
                    proc.wait()
            
            # 合并结果
            logging.info("所有批次处理完成，合并结果...")
            successful_results = 0
            with open(output_file, 'w', newline='') as outf:
                writer = None
                for i, result_file in enumerate(result_files, 1):
                    if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
                        with open(result_file, 'r', newline='') as inf:
                            reader = csv.DictReader(inf)
                            
                            # 如果这是第一个有效的结果文件，写入标题
                            if writer is None:
                                writer = csv.DictWriter(outf, fieldnames=reader.fieldnames)
                                writer.writeheader()
                            
                            # 写入结果
                            for row in reader:
                                writer.writerow(row)
                                successful_results += 1
            
            logging.info(f"处理完成！共计算 {total_comparisons} 对，保存了 {successful_results} 条结果")
            logging.info(f"结果已保存到: {output_file}")
            
            return True
            
        except Exception as e:
            logging.error(f"处理过程中出错: {e}")
            return False
    
    @staticmethod
    def check_command_exists(command):
        """检查命令是否存在"""
        try:
            subprocess.check_call(["which", command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            return False

def main():
    parser = argparse.ArgumentParser(
        description="TCRfuzzySafe - 安全的TCR序列相似度比较工具",
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
    parser.add_argument("-p", "--processes", type=int, default=4,
                        help="最大并行进程数")
    parser.add_argument("-b", "--batch-size", type=int, default=50,
                        help="每批处理的比较对数量")
    
    args = parser.parse_args()
    
    # 验证文件存在
    if not os.path.exists(args.csv_file):
        logging.error(f"文件不存在: {args.csv_file}")
        return 1
    
    # 验证阈值范围
    if not 0 <= args.threshold <= 100:
        logging.error("阈值必须在0到100之间")
        return 1
    
    # 创建TCRFuzzySafe实例并运行比较
    tcr_fuzzy = TCRFuzzySafe()
    success = tcr_fuzzy.compare_tcrs(
        args.csv_file,
        args.column1,
        args.column2,
        args.output,
        args.delimiter,
        args.encoding,
        args.skip_rows,
        args.threshold,
        args.processes,
        args.batch_size
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    try:
        result = main()
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        logging.error(f"程序执行出错: {e}")
        sys.exit(1)
