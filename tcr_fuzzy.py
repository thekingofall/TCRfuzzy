import sys
import subprocess
import importlib.util
import os
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime
import gc
import warnings
warnings.filterwarnings('ignore')
import argparse  # 添加这行
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
        self.required_packages = ['pandas', 'fuzzywuzzy', 'python-Levenshtein', 'tqdm', 'argparse']
        self.check_and_install_packages()
        
        # 导入所需库
        import pandas as pd
        from fuzzywuzzy import fuzz
        import itertools
        from tqdm import tqdm
        import argparse
        import numpy as np
        
        self.pd = pd
        self.fuzz = fuzz
        self.tqdm = tqdm

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

    def compare_tcrs(self, csv_file: str, column1: int = 0, column2: int = 1,
                    output: str = None, delimiter: str = ",", encoding: str = "utf-8",
                    skip_rows: int = 0, threshold: int = 0) -> None:
        """比较TCR序列"""
        try:
            # 读取CSV文件
            df = self.pd.read_csv(csv_file, delimiter=delimiter, 
                                encoding=encoding, skiprows=skip_rows)
            
            # 验证列索引
            if len(df.columns) <= max(column1, column2):
                logging.error(f"CSV文件只有{len(df.columns)}列，但指定了列{max(column1, column2)}")
                return
            
            # 获取列数据
            col1 = df.iloc[:, column1].astype(str).tolist()
            col2 = df.iloc[:, column2].astype(str).tolist()
            
            # 清理DataFrame
            del df
            gc.collect()
            
            # 计算总对比数
            total_comparisons = len(col1) * len(col2)
            logging.info(f"将比较 {len(col1)} x {len(col2)} = {total_comparisons} 个组合")
            
            # 存储结果
            all_results = []
            
            # 使用tqdm显示进度
            with self.tqdm(total=total_comparisons, desc="处理进度", unit="对") as pbar:
                for str1 in col1:
                    for str2 in col2:
                        result = self.calculate_similarity(str1, str2)
                        if result is not None:
                            if threshold == 0 or max(
                                result['相似度(ratio)'],
                                result['部分相似度(partial_ratio)'],
                                result['排序标记相似度(token_sort_ratio)'],
                                result['集合标记相似度(token_set_ratio)']
                            ) >= threshold:
                                all_results.append(result)
                        pbar.update(1)
            
            # 处理结果
            if not all_results:
                logging.warning("没有找到满足条件的结果")
                return
            
            # 创建结果DataFrame并保存
            try:
                results_df = self.pd.DataFrame(all_results)
                output_path = output or f"{os.path.splitext(csv_file)[0]}_TCR_similarity_results.csv"
                results_df.to_csv(output_path, index=False)
                logging.info(f"处理完成！共计算 {total_comparisons} 对，保存了 {len(results_df)} 条结果")
                logging.info(f"结果已保存到: {output_path}")
                

                
            except Exception as e:
                logging.error(f"保存结果时出错: {e}")
                
        except Exception as e:
            logging.error(f"处理过程中出错: {e}")
        

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
    
    args = parser.parse_args()
    
    # 验证文件存在
    if not os.path.exists(args.csv_file):
        logging.error(f"文件不存在: {args.csv_file}")
        sys.exit(1)
    
    # 验证阈值范围
    if not 0 <= args.threshold <= 100:
        logging.error("阈值必须在0到100之间")
        sys.exit(1)
    
    try:
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
            threshold=args.threshold
        )
    finally:
        # 确保程序完全清理
        gc.collect()
        # 关闭所有日志处理器
        for handler in logging.root.handlers:
            handler.close()
        logging.shutdown()

if __name__ == "__main__":
    # 设置NumExpr线程数
    os.environ["NUMEXPR_MAX_THREADS"] = "4"
    main()
