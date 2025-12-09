#!/usr/bin/env python3
"""
TRC论文完整实验执行脚本
自动运行所有4个核心实验并生成完整分析
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

class TRCExperimentRunner:
    """TRC实验自动化运行器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results_dir = self.project_root / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 实验配置
        self.experiments = [
            {
                'name': 'DRL + Cheat (主实验)',
                'config': 'configs/trc_balanced_realistic.yml',
                'script': 'run_trc_full_training.py',
                'type': 'drl',
                'expected_time': '约15-20分钟'
            },
            {
                'name': 'DRL + FlipIt (消融实验)',
                'config': 'configs/trc_balanced_realistic_flipit.yml',
                'script': 'run_trc_full_training.py',
                'type': 'drl',
                'expected_time': '约15-20分钟'
            },
            {
                'name': '传统 + Cheat (基线)',
                'config': 'configs/trc_traditional_baseline.yml',
                'script': 'run_traditional_experiment.py',
                'type': 'traditional',
                'episodes': 100,
                'expected_time': '约5-8分钟'
            },
            {
                'name': '传统 + FlipIt (基线)',
                'config': 'configs/trc_traditional_flipit_baseline.yml',
                'script': 'run_traditional_experiment.py',
                'type': 'traditional',
                'episodes': 100,
                'expected_time': '约5-8分钟'
            }
        ]
        
        self.execution_log = []
        
    def print_banner(self, text: str):
        """打印横幅"""
        print("\n" + "=" * 80)
        print(f"  {text}")
        print("=" * 80 + "\n")
        
    def run_experiment(self, exp_config: dict, exp_number: int) -> dict:
        """运行单个实验"""
        
        self.print_banner(f"实验 {exp_number}/4: {exp_config['name']}")
        
        print(f"📋 实验信息:")
        print(f"   - 配置文件: {exp_config['config']}")
        print(f"   - 执行脚本: {exp_config['script']}")
        print(f"   - 预计时间: {exp_config['expected_time']}")
        
        # 构建命令
        if exp_config['type'] == 'drl':
            cmd = [sys.executable, exp_config['script'], exp_config['config']]
        else:
            cmd = [sys.executable, exp_config['script'], exp_config['config'], 
                   '--episodes', str(exp_config['episodes'])]
        
        print(f"\n🚀 开始执行: {' '.join(cmd)}")
        print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        try:
            # 运行实验
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                cwd=str(self.project_root)
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 记录结果
            log_entry = {
                'experiment': exp_config['name'],
                'config': exp_config['config'],
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'duration_seconds': duration,
                'duration_formatted': f"{duration/60:.1f}分钟",
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            self.execution_log.append(log_entry)
            
            # 打印结果
            if result.returncode == 0:
                print(f"\n✅ 实验完成！")
                print(f"⏱️  用时: {duration/60:.1f}分钟")
                
                # 尝试提取关键结果
                if '攻击方胜率' in result.stdout:
                    print(f"\n📊 关键结果:")
                    for line in result.stdout.split('\n'):
                        if any(key in line for key in ['攻击方胜率', '防守方平均', '攻击方平均', '平均博弈']):
                            print(f"   {line.strip()}")
            else:
                print(f"\n❌ 实验失败！返回代码: {result.returncode}")
                print(f"错误输出:\n{result.stderr}")
                
            return log_entry
            
        except Exception as e:
            print(f"\n❌ 实验执行异常: {e}")
            log_entry = {
                'experiment': exp_config['name'],
                'config': exp_config['config'],
                'success': False,
                'error': str(e)
            }
            self.execution_log.append(log_entry)
            return log_entry
    
    def run_all_experiments(self):
        """运行所有实验"""
        
        self.print_banner("TRC论文完整实验自动化执行系统")
        
        print("📋 实验清单:")
        for i, exp in enumerate(self.experiments, 1):
            print(f"   {i}. {exp['name']} - {exp['expected_time']}")
        
        total_estimated_time = "约40-60分钟"
        print(f"\n⏰ 预计总时间: {total_estimated_time}")
        
        input("\n按Enter键开始执行所有实验...")
        
        overall_start = time.time()
        
        # 运行每个实验
        for i, exp_config in enumerate(self.experiments, 1):
            self.run_experiment(exp_config, i)
            
            # 实验间休息
            if i < len(self.experiments):
                print(f"\n⏸️  休息5秒后开始下一个实验...\n")
                time.sleep(5)
        
        overall_end = time.time()
        overall_duration = overall_end - overall_start
        
        # 生成总结报告
        self.generate_summary_report(overall_duration)
        
    def generate_summary_report(self, total_duration: float):
        """生成总结报告"""
        
        self.print_banner("实验执行总结")
        
        successful = sum(1 for log in self.execution_log if log['success'])
        failed = len(self.execution_log) - successful
        
        print(f"📊 执行统计:")
        print(f"   - 总实验数: {len(self.execution_log)}")
        print(f"   - 成功: {successful}")
        print(f"   - 失败: {failed}")
        print(f"   - 总用时: {total_duration/60:.1f}分钟")
        
        print(f"\n📋 详细结果:")
        for i, log in enumerate(self.execution_log, 1):
            status = "✅" if log['success'] else "❌"
            duration = log.get('duration_formatted', 'N/A')
            print(f"   {i}. {status} {log['experiment']} - {duration}")
        
        # 保存日志
        import json
        log_file = self.results_dir / f"execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.execution_log, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 执行日志已保存: {log_file}")
        
        # 检查是否可以运行分析
        if successful == len(self.experiments):
            print(f"\n🎉 所有实验完成！现在可以运行可视化分析了")
            print(f"\n运行以下命令生成分析报告:")
            print(f"   python analysis/trc_drl_defense_analysis.py results")
        else:
            print(f"\n⚠️  有{failed}个实验失败，请检查日志并重新运行失败的实验")

def main():
    """主函数"""
    runner = TRCExperimentRunner()
    
    try:
        runner.run_all_experiments()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行")
        runner.generate_summary_report(0)
    except Exception as e:
        print(f"\n\n❌ 执行异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()









