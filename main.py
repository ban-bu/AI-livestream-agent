"""
主程序入口文件，用于启动仿真系统
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any
import random
import time
import concurrent.futures
import json
import re
import numpy as np
from ollama_client import OllamaClient
from openai_client import OpenAIClient
from datetime import datetime
import psutil  # 如果没有安装，需要先pip install psutil
import time

# 添加调试输出
print("开始加载模块...")

from config import SIMULATION_ROUNDS, SAVE_RESULTS
print("配置加载成功")

from utils.logger import setup_logger
print("日志记录器加载成功")

from utils.llm_utils import get_llm_client, set_llm_client_type
print("LLM工具加载成功")

from utils.data_recorder import get_data_recorder
print("数据记录器加载成功")

from simulation.environment import Environment
print("环境加载成功")

from simulation.simulator import Simulator
print("仿真器加载成功")

try:
    from agents.platform_agent import PlatformAgent
    print("平台代理加载成功")
    
    from agents.streamer_agent import StreamerAgent, TopStreamerAgent, RegularStreamerAgent
    print("主播代理加载成功")
    
    from agents.merchant_agent import MerchantAgent
    print("商家代理加载成功")
    
    from agents.consumer_agent import ConsumerAgent
    print("消费者代理加载成功")
except Exception as e:
    print(f"加载代理时出错: {str(e)}")

# 配置日志级别
LOG_LEVEL = logging.INFO

# 全局变量，用于存储当前使用的客户端类型
# USE_OPENAI = False
# LLM_CLIENT = None

# def get_llm_client():
#     """
#     获取LLM客户端实例
#     
#     Returns:
#         Union[OllamaClient, OpenAIClient]: LLM客户端实例
#     """
#     global LLM_CLIENT
#     if LLM_CLIENT is None:
#         if USE_OPENAI:
#             LLM_CLIENT = OpenAIClient()
#         else:
#             LLM_CLIENT = OllamaClient()
#     return LLM_CLIENT

def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="电子商务直播供应链仿真系统")
    parser.add_argument(
        "--rounds", type=int, default=SIMULATION_ROUNDS,
        help=f"仿真回合数，默认值 {SIMULATION_ROUNDS}"
    )
    parser.add_argument(
        "--save", action="store_true", default=SAVE_RESULTS,
        help="是否保存结果"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别，默认INFO"
    )
    
    # 创建互斥组，确保--use-openai和--use-ollama不会同时使用
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument(
        "--use-openai", action="store_true", default=False,
        help="使用OpenAI API进行LLM调用"
    )
    llm_group.add_argument(
        "--use-ollama", action="store_true", default=False,
        help="使用Ollama进行LLM调用（默认行为）"
    )
    parser.add_argument(
        "--disable-reuse", action="store_true", default=False,
        help="禁用DataProcessor实例重用，每次创建新的数据处理实例"
    )
    parser.add_argument(
        "--disable-cache", action="store_true", default=False,
        help="禁用LLM响应缓存，每次都调用LLM"
    )
    
    return parser.parse_args()

def setup_simulation(args) -> Simulator:
    """
    设置仿真系统
    
    Args:
        args: 命令行参数
        
    Returns:
        Simulator: 仿真实例
    """
    logger = logging.getLogger("simulation")
    logger.info("设置仿真系统")
    
    # 导入DataProcessor的get_data_processor函数
    from utils.data_processor import get_data_processor
    
    # 为了兼容性，记录--disable-reuse参数的状态，但不影响全局实例的使用
    if args.disable_reuse:
        logger.info("注意: --disable-reuse参数被忽略，现在始终使用全局DataProcessor实例")
    
    # 初始化全局DataProcessor实例，确保所有组件都使用相同的实例
    data_processor = get_data_processor()
    logger.info(f"使用全局DataProcessor目录: {data_processor.simulation_dir}")
    
    # 设置LLM缓存状态
    if args.disable_cache:
        # 导入OllamaClient以修改缓存设置
        from ollama_client import OllamaClient
        import shutil
        import os
        
        # 彻底禁用缓存
        client = OllamaClient()
        client.use_cache = False
        client.cache = {}  # 清空内存缓存
        client.cache_access_times = {}  # 清空LRU缓存访问时间
        client.response_cache = {}  # 清空响应缓存
        
        # 清空缓存目录
        cache_dir = "cache/responses"
        if os.path.exists(cache_dir):
            try:
                # 删除所有缓存文件
                for file in os.listdir(cache_dir):
                    if file.endswith(".json"):
                        os.remove(os.path.join(cache_dir, file))
                logger.info(f"缓存目录已清空: {cache_dir}")
            except Exception as e:
                logger.warning(f"清空缓存目录失败: {str(e)}")
        
        logger.info("LLM响应缓存已完全禁用，每次都会调用LLM")
    
    # 设置全局客户端类型 (--use-ollama时use_openai为False)
    use_openai = args.use_openai
    set_llm_client_type(use_openai)
    if use_openai:
        logger.info("使用OpenAI API进行LLM调用")
    else:
        logger.info("使用Ollama进行LLM调用")
    
    # 创建仿真控制器，但禁用自动初始化
    print("开始创建仿真器...")
    simulator = Simulator(rounds=args.rounds, save_results=args.save, auto_init=False)
    print("仿真器创建成功")
    
    # 初始化代理
    print("开始初始化代理...")
    init_agents(simulator)
    print("代理初始化成功")
    
    return simulator

def init_agents(simulator: Simulator) -> None:
    """
    初始化所有代理，使用并行处理提高效率
    
    Args:
        simulator: 仿真控制器
    """
    logger = logging.getLogger("simulation")
    logger.info("初始化所有代理")
    
    # 创建一个初始化单个代理的辅助函数
    def init_single_agent(agent_type, agent_id, environment):
        """
        初始化单个代理
        
        Args:
            agent_type: 代理类型
            agent_id: 代理ID
            environment: 仿真环境
            
        Returns:
            BaseAgent: 初始化的代理实例
        """
        agent = None
        
        try:
            start_time = time.time()
            if agent_type == "platform":
                agent = PlatformAgent(environment)
            elif agent_type == "streamer":
                # 获取主播信息，判断是头部主播还是普通主播
                streamer_info = environment.get_streamer(agent_id)
                if streamer_info and streamer_info.get("streamer_type") == "top":
                    agent = TopStreamerAgent(agent_id, environment)
                    logger.info(f"创建头部主播 {agent_id}")
                else:
                    agent = RegularStreamerAgent(agent_id, environment)
                    logger.info(f"创建普通主播 {agent_id}")
            elif agent_type == "merchant":
                agent = MerchantAgent(agent_id, environment)
            elif agent_type == "consumer":
                agent = ConsumerAgent(agent_id, environment)
            
            end_time = time.time()
            logger.info(f"{agent_type.capitalize()} 代理 {agent_id} 初始化成功，耗时: {(end_time - start_time):.2f} 秒")
        except Exception as e:
            logger.error(f"初始化 {agent_type} 代理 {agent_id} 时出错: {str(e)}")
            raise
        
        return agent
    
    try:
        # 准备所有需要初始化的代理任务
        init_tasks = []
        
        # 添加平台代理初始化任务
        print("准备初始化平台代理...")
        init_tasks.append(("platform", "platform", simulator.environment))
        
        # 添加主播代理初始化任务
        print("准备初始化主播代理...")
        streamers = simulator.environment.get_streamers()
        for streamer_id in streamers:
            init_tasks.append(("streamer", streamer_id, simulator.environment))
        
        # 添加商家代理初始化任务
        print("准备初始化商家代理...")
        merchants = simulator.environment.get_merchants()
        for merchant_id in merchants:
            init_tasks.append(("merchant", merchant_id, simulator.environment))
        
        # 添加消费者代理初始化任务
        print("准备初始化消费者代理...")
        consumers = simulator.environment.get_consumers()
        for consumer_id in consumers:
            init_tasks.append(("consumer", consumer_id, simulator.environment))
        
        # 决定线程池大小：通常IO密集型任务可以使用更多线程
        # 使用min函数确保不会创建过多线程
        max_workers = min(32, len(init_tasks))
        print(f"使用 {max_workers} 个线程初始化 {len(init_tasks)} 个代理...")
        
        # 使用线程池并行初始化代理
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有初始化任务
            future_to_task = {
                executor.submit(init_single_agent, task[0], task[1], task[2]): task 
                for task in init_tasks
            }
            
            # 收集初始化结果
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                agent_type, agent_id, _ = task
                
                try:
                    agent = future.result()
                    if agent:
                        # 添加到仿真器
                        if agent_type == "platform":
                            simulator.set_platform_agent(agent)
                            logger.info("平台代理初始化成功")
                        elif agent_type == "streamer":
                            simulator.add_streamer_agent(agent_id, agent)
                            logger.info(f"主播代理 {agent_id} 初始化成功")
                        elif agent_type == "merchant":
                            simulator.add_merchant_agent(agent_id, agent)
                            logger.info(f"商家代理 {agent_id} 初始化成功")
                        elif agent_type == "consumer":
                            simulator.add_consumer_agent(agent_id, agent)
                            logger.info(f"消费者代理 {agent_id} 初始化成功")
                except Exception as e:
                    logger.error(f"处理代理初始化结果时出错: {e}")
        
        print(f"所有代理初始化成功，共 {len(init_tasks)} 个代理")
        logger.info(f"初始化了 {len(init_tasks)} 个代理")
    except Exception as e:
        logger.error(f"初始化代理时出错: {e}")
        raise

def update_simulator_methods(simulator: Simulator) -> None:
    """
    更新仿真器方法，用实际实现替换空方法
    
    Args:
        simulator: 仿真控制器
    """
    # 更新商家-主播谈判阶段
    if not hasattr(simulator, '_negotiation_updated') or not simulator._negotiation_updated:
        simulator._merchant_streamer_negotiation = merchant_streamer_negotiation.__get__(simulator, Simulator)
        simulator._negotiation_updated = True
    
    # 更新平台流量分配阶段
    if not hasattr(simulator, '_traffic_updated') or not simulator._traffic_updated:
        simulator._platform_traffic_allocation = platform_traffic_allocation.__get__(simulator, Simulator)
        simulator._traffic_updated = True
    
    # 更新直播带货阶段
    if not hasattr(simulator, '_sales_updated') or not simulator._sales_updated:
        simulator._live_streaming_sales = live_streaming_sales.__get__(simulator, Simulator)
        simulator._sales_updated = True
    
    # 更新交易与反馈阶段
    if not hasattr(simulator, '_feedback_updated') or not simulator._feedback_updated:
        simulator._transaction_feedback = transaction_feedback.__get__(simulator, Simulator)
        simulator._feedback_updated = True
        
    # 添加打印代理记忆方法
    if not hasattr(simulator, 'print_agent_memories'):
        simulator.print_agent_memories = print_agent_memories.__get__(simulator, Simulator)

def merchant_streamer_negotiation(self) -> None:
    """
    商家-主播谈判阶段 - 多轮对话版本（并行优化）
    
    Args:
        self: 仿真控制器
    """
    # 导入所需模块
    import time
    import random
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from threading import Lock
    
    logger = logging.getLogger("simulation")
    logger.info("执行商家-主播谈判阶段（并行处理）")
    
    # 获取当前状态
    state = self.environment.get_state()
    
    # 1. 商家设定初始价格、佣金
    merchant_initial_offers = {}
    
    # 批量处理商家初始报价
    merchant_tasks = []
    for merchant_id, merchant in state["merchants"].items():
        merchant_agent = self.agents.get(f"merchant_{merchant_id}")
        if merchant_agent:
            merchant_tasks.append((merchant_id, merchant_agent))
    
    # 计算合适的线程数 - 根据CPU核心数和任务数自适应调整
    try:
        # 尝试获取系统信息
        cpu_count = os.cpu_count() or 4
        available_memory = psutil.virtual_memory().available
        total_memory = psutil.virtual_memory().total
        memory_usage_ratio = available_memory / total_memory
        
        # 根据内存使用情况和CPU核心数动态调整线程数
        base_workers = cpu_count + 1  # 基础线程数
        
        # 如果内存使用率高，减少线程数以避免内存压力
        if memory_usage_ratio < 0.3:  # 可用内存少于30%
            max_workers = max(2, cpu_count // 2)  # 减少线程数
            logger.info(f"内存使用率高，减少线程数至 {max_workers}")
        elif memory_usage_ratio > 0.7:  # 可用内存超过70%
            max_workers = min(32, cpu_count * 2)  # 可以使用更多线程
            logger.info(f"内存充足，增加线程数至 {max_workers}")
        else:
            max_workers = base_workers
            logger.info(f"使用标准线程数: {max_workers}")
        
        # 确保线程数不超过任务数
        max_workers = min(max_workers, len(merchant_tasks) * 2)
    except:
        # 如果无法获取系统信息，使用保守的默认值
        max_workers = 4
        logger.warning("无法获取系统资源信息，使用默认线程数: 4")
    
    logger.info(f"使用 {max_workers} 个线程处理 {len(merchant_tasks)} 个商家初始报价任务")
    
    # 使用线程池批量处理商家初始报价
    from concurrent.futures import ThreadPoolExecutor
    
    def process_merchant_offers(task):
        merchant_id, merchant_agent = task
        try:
            logger.info(f"商家 {merchant_id} 设置初始报价")
            return merchant_id, merchant_agent.make_decision(state)
        except Exception as e:
            logger.error(f"处理商家 {merchant_id} 初始报价时出错: {str(e)}")
            return merchant_id, {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_merchant_offers, merchant_tasks))
        
        # 收集结果
        for merchant_id, decision in results:
            if decision:
                merchant_initial_offers[merchant_id] = decision
    
    # 2. 主播与商家多轮谈判
    negotiation_history = {}
    negotiation_results = {}
    MAX_NEGOTIATION_ROUNDS = 3
    
    # 初始化谈判状态
    for streamer_id in state["streamers"].keys():
        negotiation_results[streamer_id] = {
            "selected_products": [],
            "rejected_by": [],
            "agreements": {}
        }
    
    # 生成随机种子
    random_seed = int(time.time() * 1000)
    random.seed(random_seed)
    
    # 创建线程池
    from concurrent.futures import ThreadPoolExecutor
    import threading
    
    # 创建线程安全的字典
    from threading import Lock
    history_lock = Lock()
    results_lock = Lock()
    
    def negotiate_with_merchant(streamer_id, streamer_agent, product_id, merchant_id, merchant_agent, initial_offer):
        # 创建谈判会话标识
        session_key = f"{streamer_id}_merchant_{merchant_id}_product_{product_id}"
        
        # 线程安全地初始化谈判历史
        with history_lock:
            if session_key not in negotiation_history:
                negotiation_history[session_key] = []
        
        # 获取产品信息
        product_info = state["products"].get(product_id, {})
        product_name = product_info.get("name", product_id)
        
        # 第一轮：商家提出初始报价
        merchant_message = f"我是商家 {merchant_id}，对于产品 [{product_name}]，我的初始条件是: 佣金率 {initial_offer.get('commission_rate', 0.1)}, 价格 {initial_offer.get('discount', '原价')}."
        
        with history_lock:
            negotiation_history[session_key].append({"role": "merchant", "content": merchant_message})
        
        # 谈判轮数计数
        negotiation_round = 1
        agreement_reached = False
        merchant_rejected = False
        rejection_reason = ""
        
        # 多轮谈判过程 - 确保至少进行2轮谈判
        while negotiation_round <= MAX_NEGOTIATION_ROUNDS:
            random_tag = f"<!-- session_id: {random.randint(1000000, 9999999)} -->"
            
            # 主播回应
            with history_lock:
                current_history = negotiation_history[session_key].copy()
            
            streamer_prompt = _build_negotiation_prompt(
                streamer_agent,
                current_history,
                state,
                product_id,
                "streamer"
            ) + random_tag
            
            streamer_response = streamer_agent.call_llm(streamer_prompt)
            
            with history_lock:
                negotiation_history[session_key].append({"role": "streamer", "content": streamer_response})
            
            # 添加对话记忆
            streamer_agent.add_memory("dialogue", {
                "session_key": session_key,
                "product_id": product_id,
                "merchant_id": merchant_id,
                "content": streamer_response,
                "role": "streamer"
            }, self.current_round)
            
            logger.info(f"主播 {streamer_id} 的回应: {streamer_response[:50]}...")
            
            # 仅在前两轮强制继续谈判，最后一轮允许决策
            force_continue = negotiation_round < MAX_NEGOTIATION_ROUNDS - 1
            
            # 商家回应
            with history_lock:
                current_history = negotiation_history[session_key].copy()
            
            merchant_prompt = _build_negotiation_prompt(
                merchant_agent,
                current_history,
                state,
                product_id,
                "merchant"
            ) + random_tag
            
            merchant_response = merchant_agent.call_llm(merchant_prompt)
            
            with history_lock:
                negotiation_history[session_key].append({"role": "merchant", "content": merchant_response})
                
            # 添加对话记忆
            merchant_agent.add_memory("dialogue", {
                "session_key": session_key,
                "product_id": product_id,
                "streamer_id": streamer_id,
                "content": merchant_response,
                "role": "merchant"
            }, self.current_round)
            
            logger.info(f"商家 {merchant_id} 的回应: {merchant_response[:50]}...")
            
            # 只有在最后一轮才检查是否达成协议
            if not force_continue:
                # 检查是否达成协议或被拒绝
                client = get_llm_client()
                agreement_reached = False
                merchant_rejected = False
                rejection_reason = ""
                
                try:
                    parsed_response = client.parse_json_response(merchant_response)
                    if parsed_response:
                        # 检查是否明确表示达成协议
                        if "是否达成协议" in parsed_response:
                            agreement_reached = parsed_response["是否达成协议"]
                            
                            # 如果拒绝，获取拒绝原因
                            if not agreement_reached and "拒绝原因" in parsed_response:
                                rejection_reason = parsed_response["拒绝原因"]
                                merchant_rejected = True
                        
                        # 主播视角
                        elif "是否接受" in parsed_response or "是否接受当前条件" in parsed_response:
                            agreement_reached = parsed_response.get("是否接受", parsed_response.get("是否接受当前条件", False))
                            
                            # 如果拒绝，获取拒绝原因
                            if not agreement_reached and "拒绝原因" in parsed_response:
                                rejection_reason = parsed_response["拒绝原因"]
                                merchant_rejected = True
                        
                        # 分析回应文本寻找接受/拒绝信号
                        response_text = parsed_response.get("response", "")
                        if response_text and isinstance(response_text, str):
                            if "接受" in response_text.lower() or "同意" in response_text.lower() or "达成协议" in response_text.lower() or "合作" in response_text.lower():
                                agreement_reached = True
                            elif "拒绝" in response_text.lower() or "不同意" in response_text.lower() or "不接受" in response_text.lower() or "无法合作" in response_text.lower():
                                merchant_rejected = True
                                if not rejection_reason:
                                    rejection_reason = "从回应中检测到拒绝意图"
                except Exception as e:
                    logger.warning(f"解析JSON响应时出错: {str(e)}，回退到文本匹配")
                
                # 文本匹配回退
                if not agreement_reached and not merchant_rejected:
                    if "接受" in merchant_response.lower() or "同意" in merchant_response.lower() or "达成协议" in merchant_response.lower() or "合作" in merchant_response.lower():
                        agreement_reached = True
                    elif "拒绝" in merchant_response.lower() or "不同意" in merchant_response.lower() or "不接受" in merchant_response.lower() or "无法合作" in merchant_response.lower():
                        merchant_rejected = True
                        rejection_reason = "从回应中检测到拒绝意图"
                
                # 偏向于达成协议，除非明确拒绝
                if negotiation_round == MAX_NEGOTIATION_ROUNDS and not merchant_rejected:
                    agreement_reached = True
                    logger.info(f"谈判达到最大轮数，商家 {merchant_id} 和主播 {streamer_id} 最终达成协议")
                
                # 线程安全地更新谈判结果
                with results_lock:
                    if agreement_reached:
                        logger.info(f"主播 {streamer_id} 和商家 {merchant_id} 最终达成协议，产品 {product_id}")
                        final_terms = _extract_final_terms(merchant_response, initial_offer)
                        
                        # 确保不重复添加
                        if product_id not in negotiation_results[streamer_id]["selected_products"]:
                            negotiation_results[streamer_id]["selected_products"].append(product_id)
                        
                        # 更新协议内容
                        negotiation_results[streamer_id]["agreements"][product_id] = final_terms
                        
                        # 记录谈判结果到CSV
                        data_recorder = get_data_recorder()
                        data_recorder.record_negotiation(
                            round_num=self.current_round,
                            streamer_id=streamer_id,
                            merchant_id=merchant_id,
                            product_id=product_id,
                            commission_rate=final_terms.get("commission_rate", 0.0),
                            discount=final_terms.get("discount", 1.0),
                            is_accepted=True
                        )
                        
                        # 结束谈判
                        break
                    elif merchant_rejected:
                        logger.info(f"商家 {merchant_id} 拒绝了主播 {streamer_id} 对产品 {product_id} 的合作，原因: {rejection_reason}")
                        negotiation_results[streamer_id]["rejected_by"].append(merchant_id)
                        
                        # 记录谈判拒绝到CSV
                        data_recorder = get_data_recorder()
                        data_recorder.record_negotiation(
                            round_num=self.current_round,
                            streamer_id=streamer_id,
                            merchant_id=merchant_id,
                            product_id=product_id,
                            commission_rate=initial_offer.get("commission_rate", 0.0),
                            discount=initial_offer.get("discount", 1.0),
                            is_accepted=False
                        )
                        
                        # 结束谈判
                        break
                    else:
                        logger.debug(f"谈判继续，没有明确的接受或拒绝")
            
            negotiation_round += 1
    
    # 并行处理所有谈判
    # 重新计算可用线程数 - 谈判任务可能更消耗资源
    try:
        # 重新评估系统资源
        available_memory = psutil.virtual_memory().available
        memory_usage_ratio = available_memory / total_memory
        
        # 谈判过程更消耗资源，降低并行度
        if memory_usage_ratio < 0.4:  # 内存压力增大
            max_workers = max(2, cpu_count // 2)
        else:
            max_workers = max(2, min(10, cpu_count))
            
        logger.info(f"使用 {max_workers} 个线程处理谈判阶段")
    except:
        max_workers = 4
        logger.warning("无法获取系统资源信息，使用默认线程数: 4")
    
    # 批量收集谈判任务
    negotiation_tasks = []
    
    for streamer_id, streamer in state["streamers"].items():
        streamer_agent = self.agents.get(f"streamer_{streamer_id}")
        if not streamer_agent:
            continue
        
        logger.info(f"主播 {streamer_id} 开始与商家谈判")
        
        # 初始轮：主播表达合作意向
        logger.info(f"主播 {streamer_id} 表达初步合作意向")
        streamer_decision = streamer_agent.make_decision(state)
        selected_product_ids = streamer_decision.get("selection", [])
        
        if not selected_product_ids:
            logger.warning(f"主播 {streamer_id} 没有表达明确的合作意向")
            continue
        
        # 商品ID -> 商家ID的映射
        product_to_merchant = {}
        for product_id, product in state["products"].items():
            # 确保product_id是字符串类型，防止使用dict作为键
            if isinstance(product_id, dict):
                self.logger.error(f"产品ID不应该是字典类型: {product_id}")
                continue
                
            # 确保merchant_id是字符串类型
            merchant_id = product.get("merchant_id")
            if isinstance(merchant_id, dict):
                self.logger.error(f"merchant_id不应该是字典类型: {merchant_id}")
                continue
                
            # 正常情况下，将product_id映射到merchant_id
            product_to_merchant[str(product_id)] = merchant_id
        
        # 收集谈判任务
        for product_id in selected_product_ids:
            # 确保使用字符串类型的键进行查找
            merchant_id = product_to_merchant.get(str(product_id))
            if not merchant_id:
                logger.warning(f"未找到产品 {product_id} 的商家信息")
                continue
            
            merchant_agent = self.agents.get(f"merchant_{merchant_id}")
            if not merchant_agent:
                logger.warning(f"未找到商家 {merchant_id} 的代理")
                continue
            
            initial_offer = merchant_initial_offers.get(merchant_id, {}).get(product_id, {})
            if not initial_offer:
                logger.warning(f"商家 {merchant_id} 未提供产品 {product_id} 的初始报价")
                continue
            
            # 添加到任务列表
            negotiation_tasks.append((
                streamer_id,
                streamer_agent,
                product_id,
                merchant_id,
                merchant_agent,
                initial_offer
            ))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有谈判任务
        futures = [executor.submit(negotiate_with_merchant, *task) for task in negotiation_tasks]
        
        # 批量处理结果
        import time
        start_time = time.time()
        completed = 0
        total = len(futures)
        
        # 监控并展示进度
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            elapsed = time.time() - start_time
            remaining = (elapsed / completed) * (total - completed) if completed > 0 else 0
            
            if completed % 10 == 0 or completed == total:  # 每10个任务或全部完成时更新一次进度
                logger.info(f"谈判进度: {completed}/{total} ({completed*100/total:.1f}%), "
                           f"已用时间: {elapsed:.1f} 秒, 预计剩余: {remaining:.1f} 秒")
            
            # 获取结果（即使发生异常也不会中断）
            try:
                future.result()
            except Exception as e:
                logger.error(f"执行谈判任务时出错: {str(e)}")
    
    # 3. 应用谈判结果
    for streamer_id, result in negotiation_results.items():
        agreements = result["agreements"]
        if agreements:
            selected_products = list(agreements.keys())
            self.environment.update_streamer_state(
                streamer_id,
                {"selected_products": selected_products}
            )
            
            for product_id, terms in agreements.items():
                self.environment.update_product_state(
                    product_id,
                    {"commission_rate": terms.get("commission_rate", 0.1)}
                )
        else:
            logger.warning(f"主播 {streamer_id} 未成功与任何商家达成协议")
    
    # 4. 检查所有主播是否已选择商品
    for streamer_id, streamer in state["streamers"].items():
        # 获取最新的主播状态
        updated_streamer = self.environment.get_streamer(streamer_id)
        selected_products = updated_streamer.get("selected_products", [])
        
        if not selected_products:
            logger.warning(f"主播 {streamer_id} 未选择任何产品，分配默认产品")
            all_products = list(state["products"].keys())
            
            if all_products:
                random_choice = random.choice(all_products)
                self.environment.update_streamer_state(
                    streamer_id,
                    {"selected_products": [random_choice]}
                )
                logger.info(f"为主播 {streamer_id} 分配默认产品: {random_choice}")
            else:
                logger.error("系统中没有可用产品！")
    
    # 5. 打印谈判阶段的最终状态
    logger.info("商家-主播谈判阶段完成")
    updated_state = self.environment.get_state()
    for streamer_id, streamer in updated_state["streamers"].items():
        selected_products = streamer.get("selected_products", [])
        product_names = []
        
        for product_id in selected_products:
            if product_id in updated_state["products"]:
                product_names.append(updated_state["products"][product_id]["name"])
                
                # 获取主播代理和产品信息
                streamer_agent = self.agents.get(f"streamer_{streamer_id}")
                product_info = updated_state["products"][product_id]
                merchant_id = product_info.get("merchant_id")
                
                # 添加选品记忆
                if streamer_agent:
                    streamer_agent.add_memory("selection", {
                        "round_num": updated_state.get("round_num", 0),
                        "product_id": product_id,
                        "product_name": product_info.get("name", ""),
                        "merchant_id": merchant_id,
                        "commission_rate": product_info.get("commission_rate", 0),
                        "action": "selected_product"
                    }, updated_state.get("round_num", 0))
        
        logger.info(f"主播 {streamer_id} 最终选择的产品: {', '.join(product_names) if product_names else '无'}")
    
    # 6. 保存谈判历史记录
    if hasattr(self, "data_processor") and negotiation_history:
        logger.info("保存谈判历史")
        round_num = state.get("round_num", 0)
        self.data_processor.save_negotiation_history(round_num, negotiation_history)

def _build_negotiation_prompt(agent, negotiation_history, state, product_id, role):
    """
    构建谈判提示
    
    Args:
        agent: 代理（主播或商家）
        negotiation_history: 谈判历史
        state: 当前状态
        product_id: 产品ID
        role: 角色（商家或主播）
        
    Returns:
        str: 谈判提示
    """
    product_info = state["products"].get(product_id, {})
    
    # 将谈判历史格式化为对话形式
    conversation = ""
    for msg in negotiation_history:
        if msg["role"] == "merchant":
            conversation += f"商家: {msg['content']}\n\n"
        else:
            conversation += f"主播: {msg['content']}\n\n"
    
    # 判断当前谈判轮次，根据轮次调整策略
    negotiation_round = len(negotiation_history) // 2
    
    if role == "merchant":
        # 商家的利润计算
        base_price = product_info.get('base_price', 0)
        cost = product_info.get('cost', 0)
        base_profit_margin = (base_price - cost) / base_price if base_price > 0 else 0
        
        # 计算关键商业指标
        base_price = product_info.get('base_price', 0)
        cost = product_info.get('cost', 0)
        stock = product_info.get('stock', 0)
        quality = product_info.get('quality', 0)
        base_profit_margin = (base_price - cost) / base_price if base_price > 0 else 0
        min_profit_margin = base_profit_margin * 0.7
        
        # 根据库存和品质调整谈判策略
        stock_pressure = "高" if stock > 1000 else "中" if stock > 500 else "低"
        quality_level = "高" if quality > 80 else "中" if quality > 60 else "低"
        
        product_name = product_info.get('name', product_id)
        negotiation_round_text = str(negotiation_round + 1)
        
        prompt = f"你是一个聪明的电子商务商家，专注于利润和市场策略。"
        prompt += f"你正在与主播谈判产品 [{product_name}]。"
        prompt += f"这是第 {negotiation_round_text} 轮谈判。\n\n"
        
        prompt += "产品详情:\n"
        prompt += f"- 名称: {product_info.get('name', '')}\n"
        prompt += f"- 原价: {base_price} 元\n"
        prompt += f"- 成本: {cost} 元\n"
        prompt += f"- 质量评分: {quality}（满分100）\n"
        prompt += f"- 当前库存: {stock}（库存压力: {stock_pressure}）\n"
        prompt += f"- 基础利润率: {base_profit_margin:.1%}\n"
        prompt += f"- 最低可接受利润率: {min_profit_margin:.1%}\n"
        
        prompt += "商业评估:\n"
        prompt += f"- 质量水平: {quality_level}，{'适合高端定位' if quality_level == '高' else '性价比市场' if quality_level == '中' else '需要价格优势'}\n"
        prompt += f"- 库存情况: {'需要加速周转' if stock_pressure == '高' else '中等库存' if stock_pressure == '中' else '可以从容谈判'}\n"
        prompt += f"- 利润空间: {'充足' if base_profit_margin > 0.4 else '一般' if base_profit_margin > 0.25 else '有限'}\n"
        
        prompt += "参考指导（非严格限制）:\n"
        prompt += "1. 一般佣金水平: {'12-18%' if quality_level == '高' else '10-15%' if quality_level == '中' else '8-12%'}\n"
        prompt += "2. 常见折扣范围: {'8.5-9.5折' if stock_pressure == '高' else '9-9.5折' if stock_pressure == '中' else '9.5-原价'}\n"
        prompt += "3. 理想利润率: {min_profit_margin:.1%} 或更高\n"
        
        prompt += "谈判策略建议:\n"
        prompt += "1. 利润考虑: 需要考虑利润率，但可以灵活平衡长期利益和短期利润\n"
        prompt += "2. 差异化定价: 针对不同级别的主播采用不同策略\n"
        prompt += "3. 库存因素: {'库存多，可以提供更好条件' if stock_pressure == '高' else '中等库存，保持平衡' if stock_pressure == '中' else '库存有限，可以更强势谈判'}\n"
        prompt += "4. 质量因素: {'高质量，可以坚持更高价格' if quality_level == '高' else '一般质量，价格需要有竞争力' if quality_level == '中' else '低质量，需要价格优势'}\n"
        
        prompt += "这里是谈判历史:\n"
        prompt += f"{conversation}\n\n"
        
        prompt += "现在请对主播的最新回应做出战略回应。注意这是第 {negotiation_round_text} 轮谈判，你需要根据谈判进展灵活调整策略:\n"
        prompt += {'- 第一轮: 展示产品优势，给出初始报价，但保留一些谈判空间' if negotiation_round == 0 else ''}
        prompt += {'- 第二轮: 理性分析对方条件，可以做出一些让步，但不要轻易接受所有条件' if negotiation_round == 1 else ''}
        prompt += {'- 第三轮: 展示诚意但保持理性，可以做出最终让步，但在有充分理由时也可以拒绝不合理要求' if negotiation_round == 2 else ''}
        
        prompt += "\n响应格式要求:\n```json\n{{"
        prompt += "\"response\": \"你的详细回应内容，需要有商业礼貌但坚定，表达你的想法和理由\","
        prompt += "\"adjustment plan\": {{"
        prompt += "\"commission rate\": 0.XX,"
        prompt += "\"price discount\": \"XX折或原价\","
        prompt += "\"other conditions\": [condition1, condition2]}}"
        prompt += ",\"reason\": \"提出这些调整的商业理由的详细解释，可以包括让步或坚持的理由\","
        prompt += "\"agreement\": true/false,"
        prompt += "\"reason for rejection\": \"如果选择不达成协议，请提供具体合理的商业理由，而不仅仅是简单的数字比较\""
        prompt += "}}\n```"
        
        prompt += "\n记住，你需要充分展示自己的价值和专业能力，灵活运用谈判策略。既要争取好的条件，也要考虑合作的长期性和稳定性。不要简单依赖固定公式，而是根据实际情况和对话进展做出判断。"
        
        return prompt
    else:
        streamer_info = {}
        for s_id, streamer in state["streamers"].items():
            if agent.streamer_id == s_id:
                streamer_info = streamer
                break
        
        # 根据主播特性调整谈判策略
        is_top_streamer = streamer_info.get('fans_count', 0) > 500000
        negotiation_strength = "强" if is_top_streamer else "平衡"
        
        streamer_type = ' 知名' if is_top_streamer else ' 潜力'
        product_name = product_info.get('name', product_id)
        negotiation_round_text = str(negotiation_round + 1)
        
        prompt = f"你是一个{streamer_type}的电子商务直播主播，擅长谈判和销售。"
        prompt += f"你正在与商家谈判产品 [{product_name}]。"
        prompt += f"这是第 {negotiation_round_text} 轮谈判。\n\n"
        prompt += "你的信息:\n"
        prompt += f"- 粉丝数量: {streamer_info.get('fans_count', 0):,}\n"
        prompt += f"- 声誉: {streamer_info.get('reputation', 0)}（满分100）\n"
        prompt += f"- 平均转化率: {streamer_info.get('conversion_rate', 0):.1%}\n"
        prompt += f"- 谈判地位: {negotiation_strength}\n\n"
        prompt += "产品详情:\n"
        prompt += f"- 名称: {product_info.get('name', '')}\n"
        prompt += f"- 市场价格: {product_info.get('base_price', 0)} 元\n"
        prompt += f"- 质量评分: {product_info.get('quality', 0)}（满分100）\n"
        prompt += f"- 预计观众兴趣: {'高' if product_info.get('quality', 0) > 80 else '中'}\n\n"
        prompt += "谈判参考目标（非严格限制）:\n"
        prompt += f"- 佣金率参考范围: {'12-20%' if is_top_streamer else '10-15%'}\n"
        prompt += f"- 价格折扣参考范围: {'8-9折' if is_top_streamer else '8.5-9.5折'}\n"
        prompt += "- 其他支持: 如独家优惠券、售后保证、独家款式等。\n"
        prompt += "- 确保产品质量和观众满意度\n\n"
        prompt += "这里是谈判历史:\n"
        prompt += f"{conversation}\n\n"
        prompt += "现在请对商家的最新回应做出战略回应。注意这是第 {negotiation_round_text} 轮谈判，你需要根据谈判进展灵活调整策略:\n"
        prompt += {'- 第一轮: 展示你的价值和影响力，提出初步要求，但保留一些谈判空间' if negotiation_round == 0 else ''}
        prompt += {'- 第二轮: 分析商家的提议，做出一些让步但坚持核心要求，强调合作的互利' if negotiation_round == 1 else ''}
        prompt += {'- 第三轮: 努力达成协议，表达最终立场，可以做出适当妥协，但不要接受明显不利的条件' if negotiation_round == 2 else ''}
        prompt += "\n响应格式要求:\n```json\n{{"
        prompt += "\"response\": \"你的详细回应内容，需要展示谈判技巧和专业性\","
        prompt += "\"my conditions\": {{"
        prompt += "\"commission rate\": 0.XX,"
        prompt += "\"price discount\": \"XX折或原价\","
        prompt += "\"other requirements\": [requirement1, requirement2]}}"
        prompt += ",\"reason\": \"提出这些条件的理由的详细解释，包括你能为商家带来的价值\","
        prompt += "\"accept current conditions\": true/false,"
        prompt += "\"reason for rejection\": \"如果选择不接受，请提供具体合理的解释，而不仅仅是简单的数字比较\""
        prompt += "}}\n```"
        prompt += "\n记住，你需要充分展示自己的价值和专业能力，灵活运用谈判策略。既要争取好的条件，也要考虑合作的长期性和稳定性。不要简单依赖固定公式，而是根据实际情况和对话进展做出判断。"
        
        return prompt

def _extract_final_terms(merchant_response, initial_offer):
    """
    从商家响应中提取最终条款
    
    Args:
        merchant_response: 商家最终响应
        initial_offer: 初始报价
    """
    client = get_llm_client()
    logger = logging.getLogger("simulation")
    final_terms = {}
    
    # 尝试解析JSON格式响应
    try:
        parsed_response = client.parse_json_response(merchant_response)
        if parsed_response:
            # 先尝试从新格式中提取
            if "调整计划" in parsed_response:
                adjustment = parsed_response["调整计划"]
                commission = adjustment.get("commission rate", initial_offer.get("commission", 0.1))
                discount = adjustment.get("price discount", initial_offer.get("discount", "原价"))
                other_terms = adjustment.get("other conditions", [])
                
                final_terms = {
                    "commission": commission,
                    "discount": discount,
                    "other conditions": other_terms,
                    "reason": parsed_response.get("reason", "")
                }
            # 如果是主播视角的回应    
            elif "我的条件" in parsed_response:
                conditions = parsed_response["my conditions"]
                commission = conditions.get("commission rate", initial_offer.get("commission", 0.1))
                discount = conditions.get("price discount", initial_offer.get("discount", "原价"))
                other_terms = conditions.get("other requirements", [])
                
                final_terms = {
                    "commission": commission,
                    "discount": discount,
                    "other conditions": other_terms,
                    "reason": parsed_response.get("reason", "")
                }
            # 回退到旧格式
            else:
                # 寻找关键词，提取数值
                response_text = merchant_response.lower()
                
                # 提取佣金率
                commission_match = re.search(r'佣金(?:率|比)?[：:]\s*(0\.\d+|\d+%|\d+．\d+%|\d+\.\d+%)', response_text)
                if commission_match:
                    commission_str = commission_match.group(1)
                    if '%' in commission_str:
                        commission = float(commission_str.replace('%', '').replace('．', '.')) / 100
                    else:
                        commission = float(commission_str)
                else:
                    commission = initial_offer.get("commission", 0.1)
                
                # 提取折扣
                discount_match = re.search(r'折扣[：:]\s*(\d+(?:\.\d+)?折|\d+(?:\.\d+)?%|原价)', response_text)
                if discount_match:
                    discount_str = discount_match.group(1)
                    if discount_str == "原价":
                        discount = "原价"
                    elif '%' in discount_str:
                        percent = float(discount_str.replace('%', ''))
                        discount = f"{percent/10}折"
                    else:
                        discount = discount_str
                else:
                    discount = initial_offer.get("discount", "原价")
                
                final_terms = {
                    "commission": commission,
                    "discount": discount,
                    "other conditions": [],
                    "reason": "通过文本匹配"
                }
                
        else:
            logger.warning("无法将商家响应解析为JSON格式")
            final_terms = initial_offer.copy()
    except Exception as e:
        logger.warning(f"提取协议条款时出错: {str(e)}")
        final_terms = initial_offer.copy()
    
    return final_terms

def platform_traffic_allocation(self) -> None:
    """
    平台流量分配阶段
    
    Args:
        self: 仿真控制器
    """
    logger = logging.getLogger("simulation")
    logger.info("执行平台流量分配阶段")
    
    # 获取当前状态
    state = self.environment.get_state()
    
    # 平台根据主播信誉、历史GMV等分配流量
    platform_agent = self.agents.get("platform")
    if platform_agent:
        logger.info("平台做出决策")
        decision = platform_agent.make_decision(state)
        platform_agent.apply_decision(decision)

def live_streaming_sales(self) -> None:
    """
    直播带货阶段
    
    Args:
        self: 仿真控制器
    """
    logger = logging.getLogger("simulation")
    logger.info("执行直播带货阶段")
    
    # 获取当前状态
    state = self.environment.get_state()
    round_num = state.get("round_num", 0)
    
    # 初始化各种计数器
    streamer_gmv = {}  # 主播GMV
    streamer_commission = {}  # 主播佣金
    merchant_revenue = {}  # 商家收入
    product_sales = {}  # 产品销量
    total_sales_count = 0  # 总销售数量
    total_gmv = 0  # 总GMV
    total_commission = 0  # 总佣金
    
    # 计算可用CPU核心数和内存情况
    try:
        import os
        import psutil
        
        cpu_count = os.cpu_count() or 4
        available_memory = psutil.virtual_memory().available
        total_memory = psutil.virtual_memory().total
        memory_usage_ratio = available_memory / total_memory
        
        # 根据资源情况决定分批大小和并行度
        if memory_usage_ratio < 0.3:  # 内存紧张
            batch_size = 20
            max_workers = max(2, cpu_count // 2)
        else:
            batch_size = 50
            max_workers = min(cpu_count * 2, 16)
            
        logger.info(f"直播带货阶段使用 {max_workers} 个线程，每批 {batch_size} 个任务")
    except:
        # 默认保守配置
        batch_size = 20
        max_workers = 4
        logger.warning("无法获取系统资源信息，使用默认配置")
    
    # 预先收集所有直播任务
    all_streaming_tasks = []
    
    # 针对每个主播，收集直播任务
    for streamer_id, streamer in state["streamers"].items():
        # 获取主播的流量
        traffic = streamer.get("traffic", 1)  # 默认为1
        
        # 确保流量是数值类型
        if isinstance(traffic, str):
            traffic_map = {"低": 1, "中": 2, "高": 3, "超高": 4}
            traffic_value = traffic_map.get(traffic, 1)
        else:
            traffic_value = int(traffic)
        
        logger.info(f"处理主播 {streamer_id} 的直播，流量值: {traffic_value}")
        
        # 获取主播选择的商品
        selected_products = streamer.get("selected_products", [])
        if not selected_products:
            logger.warning(f"主播 {streamer_id} 没有选择任何商品，跳过直播")
            continue
            
        # 获取主播代理
        streamer_agent = self.agents.get(f"streamer_{streamer_id}")
        if not streamer_agent:
            logger.warning(f"找不到主播 {streamer_id} 的代理，跳过直播")
            continue
            
        # 让主播做出直播决策（话术、促销策略等）
        live_decision = streamer_agent.make_decision(
            state,
            direct_live=True,
            selected_products=selected_products
        )
        
        # 记录主播最新决策
        streamer_agent._record_decision(live_decision)
        
        # 简化为从所有消费者中随机选择部分观看直播
        all_consumers = list(state["consumers"].keys())
        logger.info(f"系统中共有 {len(all_consumers)} 个消费者")
        
        # 根据流量大小，确定消费者数量
        num_consumers = min(len(all_consumers), traffic_value * 2)
        logger.info(f"主播 {streamer_id} 将获得 {num_consumers} 个观众（流量值: {traffic_value}）")
        
        # 随机选择观众
        selected_consumers = random.sample(all_consumers, num_consumers)
        logger.info(f"从 {len(all_consumers)} 个消费者中随机选择 {num_consumers} 个观看主播 {streamer_id} 的直播")
        
        # 为每个商品创建消费者-商品对，模拟消费者可能购买的多个商品
        for consumer_id in selected_consumers:
            for product_id in selected_products:
                # 将任务添加到总任务列表
                all_streaming_tasks.append({
                    "consumer_id": consumer_id,
                    "product_id": product_id,
                    "streamer_id": streamer_id,
                    "live_decision": live_decision
                })
    
    # 使用批处理优化消费者决策
    logger.info(f"共收集 {len(all_streaming_tasks)} 个直播消费者任务")
    
    def process_consumer_batch(batch):
        """处理一批消费者决策"""
        results = []
        for task in batch:
            consumer_id = task["consumer_id"]
            product_id = task["product_id"]
            streamer_id = task["streamer_id"]
            live_decision = task["live_decision"]
            
            try:
                # 获取消费者代理
                consumer_agent = self.agents.get(f"consumer_{consumer_id}")
                if not consumer_agent:
                    continue
                
                # 获取商品和主播信息
                product = state["products"].get(product_id, {})
                streamer = state["streamers"].get(streamer_id, {})
                
                # 让消费者做出购买决策
                consumer_decision = consumer_agent.make_decision(
                    state,
                    streamer_id=streamer_id,
                    product_id=product_id,
                    direct_live_data=live_decision
                )
                
                # 应用消费者决策
                consumer_agent.apply_decision(
                    consumer_decision,
                    streamer_id=streamer_id,
                    product_id=product_id
                )
                
                # 记录结果
                will_purchase = consumer_decision.get("购买决定", False)
                if will_purchase:
                    results.append({
                        "consumer_id": consumer_id,
                        "streamer_id": streamer_id,
                        "product_id": product_id,
                        "success": True
                    })
                
            except Exception as e:
                logger.error(f"消费者 {consumer_id} 决策出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        return results
    
    from concurrent.futures import ThreadPoolExecutor
    import time
    
    # 分批处理所有消费者决策
    all_results = []
    processed_count = 0
    total_count = len(all_streaming_tasks)
    start_time = time.time()
    
    # 实现批处理逻辑
    for i in range(0, total_count, batch_size):
        # 获取当前批次
        current_batch = all_streaming_tasks[i:i + batch_size]
        batch_number = i // batch_size + 1
        total_batches = (total_count + batch_size - 1) // batch_size
        
        logger.info(f"处理批次 {batch_number}/{total_batches}, 任务数: {len(current_batch)}")
        
        # 计算每个线程处理的任务数
        tasks_per_thread = max(1, len(current_batch) // max_workers)
        thread_batches = []
        
        for j in range(0, len(current_batch), tasks_per_thread):
            thread_batches.append(current_batch[j:j + tasks_per_thread])
        
        # 并行处理当前批次
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(executor.map(process_consumer_batch, thread_batches))
            
            # 合并结果
            for result_list in batch_results:
                all_results.extend(result_list)
        
        # 更新进度
        processed_count += len(current_batch)
        elapsed = time.time() - start_time
        remaining = (elapsed / processed_count) * (total_count - processed_count) if processed_count > 0 else 0
        
        logger.info(f"完成 {processed_count}/{total_count} 个任务 ({processed_count * 100 / total_count:.1f}%), "
                   f"已用时间: {elapsed:.1f}秒, 预计剩余: {remaining:.1f}秒")
    
    # 完成所有处理
    logger.info(f"直播带货阶段完成，成功购买记录: {len(all_results)} 条")
    
    # 更新环境中的交易数据
    # 直接使用环境中已更新的指标
    metrics = self.environment.current_state["metrics"]
    
    # 更新指标
    for streamer_id, gmv in metrics["streamer_gmv"].items():
        streamer_gmv[streamer_id] = gmv
    
    for merchant_id, revenue in metrics["merchant_revenue"].items():
        merchant_revenue[merchant_id] = revenue
    
    for product_id, sales in metrics["product_sales"].items():
        product_sales[product_id] = sales
    
    total_sales_count = metrics["total_sales_count"]
    total_gmv = metrics["total_gmv"]
    total_commission = metrics["total_commission"]
    
    logger.info(f"直播带货阶段完成，总GMV: {total_gmv}, 总佣金: {total_commission}, 总销售: {total_sales_count}")

def transaction_feedback(self) -> None:
    """
    交易与反馈阶段
    
    Args:
        self: 仿真控制器
    """
    logger = logging.getLogger("simulation")
    logger.info("执行交易与反馈阶段")
    
    # 获取当前状态
    state = self.environment.get_state()
    
    # 此阶段主要是环境自动更新，如更新主播信誉度、更新商家库存等
    # 大部分逻辑已在ConsumerAgent.apply_decision中实现
    
    # 1. 更新主播历史GMV
    for streamer_id, streamer in state["streamers"].items():
        current_gmv = state["metrics"]["streamer_gmv"].get(streamer_id, 0)
        history_gmv = streamer.get("history_gmv", []).copy()
        history_gmv.append(current_gmv)
        
        self.environment.update_streamer_state(
            streamer_id, {"history_gmv": history_gmv}
        )
    
    # 2. 更新商家历史销售额
    for merchant_id, merchant in state["merchants"].items():
        current_revenue = state["metrics"]["merchant_revenue"].get(merchant_id, 0)
        history_revenue = merchant.get("history_revenue", []).copy()
        history_revenue.append(current_revenue)
        
        self.environment.update_merchant_state(
            merchant_id, {"history_revenue": history_revenue}
        )

def print_agent_memories(self, round_num: int) -> None:
    """
    打印所有代理的记忆摘要
    
    Args:
        self: 仿真控制器
        round_num: 回合数
    """
    # 仅在日志中记录一次标题
    logger = logging.getLogger("simulation")
    logger.info(f"======== 第 {round_num + 1} 回合所有代理记忆摘要 ========")
    
    # 使用print而不是logger.info打印到标准输出
    print(f"\n======== 第 {round_num + 1} 回合所有代理记忆摘要 ========")
    
    # 先按类型分组代理
    agents_by_type = {
        "platform": [],
        "streamer": [],
        "merchant": [],
        "consumer": []
    }
    
    for agent_key, agent in self.agents.items():
        agent_type = agent.agent_type
        if agent_type in agents_by_type:
            agents_by_type[agent_type].append(agent)
    
    # 打印平台记忆
    if agents_by_type["platform"]:
        print("======== 平台记忆 ========")
        has_memories = False
        for agent in agents_by_type["platform"]:
            summary = agent.get_memory_summary(round_num)
            if summary:
                print(f"平台 {agent.agent_id} 记忆摘要:\n{summary}")
                has_memories = True
        if not has_memories:
            print("平台记忆为空")
    
    # 打印主播记忆
    if agents_by_type["streamer"]:
        print("======== 主播记忆 ========")
        has_memories = False
        for agent in agents_by_type["streamer"]:
            summary = agent.get_memory_summary(round_num)
            if summary:
                print(f"主播 {agent.agent_id} 记忆摘要:\n{summary}")
                has_memories = True
        if not has_memories:
            print("主播记忆为空 - 请检查主播代理是否正确添加了记忆")
    
    # 打印商家记忆
    if agents_by_type["merchant"]:
        print("======== 商家记忆 ========")
        has_memories = False
        for agent in agents_by_type["merchant"]:
            summary = agent.get_memory_summary(round_num)
            if summary:
                print(f"商家 {agent.agent_id} 记忆摘要:\n{summary}")
                has_memories = True
        if not has_memories:
            print("商家记忆为空")
    
    # 打印消费者记忆
    if agents_by_type["consumer"]:
        print("======== 消费者记忆 ========")
        has_memories = False
        for agent in agents_by_type["consumer"]:
            summary = agent.get_memory_summary(round_num)
            if summary:
                print(f"消费者 {agent.agent_id} 记忆摘要:\n{summary}")
                has_memories = True
        if not has_memories:
            print("消费者记忆为空 - 请检查消费者代理是否正确添加了记忆")

def main():
    """
    主函数
    """
    print("开始main函数...")
    try:
        # 解析命令行参数
        args = parse_args()
        print("命令行参数解析成功")
        
        # 设置仿真系统
        simulator = setup_simulation(args)
        print("仿真系统设置成功")
        
        # 更新仿真控制器方法
        update_simulator_methods(simulator)
        print("仿真控制器方法更新成功")
        
        # 运行仿真
        print("开始仿真...")
        result = simulator.run_simulation()
        print("仿真完成")
        
        # 打印结果摘要
        print("\n=================")
        print("仿真结束! 结果摘要:")
        print(f"总回合数: {result['total_rounds']}")
        print(f"总GMV: {result['final_metrics']['total_gmv']}")
        print(f"总佣金: {result['final_metrics']['total_commission']}")
        print(f"总销售: {result['final_metrics']['total_sales_count']}")
        print("=================\n")
        
        # 打印所有代理的最终记忆摘要
        simulator.print_agent_memories(result['total_rounds'] - 1)
    except Exception as e:
        print(f"运行时出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return 0

if __name__ == "__main__":
    print("程序启动...")
    sys.exit(main())