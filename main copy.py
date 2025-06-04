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
print("start loading modules...")

from config import SIMULATION_ROUNDS, SAVE_RESULTS
print("load config successfully")

from utils.logger import setup_logger
print("load logger successfully")

from utils.llm_utils import get_llm_client, set_llm_client_type
print("load llm_utils successfully")

from simulation.environment import Environment
print("load Environment successfully")

from simulation.simulator import Simulator
print("load Simulator successfully")

try:
    from agents.platform_agent import PlatformAgent
    print("load PlatformAgent successfully")
    
    from agents.streamer_agent import StreamerAgent, TopStreamerAgent, RegularStreamerAgent
    print("load StreamerAgent successfully")
    
    from agents.merchant_agent import MerchantAgent
    print("load MerchantAgent successfully")
    
    from agents.consumer_agent import ConsumerAgent
    print("load ConsumerAgent successfully")
except Exception as e:
    print(f"load Agent error: {str(e)}")

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
    parse command line arguments
    
    Returns:
        argparse.Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser(description="e-commerce live streaming supply chain simulation system")
    parser.add_argument(
        "--rounds", type=int, default=SIMULATION_ROUNDS,
        help=f"simulation rounds, default {SIMULATION_ROUNDS}"
    )
    parser.add_argument(
        "--save", action="store_true", default=SAVE_RESULTS,
        help="whether to save results"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="log level, default INFO"
    )
    
    # 创建互斥组，确保--use-openai和--use-ollama不会同时使用
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument(
        "--use-openai", action="store_true", default=False,
        help="use OpenAI API for LLM calling"
    )
    llm_group.add_argument(
        "--use-ollama", action="store_true", default=False,
        help="use Ollama for LLM calling (default behavior)"
    )
    parser.add_argument(
        "--disable-reuse", action="store_true", default=False,
        help="disable DataProcessor instance reuse, create a new data processor instance each time"
    )
    parser.add_argument(
        "--disable-cache", action="store_true", default=False,
        help="disable LLM response cache, call LLM again each time"
    )
    
    return parser.parse_args()

def setup_simulation(args) -> Simulator:
    """
    setup simulation system
    
    Args:
        args: command line arguments
        
    Returns:
        Simulator: simulation instance
    """
    logger = logging.getLogger("simulation")
    logger.info("setup simulation system")
    
    # 导入DataProcessor的get_data_processor函数
    from utils.data_processor import get_data_processor
    
    # 为了兼容性，记录--disable-reuse参数的状态，但不影响全局实例的使用
    if args.disable_reuse:
        logger.info("note: --disable-reuse parameter is ignored, now always use the global DataProcessor instance")
    
    # 初始化全局DataProcessor实例，确保所有组件都使用相同的实例
    data_processor = get_data_processor()
    logger.info(f"use the global DataProcessor directory: {data_processor.simulation_dir}")
    
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
                logger.info(f"cache directory has been cleared: {cache_dir}")
            except Exception as e:
                logger.warning(f"failed to clear cache directory: {str(e)}")
        
        logger.info("LLM response cache has been completely disabled, LLM will be called again each time")
    
    # 设置全局客户端类型 (--use-ollama时use_openai为False)
    use_openai = args.use_openai
    set_llm_client_type(use_openai)
    if use_openai:
        logger.info("use OpenAI API for LLM calling")
    else:
        logger.info("use Ollama for LLM calling")
    
    # 创建仿真控制器，但禁用自动初始化
    print("start creating Simulator...")
    simulator = Simulator(rounds=args.rounds, save_results=args.save, auto_init=False)
    print("Simulator created successfully")
    
    # 初始化Agent
    print("start initializing Agent...")
    init_agents(simulator)
    print("Agent initialized successfully")
    
    return simulator

def init_agents(simulator: Simulator) -> None:
    """
    initialize all agents, use parallel processing to improve efficiency
    
    Args:
        simulator: simulation controller
    """
    logger = logging.getLogger("simulation")
    logger.info("initialize all agents")
    
    # 创建一个初始化单个agent的辅助函数
    def init_single_agent(agent_type, agent_id, environment):
        """
        initialize a single agent
        
        Args:
            agent_type: Agent type
            agent_id: Agent ID
            environment: simulation environment
            
        Returns:
            BaseAgent: initialized agent instance
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
            logger.info(f"{agent_type.capitalize()} Agent {agent_id} initialized successfully, time cost: {(end_time - start_time):.2f} seconds")
        except Exception as e:
            logger.error(f"error initializing {agent_type} Agent {agent_id}: {str(e)}")
            raise
        
        return agent
    
    try:
        # 准备所有需要初始化的agent任务
        init_tasks = []
        
        # 添加平台Agent初始化任务
        print("prepare to initialize platform agent...")
        init_tasks.append(("platform", "platform", simulator.environment))
        
        # 添加主播Agent初始化任务
        print("prepare to initialize streamer agent...")
        streamers = simulator.environment.get_streamers()
        for streamer_id in streamers:
            init_tasks.append(("streamer", streamer_id, simulator.environment))
        
        # 添加商家Agent初始化任务
        print("prepare to initialize merchant agent...")
        merchants = simulator.environment.get_merchants()
        for merchant_id in merchants:
            init_tasks.append(("merchant", merchant_id, simulator.environment))
        
        # 添加消费者Agent初始化任务
        print("prepare to initialize consumer agent...")
        consumers = simulator.environment.get_consumers()
        for consumer_id in consumers:
            init_tasks.append(("consumer", consumer_id, simulator.environment))
        
        # 决定线程池大小：通常IO密集型任务可以使用更多线程
        # 使用min函数确保不会创建过多线程
        max_workers = min(32, len(init_tasks))
        print(f"use {max_workers} threads to initialize {len(init_tasks)} agents...")
        
        # 使用线程池并行初始化agent
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
                            logger.info("platform agent initialized successfully")
                        elif agent_type == "streamer":
                            simulator.add_streamer_agent(agent_id, agent)
                            logger.info(f"streamer agent {agent_id} initialized successfully")
                        elif agent_type == "merchant":
                            simulator.add_merchant_agent(agent_id, agent)
                            logger.info(f"merchant agent {agent_id} initialized successfully")
                        elif agent_type == "consumer":
                            simulator.add_consumer_agent(agent_id, agent)
                            logger.info(f"consumer agent {agent_id} initialized successfully")
                except Exception as e:
                    logger.error(f"error processing agent initialization result: {e}")
        
        print(f"all agents initialized successfully, {len(init_tasks)} agents in total")
        logger.info(f"initialized {len(init_tasks)} agents")
    except Exception as e:
        logger.error(f"error initializing agents: {e}")
        raise

def update_simulator_methods(simulator: Simulator) -> None:
    """
    update simulator methods, replace empty methods with actual implementations
    
    Args:
        simulator: simulation controller
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
        
    # 添加打印Agent记忆方法
    if not hasattr(simulator, 'print_agent_memories'):
        simulator.print_agent_memories = print_agent_memories.__get__(simulator, Simulator)

def merchant_streamer_negotiation(self) -> None:
    """
    merchant-streamer negotiation stage - multi-round dialogue version (parallel optimization)
    
    Args:
        self: simulation controller
    """
    # 导入所需模块
    import time
    import random
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from threading import Lock
    
    logger = logging.getLogger("simulation")
    logger.info("execute merchant-streamer negotiation stage (parallel processing)")
    
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
            max_workers = max(2, cpu_count // 2)  # reduce the number of threads
            logger.info(f"high memory usage, reduce the number of threads to {max_workers}")
        elif memory_usage_ratio > 0.7:  # 可用内存超过70%
            max_workers = min(32, cpu_count * 2)  # 可以使用更多线程
            logger.info(f"内存充足，增加线程数至{max_workers}")
        else:
            max_workers = base_workers
            logger.info(f"use standard number of threads: {max_workers}")
        
        # 确保线程数不超过任务数
        max_workers = min(max_workers, len(merchant_tasks) * 2)
    except:
        # 如果无法获取系统信息，使用保守的默认值
        max_workers = 4
        logger.warning("failed to get system resource information, use default number of threads: 4")
    
    logger.info(f"use {max_workers} threads to process {len(merchant_tasks)} tasks in merchant initial offer stage")
    
    # 使用线程池批量处理商家初始报价
    from concurrent.futures import ThreadPoolExecutor
    
    def process_merchant_offers(task):
        merchant_id, merchant_agent = task
        try:
            logger.info(f"merchant {merchant_id} set initial offer")
            return merchant_id, merchant_agent.make_decision(state)
        except Exception as e:
            logger.error(f"error processing merchant {merchant_id} initial offer: {str(e)}")
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
        merchant_message = f"I am merchant {merchant_id}, for product [{product_name}], my initial conditions are: commission rate {initial_offer.get('commission_rate', 0.1)}, price {initial_offer.get('discount', 'original price')}."
        
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
            
            logger.info(f"streamer {streamer_id} response: {streamer_response[:50]}...")
            
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
            
            logger.info(f"merchant {merchant_id} response: {merchant_response[:50]}...")
            
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
                        if "whether the agreement is reached" in parsed_response:
                            agreement_reached = parsed_response["whether the agreement is reached"]
                            
                            # 如果拒绝，获取拒绝原因
                            if not agreement_reached and "if the reason for rejection" in parsed_response:
                                rejection_reason = parsed_response["if the reason for rejection"]
                                merchant_rejected = True
                        
                        # 主播视角
                        elif "whether to accept" in parsed_response or "whether to accept the current conditions" in parsed_response:
                            agreement_reached = parsed_response.get("whether to accept", parsed_response.get("whether to accept the current conditions", False))
                            
                            # 如果拒绝，获取拒绝原因
                            if not agreement_reached and "if the reason for rejection" in parsed_response:
                                rejection_reason = parsed_response["if the reason for rejection"]
                                merchant_rejected = True
                        
                        # 分析回应文本寻找接受/拒绝信号
                        response_text = parsed_response.get("response", "")
                        if response_text and isinstance(response_text, str):
                            if "accept" in response_text.lower() or "agree" in response_text.lower() or "reach the agreement" in response_text.lower() or "cooperate" in response_text.lower():
                                agreement_reached = True
                            elif "reject" in response_text.lower() or "disagree" in response_text.lower() or "not accept" in response_text.lower() or "cannot cooperate" in response_text.lower():
                                merchant_rejected = True
                                if not rejection_reason:
                                    rejection_reason = "detected rejection intent from the response"
                except Exception as e:
                    logger.warning(f"error parsing JSON response: {str(e)}, fallback to text matching")
                
                # 文本匹配回退
                if not agreement_reached and not merchant_rejected:
                    if "accept" in merchant_response.lower() or "agree" in merchant_response.lower() or "reach the agreement" in merchant_response.lower() or "cooperate" in merchant_response.lower():
                        agreement_reached = True
                    elif "reject" in merchant_response.lower() or "disagree" in merchant_response.lower() or "not accept" in merchant_response.lower() or "cannot cooperate" in merchant_response.lower():
                        merchant_rejected = True
                        rejection_reason = "detected rejection intent from the response"
                
                # 偏向于达成协议，除非明确拒绝
                if negotiation_round == MAX_NEGOTIATION_ROUNDS and not merchant_rejected:
                    agreement_reached = True
                    logger.info(f"negotiation reached the maximum rounds, merchant {merchant_id} and streamer {streamer_id} finally reach an agreement")
                
                # 线程安全地更新谈判结果
                with results_lock:
                    if agreement_reached:
                        logger.info(f"streamer {streamer_id} and merchant {merchant_id} finally reach an agreement for product {product_id}")
                        final_terms = _extract_final_terms(merchant_response, initial_offer)
                        
                        # 确保不重复添加
                        if product_id not in negotiation_results[streamer_id]["selected_products"]:
                            negotiation_results[streamer_id]["selected_products"].append(product_id)
                        
                        # 更新协议内容
                        negotiation_results[streamer_id]["agreements"][product_id] = final_terms
                        
                        # 结束谈判
                        break
                    elif merchant_rejected:
                        logger.info(f"merchant {merchant_id} rejected the cooperation of streamer {streamer_id} for product {product_id}, reason: {rejection_reason}")
                        negotiation_results[streamer_id]["rejected_by"].append(merchant_id)
                        
                        # 结束谈判
                        break
                    else:
                        logger.debug(f"negotiation continues, no clear acceptance or rejection")
            
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
            
        logger.info(f"use {max_workers} threads to process negotiation stage")
    except:
        max_workers = 4
        logger.warning("failed to get system resource information, use default number of threads: 4")
    
    # 批量收集谈判任务
    negotiation_tasks = []
    
    for streamer_id, streamer in state["streamers"].items():
        streamer_agent = self.agents.get(f"streamer_{streamer_id}")
        if not streamer_agent:
            continue
        
        logger.info(f"streamer {streamer_id} starts negotiation with merchants")
        
        # 初始轮：主播表达合作意向
        logger.info(f"streamer {streamer_id} expresses initial cooperation intent")
        streamer_decision = streamer_agent.make_decision(state)
        selected_product_ids = streamer_decision.get("selection", [])
        
        if not selected_product_ids:
            logger.warning(f"streamer {streamer_id} does not express clear cooperation intent")
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
                self.logger.error(f"merchant_id should not be a dictionary: {merchant_id}")
                continue
                
            # 正常情况下，将product_id映射到merchant_id
            product_to_merchant[str(product_id)] = merchant_id
        
        # 收集谈判任务
        for product_id in selected_product_ids:
            # 确保使用字符串类型的键进行查找
            merchant_id = product_to_merchant.get(str(product_id))
            if not merchant_id:
                logger.warning(f"no merchant information found for product {product_id}")
                continue
            
            merchant_agent = self.agents.get(f"merchant_{merchant_id}")
            if not merchant_agent:
                logger.warning(f"no merchant agent found for merchant {merchant_id}")
                continue
            
            initial_offer = merchant_initial_offers.get(merchant_id, {}).get(product_id, {})
            if not initial_offer:
                logger.warning(f"merchant {merchant_id} does not provide initial offer for product {product_id}")
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
                logger.info(f"negotiation progress: {completed}/{total} ({completed*100/total:.1f}%), "
                           f"elapsed time: {elapsed:.1f} seconds, remaining: {remaining:.1f} seconds")
            
            # 获取结果（即使发生异常也不会中断）
            try:
                future.result()
            except Exception as e:
                logger.error(f"error executing negotiation task: {str(e)}")
    
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
            logger.warning(f"streamer {streamer_id} did not successfully reach an agreement with any merchants")
    
    # 4. 检查所有主播是否已选择商品
    for streamer_id, streamer in state["streamers"].items():
        # 获取最新的主播状态
        updated_streamer = self.environment.get_streamer(streamer_id)
        selected_products = updated_streamer.get("selected_products", [])
        
        if not selected_products:
            logger.warning(f"streamer {streamer_id} did not select any products, allocate default products")
            all_products = list(state["products"].keys())
            
            if all_products:
                random_choice = random.choice(all_products)
                self.environment.update_streamer_state(
                    streamer_id,
                    {"selected_products": [random_choice]}
                )
                logger.info(f"allocate default product for streamer {streamer_id}: {random_choice}")
            else:
                logger.error("no available products in the system!")
    
    # 5. 打印谈判阶段的最终状态
    logger.info("merchant-streamer negotiation stage completed")
    updated_state = self.environment.get_state()
    for streamer_id, streamer in updated_state["streamers"].items():
        selected_products = streamer.get("selected_products", [])
        product_names = []
        
        for product_id in selected_products:
            if product_id in updated_state["products"]:
                product_names.append(updated_state["products"][product_id]["name"])
                
                # 获取主播agent和产品信息
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
        
        logger.info(f"streamer {streamer_id} finally selected products: {', '.join(product_names) if product_names else 'none'}")
    
    # 6. 保存谈判历史记录
    if hasattr(self, "data_processor") and negotiation_history:
        logger.info("save negotiation history")
        round_num = state.get("round_num", 0)
        self.data_processor.save_negotiation_history(round_num, negotiation_history)

def _build_negotiation_prompt(agent, negotiation_history, state, product_id, role):
    """
    build negotiation prompt
    
    Args:
        agent: agent (streamer or merchant)
        negotiation_history: negotiation history
        state: current state
        product_id: product ID
        role: role (merchant or streamer)
        
    Returns:
        str: negotiation prompt
    """
    product_info = state["products"].get(product_id, {})
    
    # 将谈判历史格式化为对话形式
    conversation = ""
    for msg in negotiation_history:
        if msg["role"] == "merchant":
            conversation += f"merchant: {msg['content']}\n\n"
        else:
            conversation += f"streamer: {msg['content']}\n\n"
    
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
        stock_pressure = "high" if stock > 1000 else "medium" if stock > 500 else "low"
        quality_level = "high" if quality > 80 else "medium" if quality > 60 else "low"
        
        product_name = product_info.get('name', product_id)
        negotiation_round_text = str(negotiation_round+1)
        
        prompt = f"You are a smart e-commerce merchant, focusing on profit and market strategy. "
        prompt += f"You are negotiating with a streamer about the product [{product_name}]. "
        prompt += f"This is the {negotiation_round_text} round of negotiation.\n\n"
        
        prompt += "product details:\n"
        prompt += f"- name: {product_info.get('name', '')}\n"
        prompt += f"- original price: {base_price} yuan\n"
        prompt += f"- cost: {cost} yuan\n"
        prompt += f"- quality score: {quality} (out of 100)\n"
        prompt += f"- current stock: {stock} (stock pressure: {stock_pressure})\n"
        prompt += f"- base profit margin: {base_profit_margin:.1%}\n"
        prompt += f"- minimum acceptable profit margin: {min_profit_margin:.1%}\n"
        
        prompt += "business assessment:\n"
        prompt += f"- quality level: {quality_level}, {'suitable for high-end positioning' if quality_level == 'high' else 'cost-effective market' if quality_level == 'medium' else 'needs price advantage'}\n"
        prompt += f"- stock situation: {'needs to accelerate turnover' if stock_pressure == 'high' else 'medium stock' if stock_pressure == 'medium' else 'can从容谈判'}\n"
        prompt += f"- profit space: {'sufficient' if base_profit_margin > 0.4 else 'general' if base_profit_margin > 0.25 else 'limited'}\n"
        
        prompt += "reference guidance (not strict limits):\n"
        prompt += "1. general commission level: {'12-18%' if quality_level == 'high' else '10-15%' if quality_level == 'medium' else '8-12%'}\n"
        prompt += "2. common discount range: {'8.5-9.5折' if stock_pressure == 'high' else '9-9.5折' if stock_pressure == 'medium' else '9.5-原价'}\n"
        prompt += "3. ideal profit margin: {min_profit_margin:.1%} or more\n"
        
        prompt += "negotiation strategy suggestions:\n"
        prompt += "1. profit consideration: need to consider profit margin, but can flexibly balance long-term benefits and short-term profits\n"
        prompt += "2. differentiated pricing: different strategies for different levels of streamers\n"
        prompt += "3. stock factor: {'more stock, can provide better conditions' if stock_pressure == 'high' else 'medium stock, maintain balance' if stock_pressure == 'medium' else 'limited stock, can be more强势谈判'}\n"
        prompt += "4. quality factor: {'high quality, can insist on higher price' if quality_level == 'high' else 'average quality, price needs to be competitive' if quality_level == 'medium' else 'low quality, needs price advantage'}\n"
        
        prompt += "here is the negotiation history:\n"
        prompt += f"{conversation}\n\n"
        
        prompt += "now please make a strategic response to the latest response from the streamer. note that this is the {negotiation_round_text} round of negotiation, you need to adjust the strategy flexibly based on the negotiation progress:\n"
        prompt += {'- the first round: show the advantages of the product, give the initial offer, but keep some negotiation space' if negotiation_round == 0 else ''}
        prompt += {'- the second round: rational analysis of the other party\'s conditions, can make some concessions but do not easily accept all conditions' if negotiation_round == 1 else ''}
        prompt += {'- the third round: show sincerity but stay rational, can make the final concession, but can also refuse unreasonable requests when there is sufficient reason' if negotiation_round == 2 else ''}
        
        prompt += "\nresponse format requirements:\n```json\n{{"
        prompt += "\"response\": \"your detailed response content, need to have business politeness but firm, express your ideas and reasons\","
        prompt += "\"adjustment plan\": {{"
        prompt += "\"commission rate\": 0.XX,"
        prompt += "\"price discount\": \"XX discount or original price\","
        prompt += "\"other conditions\": [condition1, condition2]}}"
        prompt += ",\"reason\": \"detailed explanation of the commercial reasons for making these adjustments, can include concessions or reasons for坚持\","
        prompt += "\"agreement\": true/false,"
        prompt += "\"reason for rejection\": \"if you choose not to reach an agreement, please provide specific and reasonable commercial reasons, not just a simple numerical comparison\""
        prompt += "}}\n```"
        
        prompt += "\nremember, you need to充分展示自己的价值和专业能力,灵活运用谈判策略。既要争取好的条件,也要考虑合作的长期性和稳定性。不要简单依赖固定公式,而是根据实际情况和对话进展做出判断。"
        
        return prompt
    else:
        streamer_info = {}
        for s_id, streamer in state["streamers"].items():
            if agent.streamer_id == s_id:
                streamer_info = streamer
                break
        
        # 根据主播特性调整谈判策略
        is_top_streamer = streamer_info.get('fans_count', 0) > 500000
        negotiation_strength = "strong" if is_top_streamer else "balanced"
        
        streamer_type = ' well-known' if is_top_streamer else ' potential'
        product_name = product_info.get('name', product_id)
        negotiation_round_text = str(negotiation_round+1)
        
        prompt = f"you are a{streamer_type} e-commerce live streaming anchor, good at negotiation and sales. "
        prompt += f"you are negotiating with a merchant about the product [{product_name}]. "
        prompt += f"this is the {negotiation_round_text} round of negotiation.\n\n"
        prompt += "your information:\n"
        prompt += f"- fans count: {streamer_info.get('fans_count', 0):,}\n"
        prompt += f"- reputation: {streamer_info.get('reputation', 0)} (out of 100)\n"
        prompt += f"- average conversion rate: {streamer_info.get('conversion_rate', 0):.1%}\n"
        prompt += f"- negotiation position: {negotiation_strength}\n\n"
        prompt += "product details:\n"
        prompt += f"- name: {product_info.get('name', '')}\n"
        prompt += f"- market price: {product_info.get('base_price', 0)} yuan\n"
        prompt += f"- quality score: {product_info.get('quality', 0)} (out of 100)\n"
        prompt += f"- estimated audience interest: {'high' if product_info.get('quality', 0) > 80 else 'medium'}\n\n"
        prompt += "reference target for negotiation (not strict limits):\n"
        prompt += f"- commission rate reference range: {'12-20%' if is_top_streamer else '10-15%'}\n"
        prompt += f"- price discount reference range: {'8-9折' if is_top_streamer else '8.5-9.5折'}\n"
        prompt += "- other support: such as exclusive coupons, after-sales guarantee, exclusive styles, etc.\n"
        prompt += "- ensure product quality and audience satisfaction\n\n"
        prompt += "here is the negotiation history:\n"
        prompt += f"{conversation}\n\n"
        prompt += "now please make a strategic response to the latest response from the merchant. note that this is the {negotiation_round_text} round of negotiation, you need to adjust the strategy flexibly based on the negotiation progress:\n"
        prompt += {'- the first round: show your value and influence, make the initial request, but keep some negotiation space' if negotiation_round == 0 else ''}
        prompt += {'- the second round: analyze the merchant\'s proposal, make some concessions but stick to the core demands, emphasize the mutual benefits of cooperation' if negotiation_round == 1 else ''}
        prompt += {'- the third round: strive to reach an agreement, express the final stance, can make appropriate compromises but do not accept conditions that are obviously unfavorable' if negotiation_round == 2 else ''}
        prompt += "\nresponse format requirements:\n```json\n{{"
        prompt += "\"response\": \"your detailed response content, need to show negotiation skills and professionalism\","
        prompt += "\"my conditions\": {{"
        prompt += "\"commission rate\": 0.XX,"
        prompt += "\"price discount\": \"XX discount or original price\","
        prompt += "\"other requirements\": [requirement1, requirement2]}}"
        prompt += ",\"reason\": \"detailed explanation of the reasons for proposing these conditions, including the value you can bring to the merchant\","
        prompt += "\"accept current conditions\": true/false,"
        prompt += "\"reason for rejection\": \"if you choose not to accept, please provide specific and reasonable explanations, not just a simple numerical comparison\""
        prompt += "}}\n```"
        prompt += "\nremember, you need to充分展示自己的价值和专业能力,灵活运用谈判策略。既要争取好的条件,也要考虑合作的长期性和稳定性。不要简单依赖固定公式,而是根据实际情况和对话进展做出判断。"
        
        return prompt

def _extract_final_terms(merchant_response, initial_offer):
    """
    extract final terms from merchant response
    
    Args:
        merchant_response: merchant final response
        initial_offer: initial offer
    """
    client = get_llm_client()
    logger = logging.getLogger("simulation")
    final_terms = {}
    
    # 尝试解析JSON格式响应
    try:
        parsed_response = client.parse_json_response(merchant_response)
        if parsed_response:
            # 先尝试从新格式中提取
            if "adjustment plan" in parsed_response:
                adjustment = parsed_response["adjustment plan"]
                commission = adjustment.get("commission rate", initial_offer.get("commission", 0.1))
                discount = adjustment.get("price discount", initial_offer.get("discount", "original price"))
                other_terms = adjustment.get("other conditions", [])
                
                final_terms = {
                    "commission": commission,
                    "discount": discount,
                    "other conditions": other_terms,
                    "reason": parsed_response.get("reason", "")
                }
            # 如果是主播视角的回应    
            elif "my conditions" in parsed_response:
                conditions = parsed_response["my conditions"]
                commission = conditions.get("commission rate", initial_offer.get("commission", 0.1))
                discount = conditions.get("price discount", initial_offer.get("discount", "original price"))
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
                commission_match = re.search(r'commission(?:rate|ratio)?[：:]\s*(0\.\d+|\d+%|\d+．\d+%|\d+\.\d+%)', response_text)
                if commission_match:
                    commission_str = commission_match.group(1)
                    if '%' in commission_str:
                        commission = float(commission_str.replace('%', '').replace('．', '.')) / 100
                    else:
                        commission = float(commission_str)
                else:
                    commission = initial_offer.get("commission", 0.1)
                
                # 提取折扣
                discount_match = re.search(r'discount[：:]\s*(\d+(?:\.\d+)?discount|\d+(?:\.\d+)?%|original price)', response_text)
                if discount_match:
                    discount_str = discount_match.group(1)
                    if discount_str == "original price":
                        discount = "original price"
                    elif '%' in discount_str:
                        percent = float(discount_str.replace('%', ''))
                        discount = f"{percent/10}discount"
                    else:
                        discount = discount_str
                else:
                    discount = initial_offer.get("discount", "original price")
                
                final_terms = {
                    "commission": commission,
                    "discount": discount,
                    "other conditions": [],
                    "reason": "by text matching"
                }
                
        else:
            logger.warning("cannot parse merchant response as JSON format")
            final_terms = initial_offer.copy()
    except Exception as e:
        logger.warning(f"error extracting agreement terms: {str(e)}")
        final_terms = initial_offer.copy()
    
    return final_terms

def platform_traffic_allocation(self) -> None:
    """
    平台流量分配阶段
    
    Args:
        self: 仿真控制器
    """
    logger = logging.getLogger("simulation")
    logger.info("execute platform traffic allocation stage")
    
    # 获取当前状态
    state = self.environment.get_state()
    
    # 平台根据主播信誉、历史GMV等分配流量
    platform_agent = self.agents.get("platform")
    if platform_agent:
        logger.info("platform makes a decision")
        decision = platform_agent.make_decision(state)
        platform_agent.apply_decision(decision)

def live_streaming_sales(self) -> None:
    """
    直播带货阶段
    
    Args:
        self: 仿真控制器
    """
    logger = logging.getLogger("simulation")
    logger.info("execute live streaming sales stage")
    
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
            
        logger.info(f"直播带货阶段使用{max_workers}个线程，每批{batch_size}个任务")
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
            
        # 获取主播Agent
        streamer_agent = self.agents.get(f"streamer_{streamer_id}")
        if not streamer_agent:
            logger.warning(f"找不到主播 {streamer_id} 的Agent，跳过直播")
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
                # 获取消费者Agent
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
        current_batch = all_streaming_tasks[i:i+batch_size]
        batch_number = i // batch_size + 1
        total_batches = (total_count + batch_size - 1) // batch_size
        
        logger.info(f"处理批次 {batch_number}/{total_batches}, 任务数: {len(current_batch)}")
        
        # 计算每个线程处理的任务数
        tasks_per_thread = max(1, len(current_batch) // max_workers)
        thread_batches = []
        
        for j in range(0, len(current_batch), tasks_per_thread):
            thread_batches.append(current_batch[j:j+tasks_per_thread])
        
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
        
        logger.info(f"完成 {processed_count}/{total_count} 个任务 ({processed_count*100/total_count:.1f}%), "
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
    
    logger.info(f"Live streaming sales stage completed, total GMV: {total_gmv}, total commission: {total_commission}, total sales: {total_sales_count}")

def transaction_feedback(self) -> None:
    """
    交易与反馈阶段
    
    Args:
        self: 仿真控制器
    """
    logger = logging.getLogger("simulation")
    logger.info("Execute transaction and feedback stage")
    
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
    打印所有Agent的记忆摘要
    
    Args:
        self: 仿真控制器
        round_num: 回合数
    """
    # 仅在日志中记录一次标题
    logger = logging.getLogger("simulation")
    logger.info(f"======== All Agent memories summary for round {round_num+1} ========")
    
    # 使用print而不是logger.info打印到标准输出
    print(f"\n======== All Agent memories summary for round {round_num+1} ========")
    
    # 先按类型分组Agent
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
        print("======== Platform memory ========")
        has_memories = False
        for agent in agents_by_type["platform"]:
            summary = agent.get_memory_summary(round_num)
            if summary:
                print(f"Platform {agent.agent_id} memory summary:\n{summary}")
                has_memories = True
        if not has_memories:
            print("Platform memory is empty")
    
    # 打印主播记忆
    if agents_by_type["streamer"]:
        print("======== Streamer memory ========")
        has_memories = False
        for agent in agents_by_type["streamer"]:
            summary = agent.get_memory_summary(round_num)
            if summary:
                print(f"Streamer {agent.agent_id} memory summary:\n{summary}")
                has_memories = True
        if not has_memories:
            print("Streamer memory is empty - please check if the streamer agent has correctly added memories")
    
    # 打印商家记忆
    if agents_by_type["merchant"]:
        print("======== Merchant memory ========")
        has_memories = False
        for agent in agents_by_type["merchant"]:
            summary = agent.get_memory_summary(round_num)
            if summary:
                print(f"Merchant {agent.agent_id} memory summary:\n{summary}")
                has_memories = True
        if not has_memories:
            print("Merchant memory is empty")
    
    # 打印消费者记忆
    if agents_by_type["consumer"]:
        print("======== Consumer memory ========")
        has_memories = False
        for agent in agents_by_type["consumer"]:
            summary = agent.get_memory_summary(round_num)
            if summary:
                print(f"Consumer {agent.agent_id} memory summary:\n{summary}")
                has_memories = True
        if not has_memories:
            print("Consumer memory is empty - please check if the consumer agent has correctly added memories")

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
        print("Simulation system setup successful")
        
        # 更新仿真控制器方法
        update_simulator_methods(simulator)
        print("Simulation controller method update successful")
        
        # 运行仿真
        print("Starting simulation...")
        result = simulator.run_simulation()
        print("Simulation completed")
        
        # 打印结果摘要
        print("\n=================")
        print("仿真结束! 结果摘要:")
        print(f"Total rounds: {result['total_rounds']}")
        print(f"Total GMV: {result['final_metrics']['total_gmv']}")
        print(f"Total commission: {result['final_metrics']['total_commission']}")
        print(f"Total sales: {result['final_metrics']['total_sales_count']}")
        print("=================\n")
        
        # 打印所有Agent的最终记忆摘要
        simulator.print_agent_memories(result['total_rounds'] - 1)
    except Exception as e:
        print(f"运行时出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return 0

if __name__ == "__main__":
    print("Program started...")
    sys.exit(main())