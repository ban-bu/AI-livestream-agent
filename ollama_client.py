"""
Ollama API客户端，用于与本地Ollama模型通信
"""

import json
import requests
import re
import hashlib
from typing import Dict, Any, Optional, List, Union
import logging
import os
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor

from config import OLLAMA_BASE_URL, MODEL_NAME, MODEL_PARAMS, USE_CACHING

class OllamaClient:
    """
    Ollama API客户端，用于与本地Ollama模型通信
    """
    
    # 单例模式实现 - 类变量用于存储实例
    _instance = None
    _instance_lock = threading.Lock()
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """
        单例模式实现 - 确保只有一个客户端实例
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super(OllamaClient, cls).__new__(cls)
            return cls._instance
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model_name: str = MODEL_NAME):
        """
        初始化Ollama客户端
        
        Args:
            base_url: Ollama API的基础URL
            model_name: 使用的模型名称
        """
        # 防止重复初始化
        if OllamaClient._initialized:
            return
            
        self.base_url = base_url
        self.model_name = model_name
        self.generate_url = f"{base_url}/api/generate"
        self.logger = logging.getLogger("ollama_client")
        
        # 启用响应缓存
        self.use_cache = True  # 总是启用缓存以提高性能
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 连接状态
        self._connection_tested = False
        self._connection_ok = False
        
        # 确保缓存目录存在
        os.makedirs("cache", exist_ok=True)
        os.makedirs("cache/responses", exist_ok=True)
        
        # 添加请求队列和响应缓存
        self.request_queue = queue.Queue()
        self.response_cache = {}
        self.cache_lock = threading.Lock()
        
        # 增加缓存大小以减少LLM调用
        self.max_cache_size = 5000  # 增加缓存容量
        
        # LRU缓存实现
        self.cache_access_times = {}
        
        # 添加批处理设置 - 优化批处理大小和等待时间
        self.batch_size = 10  # 增加批处理大小
        self.batch_wait_time = 0.05  # 减少等待时间
        
        # 添加HTTP会话复用
        self.session = requests.Session()
        self.session.mount(
            'http://', 
            requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=20,
                max_retries=3
            )
        )
        
        # 测试连接
        if not self._test_connection():
            self.logger.error("无法连接到Ollama服务，请确保服务正在运行且配置正确")
            return
            
        # 添加预热请求
        self._warm_up()
        
        # 启动后台批处理线程
        self._start_batch_processor()
        
        # 标记为已初始化
        OllamaClient._initialized = True
        
        self.logger.info("OllamaClient初始化完成")
    
    def _warm_up(self):
        """发送一个预热请求以初始化模型"""
        try:
            self._call_api("你好，这是一个预热请求")
            self.logger.info("模型预热完成")
        except Exception as e:
            self.logger.warning(f"模型预热失败: {str(e)}")
    
    def _start_batch_processor(self):
        """启动后台批处理线程"""
        self.batch_processor_thread = threading.Thread(
            target=self._batch_processor_worker,
            daemon=True
        )
        self.batch_processor_thread.start()
    
    def _batch_processor_worker(self):
        """批处理工作线程，合并多个请求"""
        while True:
            batch = []
            # 收集请求直到达到批处理大小或等待超时
            try:
                # 获取第一个请求，阻塞等待
                first_item = self.request_queue.get(block=True)
                batch.append(first_item)
                
                # 尝试获取更多请求，直到达到批处理大小或超时
                timeout = time.time() + self.batch_wait_time
                while len(batch) < self.batch_size and time.time() < timeout:
                    try:
                        item = self.request_queue.get(block=False)
                        batch.append(item)
                    except queue.Empty:
                        # 短暂休眠，避免CPU高占用
                        time.sleep(0.005)  # 减少休眠时间
                        if self.request_queue.empty():
                            break
            
            except Exception as e:
                self.logger.error(f"批处理工作线程出错: {str(e)}")
                continue
            
            # 处理批次请求
            if batch:
                with ThreadPoolExecutor(max_workers=min(len(batch), 20)) as executor:  # 限制最大并行数
                    # 并行处理批次中的所有请求
                    for prompt_data in batch:
                        prompt, result_queue, use_cache = prompt_data
                        
                        # 如果使用缓存且缓存命中，直接返回缓存结果
                        if use_cache:
                            with self.cache_lock:
                                if prompt in self.response_cache:
                                    # 更新LRU访问时间
                                    self.cache_access_times[prompt] = time.time()
                                    result_queue.put(self.response_cache[prompt])
                                    self.cache_hits += 1
                                    continue
                        
                        # 提交请求
                        executor.submit(self._process_single_request, prompt, result_queue, use_cache)
    
    def _process_single_request(self, prompt, result_queue, use_cache):
        """处理单个请求"""
        try:
            # 计算缓存键
            cache_key = hashlib.md5(prompt.encode('utf-8')).hexdigest()
            cache_file = os.path.join("cache/responses", f"{cache_key}.json")
            
            # 优先从文件缓存加载
            if use_cache and os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        response = f.read()
                        # 保存到内存缓存
                        with self.cache_lock:
                            self.response_cache[prompt] = response
                            self.cache_access_times[prompt] = time.time()
                        result_queue.put(response)
                        self.cache_hits += 1
                        return
                except Exception:
                    # 文件缓存读取失败，继续使用API
                    pass
            
            # 文件缓存不存在，发送API请求
            self.cache_misses += 1
            response = self._call_api(prompt)
            
            # 如果启用缓存，保存结果
            if use_cache:
                with self.cache_lock:
                    # 如果缓存已满，移除最久未使用的项目
                    if len(self.response_cache) >= self.max_cache_size:
                        self._evict_lru_cache_items()
                    
                    # 保存到内存缓存
                    self.response_cache[prompt] = response
                    self.cache_access_times[prompt] = time.time()
                
                # 保存到文件缓存
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        f.write(response)
                except Exception as e:
                    self.logger.warning(f"保存缓存文件失败: {str(e)}")
            
            # 返回结果
            result_queue.put(response)
            
        except Exception as e:
            # 出错时返回错误信息
            self.logger.error(f"处理请求出错: {str(e)}")
            result_queue.put(f"错误: {str(e)}")
    
    def _evict_lru_cache_items(self):
        """从缓存中移除最久未使用的项目"""
        # 移除10%的最久未使用项目
        items_to_remove = max(1, int(self.max_cache_size * 0.1))
        if not self.cache_access_times:
            return
            
        # 按访问时间排序
        sorted_items = sorted(self.cache_access_times.items(), key=lambda x: x[1])
        
        # 移除最久未使用的项目
        for i in range(min(items_to_remove, len(sorted_items))):
            key = sorted_items[i][0]
            if key in self.response_cache:
                del self.response_cache[key]
            if key in self.cache_access_times:
                del self.cache_access_times[key]
    
    def _call_api(self, prompt: str) -> str:
        """
        调用Ollama API
        
        Args:
            prompt: 输入提示
            
        Returns:
            str: 模型响应
        """
        url = f"{self.base_url}/api/generate"
        
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        # 添加模型参数
        data.update(MODEL_PARAMS)
        
        try:
            start_time = time.time()
            # 使用会话进行请求以复用连接
            response = self.session.post(url, json=data, timeout=(5, 60))  # 添加超时设置
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                completion = result.get("response", "")
                tokens = result.get("eval_count", 0)
                duration = end_time - start_time
                
                self.logger.debug(f"请求完成，耗时: {duration:.2f}秒，令牌数: {tokens}")
                return completion
            else:
                error_msg = f"API请求失败，状态码: {response.status_code}"
                self.logger.error(error_msg)
                return f"错误: {error_msg}"
        except Exception as e:
            self.logger.error(f"API请求异常: {str(e)}")
            return f"错误: {str(e)}"
    
    def get_completion(self, prompt: str, use_cache: bool = None) -> str:
        """
        获取模型响应（单个请求）
        
        Args:
            prompt: 输入提示
            use_cache: 是否使用缓存，如果为None则使用实例的use_cache设置
            
        Returns:
            str: 模型响应
        """
        result_queue = queue.Queue()
        
        # 如果没有指定use_cache，使用实例的设置
        if use_cache is None:
            use_cache = self.use_cache
        
        # 添加到请求队列
        self.request_queue.put((prompt, result_queue, use_cache))
        
        # 等待结果
        return result_queue.get()
    
    def batch_inference(self, prompts: List[str], use_cache: bool = None) -> List[str]:
        """
        批量获取模型响应
        
        Args:
            prompts: 输入提示列表
            use_cache: 是否使用缓存，如果为None则使用实例的use_cache设置
            
        Returns:
            List[str]: 模型响应列表
        """
        if not prompts:
            return []
        
        result_queues = [queue.Queue() for _ in range(len(prompts))]
        
        # 如果没有指定use_cache，使用实例的设置
        if use_cache is None:
            use_cache = self.use_cache
        
        # 提交所有请求
        for i, prompt in enumerate(prompts):
            self.request_queue.put((prompt, result_queues[i], use_cache))
        
        # 收集所有结果
        results = [q.get() for q in result_queues]
        return results
    
    def clear_cache(self):
        """清除响应缓存"""
        with self.cache_lock:
            self.response_cache.clear()
            self.logger.info("已清空响应缓存")
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成文本响应（原始API兼容性）
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            params: 生成参数，将覆盖默认参数
            
        Returns:
            Dict: 生成结果
        """
        # 构建完整提示
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
        
        # 检查缓存 - 只在use_cache为True时使用缓存
        if self.use_cache:
            cache_key = self._get_cache_key(prompt, system_prompt, params)
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                self.logger.debug(f"缓存命中! 缓存命中率: {self.cache_hits/(self.cache_hits+self.cache_misses)*100:.2f}%")
                return cached_result
        
        # 获取响应
        response = self.get_completion(full_prompt)
        
        # 构建结果字典
        result_dict = {
            "response": response,
            "raw": {"response": response}  # 简化raw字段以兼容旧接口
        }
        
        # 保存到缓存 - 只在use_cache为True时保存缓存
        if self.use_cache:
            cache_key = self._get_cache_key(prompt, system_prompt, params)
            self._save_to_cache(cache_key, result_dict)
        
        return result_dict
    
    def _test_connection(self) -> bool:
        """
        测试与Ollama的连接
        
        Returns:
            bool: 连接是否成功
        """
        # 如果已经测试过连接，直接返回结果
        if self._connection_tested:
            return self._connection_ok
            
        try:
            self.logger.info("开始测试与Ollama的连接...")
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                
                if self.model_name not in available_models:
                    self.logger.warning(
                        f"模型 {self.model_name} 未在Ollama中找到。可用模型: {available_models}"
                    )
                    self._connection_ok = False
                else:
                    self.logger.info(f"成功连接到Ollama，使用模型: {self.model_name}")
                    self._connection_ok = True
            else:
                self.logger.error(f"连接Ollama失败: {response.status_code}")
                self._connection_ok = False
        except Exception as e:
            self.logger.error(f"连接Ollama时发生错误: {str(e)}")
            self._connection_ok = False
        
        self._connection_tested = True
        return self._connection_ok
    
    def _get_cache_key(self, prompt: str, system_prompt: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> str:
        """
        生成缓存键
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            params: 生成参数
            
        Returns:
            str: 缓存键
        """
        # 创建包含所有相关信息的字符串
        cache_str = f"{prompt}|{system_prompt or ''}|{json.dumps(params or {}, sort_keys=True)}"
        # 计算哈希值作为缓存键
        return hashlib.md5(cache_str.encode('utf-8')).hexdigest()
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """
        将结果保存到缓存
        
        Args:
            cache_key: 缓存键
            result: 生成结果
        """
        if not self.use_cache:
            return
        
        self.cache[cache_key] = result
        
        # 保存到文件
        try:
            cache_file = os.path.join("cache", f"{cache_key}.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f)
        except Exception as e:
            self.logger.warning(f"保存缓存失败: {str(e)}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        从缓存加载结果
        
        Args:
            cache_key: 缓存键
            
        Returns:
            Optional[Dict]: 缓存结果，没有则为None
        """
        if not self.use_cache:
            return None
        
        # 首先从内存缓存中查找
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        # 然后从文件中查找
        try:
            cache_file = os.path.join("cache", f"{cache_key}.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    self.cache[cache_key] = result  # 加载到内存缓存
                    self.cache_hits += 1
                    return result
        except Exception as e:
            self.logger.warning(f"读取缓存失败: {str(e)}")
        
        self.cache_misses += 1
        return None
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        解析JSON格式的响应文本，使用更高效的方法确保解析成功
        
        Args:
            response: LLM响应文本
            
        Returns:
            Dict: 解析后的JSON对象
        """
        # 提取JSON内容
        json_content = self._extract_json_content(response)
        if not json_content:
            self.logger.warning(f"未找到有效的JSON内容，返回空对象")
            return {}
        
        # 清理控制字符
        json_content = self._clean_control_characters(json_content)
            
        # 尝试标准JSON解析
        try:
            parsed = json.loads(json_content)
            return self._process_parsed_json(parsed)
        except json.JSONDecodeError as e:
            self.logger.warning(f"标准JSON解析失败: {str(e)}")
            
        # 尝试移除注释后解析
        try:
            cleaned_json = self._remove_json_comments(json_content)
            parsed = json.loads(cleaned_json)
            self.logger.info("通过移除注释成功解析JSON")
            return self._process_parsed_json(parsed)
        except json.JSONDecodeError as e:
            self.logger.warning(f"移除注释后解析失败: {str(e)}")
            
        # 尝试修复引号问题
        try:
            fixed_json = self._fix_json_quotes(json_content)
            parsed = json.loads(fixed_json)
            self.logger.info("通过修复引号成功解析JSON")
            return self._process_parsed_json(parsed)
        except json.JSONDecodeError as e:
            self.logger.warning(f"修复引号后解析失败: {str(e)}")
        
        # 尝试更多的修复方法
        try:
            # 尝试替换所有不配对的引号
            unbalanced_fixed = self._fix_unbalanced_quotes(json_content)
            parsed = json.loads(unbalanced_fixed)
            self.logger.info("通过修复不配对引号成功解析JSON")
            return self._process_parsed_json(parsed)
        except json.JSONDecodeError as e:
            self.logger.warning(f"修复不配对引号解析失败: {str(e)}")
            
        # 尝试宽松解析 - demjson3库提供了更宽松的JSON解析
        try:
            import demjson3
            parsed = demjson3.decode(json_content)
            self.logger.info("通过demjson3成功解析JSON")
            return self._process_parsed_json(parsed)
        except:
            try:
                # 如果demjson3解析失败，再尝试修复后用demjson3解析
                import demjson3
                fixed_json = self._fix_json_for_demjson(json_content)
                parsed = demjson3.decode(fixed_json)
                self.logger.info("通过修复后的demjson3成功解析JSON")
                return self._process_parsed_json(parsed)
            except:
                pass
        
        # 尝试使用ast.literal_eval作为最后的手段
        try:
            import ast
            parsed = ast.literal_eval(json_content)
            self.logger.info("通过ast.literal_eval成功解析")
            return self._process_parsed_json(parsed)
        except:
            pass
        
        # 所有解析方法都失败
        self.logger.error(f"所有JSON解析方法失败，返回空对象")
        return {}
    
    def _clean_control_characters(self, text: str) -> str:
        """
        清理JSON中的控制字符
        
        Args:
            text: 输入文本
            
        Returns:
            str: 清理后的文本
        """
        # 替换或移除ASCII控制字符 (0-31, 127)
        result = ""
        for char in text:
            if ord(char) < 32 or ord(char) == 127:
                # 替换制表符、换行符和回车符为空格
                if char in '\t\n\r':
                    result += ' '
                # 其他控制字符直接丢弃
            else:
                result += char
        return result
    
    def _fix_unbalanced_quotes(self, json_str: str) -> str:
        """
        修复不配对的引号问题
        
        Args:
            json_str: JSON字符串
            
        Returns:
            str: 修复后的字符串
        """
        # 检查双引号平衡性
        quote_count = json_str.count('"')
        if quote_count % 2 != 0:  # 引号数量为奇数
            # 查找所有键或值开始位置缺少引号的情况并修复
            import re
            # 修复键名前缺少引号: {key: "value"}
            json_str = re.sub(r'({|\,)\s*([a-zA-Z0-9_]+)\s*:'
                              r'\1 "\2":', json_str)
            
            # 修复字符串值缺少引号: {"key": value}
            json_str = re.sub(r':\s*([a-zA-Z0-9_]+)([,}])', r': "\1"\2', json_str)
            
            # 检查修复后的引号平衡性
            if json_str.count('"') % 2 != 0:
                # 如果仍不平衡，可能是结尾处缺少引号
                if json_str.endswith('}') and json_str[-2] != '"':
                    # 在最后一个}前可能缺少引号
                    json_str = json_str[:-1] + '"' + json_str[-1]
        
        return json_str
    
    def _fix_json_quotes(self, json_str: str) -> str:
        """
        修复JSON中的引号问题
        
        Args:
            json_str: JSON字符串
            
        Returns:
            str: 修复后的JSON字符串
        """
        import re
        
        # 修复没有引号的键
        # 例如 {key: "value"} 改为 {"key": "value"}
        pattern = r'(?<={|,)\s*([a-zA-Z0-9_]+)\s*:'
        fixed_str = re.sub(pattern, r'"\1":', json_str)
        
        # 修复单引号问题
        # 例如 {'key': 'value'} 改为 {"key": "value"}
        fixed_str = fixed_str.replace("'", '"')
        
        # 修复多引号问题
        # 例如 {""key"": ""value""} 改为 {"key": "value"}
        fixed_str = re.sub(r'"{2,}', '"', fixed_str)
        
        return fixed_str
        
    def _fix_json_for_demjson(self, json_str: str) -> str:
        """
        专门为demjson修复JSON
        
        Args:
            json_str: JSON字符串
            
        Returns:
            str: 修复后的JSON字符串
        """
        # 去除可能的逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # 修复布尔值和null
        json_str = json_str.replace('True', 'true')
        json_str = json_str.replace('False', 'false')
        json_str = json_str.replace('None', 'null')
        
        return json_str

    def _extract_json_content(self, text: str) -> str:
        """提取文本中的JSON内容 - 优化实现"""
        # 首先尝试解析Markdown代码块中的JSON
        json_code_block = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_code_block:
            return json_code_block.group(1).strip()
        
        # 如果没有代码块，尝试找出JSON对象
        json_obj_match = re.search(r'({[\s\S]*?})(?:\s*$|\n)', text)
        if json_obj_match:
            return json_obj_match.group(1)
        
        # 尝试找出JSON数组    
        json_array_match = re.search(r'(\[[\s\S]*?\])(?:\s*$|\n)', text)
        if json_array_match:
            return json_array_match.group(1)
                
        return ""
        
    def _remove_json_comments(self, json_str: str) -> str:
        """移除JSON字符串中的注释"""
        # 移除行内注释 // ...
        no_comments = re.sub(r'//.*?($|\n)', r'\1', json_str)
        # 移除块级注释 /* ... */
        no_comments = re.sub(r'/\*.*?\*/', '', no_comments, flags=re.DOTALL)
        return no_comments
    
    def _process_parsed_json(self, parsed: Any) -> Dict[str, Any]:
        """处理解析后的JSON对象，统一格式"""
        # 处理数组格式的响应
        if isinstance(parsed, list):
            result = {}
            for item in parsed:
                if isinstance(item, dict) and "product_id" in item:
                    product_id = item.pop("product_id")
                    # 统一字段名称
                    if "commission_rate" in item and "commission" not in item:
                        item["commission"] = item.pop("commission_rate")
                    # 统一库存字段
                    if "inventory" in item and "stock" not in item:
                        item["stock"] = item.pop("inventory")
                    elif "stock" in item and "inventory" not in item:
                        item["inventory"] = item["stock"]
                    result[product_id] = item
            return result
                    
        return parsed