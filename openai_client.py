"""
OpenAI API客户端，用于与特定的OpenAI兼容API通信
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

# 特定的API配置
API_URL = "https://api.deepbricks.ai/v1/chat/completions"
API_KEY = "sk-Kp59pIj8PfqzLzYaAABh2jKsQLB0cUKU3n8l7TIK3rpU61QG"
MODEL_NAME = "gpt-4o-mini"

class OpenAIClient:
    """
    OpenAI API客户端，用于与OpenAI兼容API通信
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
                cls._instance = super(OpenAIClient, cls).__new__(cls)
            return cls._instance
    
    def __init__(self, api_url: str = API_URL, api_key: str = API_KEY, model_name: str = MODEL_NAME):
        """
        初始化OpenAI客户端
        
        Args:
            api_url: OpenAI API的基础URL
            api_key: OpenAI API密钥
            model_name: 使用的模型名称
        """
        # 防止重复初始化
        if OpenAIClient._initialized:
            return
            
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.logger = logging.getLogger("openai_client")
        
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
        os.makedirs("cache/openai_responses", exist_ok=True)
        
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
        self.session.mount(
            'https://', 
            requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=20,
                max_retries=3
            )
        )
        
        # 添加预热请求
        self._warm_up()
        
        # 启动后台批处理线程
        self._start_batch_processor()
        
        OpenAIClient._initialized = True
    
    def _warm_up(self):
        """
        预热API连接
        """
        try:
            self._test_connection()
        except Exception as e:
            self.logger.warning(f"API预热失败: {str(e)}")
    
    def _start_batch_processor(self):
        """
        启动后台批处理线程
        """
        thread = threading.Thread(target=self._batch_processor_worker, daemon=True)
        thread.start()
    
    def _batch_processor_worker(self):
        """
        批处理工作线程
        """
        while True:
            batch = []
            result_queues = []
            use_cache_flags = []
            
            # 收集批处理请求
            try:
                # 获取第一个请求
                prompt, result_queue, use_cache = self.request_queue.get(timeout=1.0)
                batch.append(prompt)
                result_queues.append(result_queue)
                use_cache_flags.append(use_cache)
                
                # 尝试获取更多请求直到达到批处理大小或等待超时
                batch_timeout = time.time() + self.batch_wait_time
                while len(batch) < self.batch_size and time.time() < batch_timeout:
                    try:
                        prompt, result_queue, use_cache = self.request_queue.get(block=False)
                        batch.append(prompt)
                        result_queues.append(result_queue)
                        use_cache_flags.append(use_cache)
                    except queue.Empty:
                        break
                
                # 处理批处理请求
                if batch:
                    with ThreadPoolExecutor(max_workers=min(len(batch), 10)) as executor:
                        futures = []
                        for i, prompt in enumerate(batch):
                            futures.append(executor.submit(
                                self._process_single_request, 
                                prompt, 
                                result_queues[i],
                                use_cache_flags[i]
                            ))
                        
                        # 等待所有请求完成
                        for future in futures:
                            future.result()
                
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                self.logger.error(f"批处理工作线程错误: {str(e)}")
                # 对于未处理的请求，返回错误
                for result_queue in result_queues:
                    try:
                        result_queue.put(f"错误: {str(e)}")
                    except:
                        pass
    
    def _process_single_request(self, prompt, result_queue, use_cache):
        """
        处理单个请求
        
        Args:
            prompt: 提示文本
            result_queue: 结果队列
            use_cache: 是否使用缓存
        """
        try:
            # 检查缓存
            if use_cache:
                cache_key = self._get_cache_key(prompt)
                with self.cache_lock:
                    cached_result = self.cache.get(cache_key)
                    if cached_result:
                        self.cache_hits += 1
                        self.cache_access_times[cache_key] = time.time()
                        result_queue.put(cached_result)
                        return
            
            # 缓存未命中，调用API
            self.cache_misses += 1
            response = self._call_api(prompt)
            
            # 更新缓存
            if use_cache:
                with self.cache_lock:
                    self.cache[cache_key] = response
                    self.cache_access_times[cache_key] = time.time()
                    
                    # 如果缓存太大，清理最近最少使用的项目
                    if len(self.cache) > self.max_cache_size:
                        self._evict_lru_cache_items()
            
            # 返回结果
            result_queue.put(response)
            
        except Exception as e:
            self.logger.error(f"处理请求错误: {str(e)}")
            result_queue.put(f"错误: {str(e)}")
    
    def _evict_lru_cache_items(self):
        """
        清理最近最少使用的缓存项目
        """
        # 计算要删除的项目数量
        items_to_remove = int(self.max_cache_size * 0.2)  # 删除20%的缓存
        
        # 按访问时间排序
        sorted_items = sorted(
            self.cache_access_times.items(), 
            key=lambda x: x[1]
        )
        
        # 删除最旧的项目
        for i in range(min(items_to_remove, len(sorted_items))):
            key = sorted_items[i][0]
            del self.cache[key]
            del self.cache_access_times[key]
    
    def _call_api(self, prompt: str) -> str:
        """
        调用OpenAI API
        
        Args:
            prompt: 提示文本
            
        Returns:
            str: API响应
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 构建请求数据
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 500
        }
        
        # 发送请求
        try:
            self.logger.info(f"发送请求到 {self.api_url}")
            self.logger.debug(f"请求数据: {json.dumps(data, ensure_ascii=False)}")
            
            response = self.session.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            # 解析响应
            self.logger.info(f"收到API响应，状态码: {response.status_code}")
            self.logger.debug(f"响应内容: {response.text}")
            
            result = response.json()
            self.logger.debug(f"解析后的响应: {json.dumps(result, ensure_ascii=False)}")
            
            # 提取生成的文本
            if "choices" in result and len(result["choices"]) > 0:
                if "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                    content = result["choices"][0]["message"]["content"]
                    self.logger.info(f"成功提取响应内容，长度: {len(content)}")
                    return content
                else:
                    self.logger.warning(f"无法从choices中提取message.content: {json.dumps(result['choices'][0], ensure_ascii=False)}")
            else:
                self.logger.warning(f"响应中没有choices字段: {json.dumps(result, ensure_ascii=False)}")
                # 尝试从其他字段提取内容
                if "response" in result:
                    content = result["response"]
                    self.logger.info(f"从response字段提取内容成功，长度: {len(content)}")
                    return content
            
            # 如果无法提取文本，返回原始响应
            self.logger.warning("无法提取文本内容，返回原始响应JSON")
            return json.dumps(result)
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API请求错误: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"解析响应JSON时出错: {str(e)}")
            self.logger.error(f"原始响应: {response.text}")
            raise
    
    def get_completion(self, prompt: str, use_cache: bool = True) -> str:
        """
        获取文本补全
        
        Args:
            prompt: 提示文本
            use_cache: 是否使用缓存
            
        Returns:
            str: 生成的文本
        """
        # 创建结果队列
        result_queue = queue.Queue()
        
        # 将请求添加到队列
        self.request_queue.put((prompt, result_queue, use_cache))
        
        # 等待结果
        result = result_queue.get()
        
        # 检查是否有错误
        if isinstance(result, str) and result.startswith("错误:"):
            raise Exception(result)
        
        return result
    
    def batch_inference(self, prompts: List[str], use_cache: bool = True) -> List[str]:
        """
        批量推理
        
        Args:
            prompts: 提示文本列表
            use_cache: 是否使用缓存
            
        Returns:
            List[str]: 生成的文本列表
        """
        # 创建结果队列列表
        result_queues = [queue.Queue() for _ in range(len(prompts))]
        
        # 将请求添加到队列
        for i, prompt in enumerate(prompts):
            self.request_queue.put((prompt, result_queues[i], use_cache))
        
        # 等待所有结果
        results = [queue.get() for queue in result_queues]
        
        # 检查是否有错误
        for i, result in enumerate(results):
            if isinstance(result, str) and result.startswith("错误:"):
                self.logger.error(f"批量推理错误 (prompt {i}): {result}")
                results[i] = f"错误: {result}"
        
        return results
    
    def clear_cache(self):
        """
        清除缓存
        """
        with self.cache_lock:
            self.cache.clear()
            self.cache_access_times.clear()
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成文本
        
        Args:
            prompt: 用户提示文本
            system_prompt: 系统提示文本
            params: 生成参数
            
        Returns:
            Dict[str, Any]: 生成结果
        """
        # 检查缓存
        if self.use_cache:
            cache_key = self._get_cache_key(prompt, system_prompt, params)
            cached_result = self._load_from_cache(cache_key)
            if cached_result:
                self.cache_hits += 1
                return cached_result
        
        # 构建完整提示
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # 构建请求数据
        data = {
            "model": self.model_name,
            "messages": []
        }
        
        # 添加系统提示
        if system_prompt:
            data["messages"].append({"role": "system", "content": system_prompt})
        
        # 添加用户提示
        data["messages"].append({"role": "user", "content": prompt})
        
        # 添加参数
        if params:
            data.update(params)
        
        # 发送请求
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            self.logger.info(f"Generate: 发送请求到 {self.api_url}")
            self.logger.debug(f"Generate: 请求数据: {json.dumps(data, ensure_ascii=False)}")
            
            response = self.session.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            # 解析响应
            self.logger.info(f"Generate: 收到API响应，状态码: {response.status_code}")
            self.logger.debug(f"Generate: 响应内容: {response.text}")
            
            result = response.json()
            self.logger.debug(f"Generate: 解析后的响应: {json.dumps(result, ensure_ascii=False)}")
            
            # 构建标准格式的响应
            standardized_result = {
                "id": result.get("id", ""),
                "object": result.get("object", ""),
                "created": result.get("created", 0),
                "model": result.get("model", self.model_name),
                "choices": []
            }
            
            # 提取生成的文本
            if "choices" in result and len(result["choices"]) > 0:
                for choice in result["choices"]:
                    if "message" in choice and "content" in choice["message"]:
                        standardized_choice = {
                            "index": choice.get("index", 0),
                            "message": {
                                "role": choice["message"].get("role", "assistant"),
                                "content": choice["message"]["content"]
                            },
                            "finish_reason": choice.get("finish_reason", "stop")
                        }
                        standardized_result["choices"].append(standardized_choice)
                    else:
                        self.logger.warning(f"Generate: 无法从choice中提取message.content: {json.dumps(choice, ensure_ascii=False)}")
            else:
                self.logger.warning(f"Generate: 响应中没有choices字段: {json.dumps(result, ensure_ascii=False)}")
                # 尝试从其他字段提取内容
                if "response" in result:
                    standardized_choice = {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result["response"]
                        },
                        "finish_reason": "stop"
                    }
                    standardized_result["choices"].append(standardized_choice)
                    self.logger.info(f"Generate: 从response字段提取内容成功，长度: {len(result['response'])}")
            
            # 缓存结果
            if self.use_cache:
                self._save_to_cache(cache_key, standardized_result)
            
            return standardized_result
            
        except Exception as e:
            self.logger.error(f"Generate: 生成文本错误: {str(e)}")
            raise
    
    def _test_connection(self) -> bool:
        """
        测试API连接
        
        Returns:
            bool: 连接是否成功
        """
        if self._connection_tested:
            return self._connection_ok
        
        try:
            # 简单的测试请求
            response = self.generate("Hello, world!", params={"max_tokens": 5})
            self._connection_ok = True
        except Exception as e:
            self.logger.error(f"API连接测试失败: {str(e)}")
            self._connection_ok = False
        
        self._connection_tested = True
        return self._connection_ok
    
    def _get_cache_key(self, prompt: str, system_prompt: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> str:
        """
        获取缓存键
        
        Args:
            prompt: 提示文本
            system_prompt: 系统提示文本
            params: 生成参数
            
        Returns:
            str: 缓存键
        """
        # 构建缓存键
        key_parts = [prompt]
        if system_prompt:
            key_parts.append(system_prompt)
        if params:
            key_parts.append(json.dumps(params, sort_keys=True))
        
        # 计算哈希值
        key = hashlib.md5("".join(key_parts).encode()).hexdigest()
        return key
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """
        保存结果到缓存
        
        Args:
            cache_key: 缓存键
            result: 结果
        """
        # 内存缓存
        with self.cache_lock:
            self.cache[cache_key] = result
            self.cache_access_times[cache_key] = time.time()
            
            # 如果缓存太大，清理最近最少使用的项目
            if len(self.cache) > self.max_cache_size:
                self._evict_lru_cache_items()
        
        # 文件缓存
        try:
            cache_file = os.path.join("cache", "openai_responses", f"{cache_key}.json")
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"保存缓存文件失败: {str(e)}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        从缓存加载结果
        
        Args:
            cache_key: 缓存键
            
        Returns:
            Optional[Dict[str, Any]]: 缓存的结果
        """
        # 检查内存缓存
        with self.cache_lock:
            if cache_key in self.cache:
                self.cache_access_times[cache_key] = time.time()
                return self.cache[cache_key]
        
        # 检查文件缓存
        try:
            cache_file = os.path.join("cache", "openai_responses", f"{cache_key}.json")
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    result = json.load(f)
                
                # 更新内存缓存
                with self.cache_lock:
                    self.cache[cache_key] = result
                    self.cache_access_times[cache_key] = time.time()
                
                return result
        except Exception as e:
            self.logger.warning(f"加载缓存文件失败: {str(e)}")
        
        return None
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        解析JSON响应，尝试多种方法修复不规范的JSON
        """
        if not response or not isinstance(response, str):
            return {"response": response}
        
        # 保存原始响应以便需要时使用LLM辅助解析
        original_response = response
        
        try:
            # 1. 首先检查是否是Markdown格式的代码块
            if '```json' in response or '```' in response:
                self.logger.debug("检测到Markdown格式的代码块，尝试提取JSON内容")
                # 提取代码块内容
                pattern = r'```(?:json)?\s*([\s\S]*?)```'
                matches = re.findall(pattern, response)
                if matches:
                    # 使用第一个匹配的代码块
                    response = matches[0].strip()
                    self.logger.debug(f"从Markdown中提取到JSON内容: {response[:100]}...")
            
            # 2. 直接尝试解析
            json_obj = json.loads(response)
            self.logger.debug("JSON直接解析成功")
            return json_obj
        except json.JSONDecodeError as e:
            self.logger.debug(f"直接解析JSON失败: {str(e)}")
            
        # 3. 尝试修复和清理JSON
        try:
            # 清理控制字符
            cleaned_json = self._clean_control_characters(response)
            # 修复引号问题
            cleaned_json = self._fix_unbalanced_quotes(cleaned_json)
            cleaned_json = self._fix_json_quotes(cleaned_json)
            # 移除注释
            cleaned_json = self._remove_json_comments(cleaned_json)
            # 尝试JSON解析
            json_obj = json.loads(cleaned_json)
            self.logger.debug("修复后的JSON解析成功")
            return json_obj
        except json.JSONDecodeError as e:
            self.logger.debug(f"解析修复后的JSON失败: {str(e)}")
            
        # 4. 使用demjson进行更宽松的解析
        try:
            import demjson3 as demjson
            fixed_json = self._fix_json_for_demjson(response)
            json_obj = demjson.decode(fixed_json)
            self.logger.debug("使用demjson解析成功")
            return json_obj
        except Exception as e:
            self.logger.debug(f"demjson解析失败: {str(e)}")
        
        # 5. 尝试提取JSON内容
        try:
            extracted_json = self._extract_json_content(response)
            if extracted_json:
                json_obj = json.loads(extracted_json)
                self.logger.debug("从文本中提取并解析JSON成功")
                return json_obj
        except json.JSONDecodeError as e:
            self.logger.debug(f"解析提取的JSON内容失败: {str(e)}")
            
        # 6. 最后尝试使用正则表达式解析键值对
        try:
            # 寻找可能的JSON键值对
            pattern = r'"([^"]+)"\s*:\s*("(?:\\.|[^"\\])*"|null|true|false|[\d.-]+|\{[^}]*\}|\[[^\]]*\])'
            matches = re.findall(pattern, response)
            if matches:
                result = {}
                for key, value in matches:
                    try:
                        # 尝试将值解析为JSON
                        parsed_value = json.loads(value)
                    except:
                        # 如果解析失败，保留原始字符串
                        parsed_value = value
                    result[key] = parsed_value
                self.logger.debug("使用正则表达式提取键值对成功")
                return result
        except Exception as e:
            self.logger.debug(f"正则表达式解析失败: {str(e)}")
            
        # 7. 所有常规方法都失败，尝试使用LLM辅助解析
        try:
            self.logger.warning("所有标准解析方法失败，尝试使用LLM辅助解析")
            return self._llm_assisted_parsing(original_response)
        except Exception as e:
            self.logger.error(f"LLM辅助解析失败: {str(e)}")
        
        # 所有解析尝试都失败，将整个响应作为'response'字段返回
        self.logger.debug("所有解析尝试都失败，将整个响应作为'response'字段返回")
        return {"response": response}
    
    def _clean_control_characters(self, text: str) -> str:
        """
        清理控制字符
        
        Args:
            text: 输入文本
            
        Returns:
            str: 清理后的文本
        """
        # 移除不可打印字符
        text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\t', '\r'])
        
        # 移除ANSI转义序列
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        text = ansi_escape.sub('', text)
        
        # 移除零宽字符
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
        
        return text
    
    def _fix_unbalanced_quotes(self, json_str: str) -> str:
        """
        修复不平衡的引号
        
        Args:
            json_str: JSON字符串
            
        Returns:
            str: 修复后的JSON字符串
        """
        # 计算引号数量
        double_quotes = json_str.count('"')
        
        # 如果引号数量为奇数，尝试修复
        if double_quotes % 2 == 1:
            # 查找最后一个引号
            last_quote_pos = json_str.rfind('"')
            if last_quote_pos != -1:
                # 检查是否是值的结束引号
                if last_quote_pos > 0 and json_str[last_quote_pos-1:last_quote_pos] != '\\':
                    # 添加缺失的引号
                    json_str += '"'
        
        # 修复常见的键值对问题
        json_str = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', json_str)
        
        return json_str
    
    def _fix_json_quotes(self, json_str: str) -> str:
        """
        修复JSON引号
        
        Args:
            json_str: JSON字符串
            
        Returns:
            str: 修复后的JSON字符串
        """
        # 替换单引号为双引号（但不替换转义的单引号）
        result = ""
        i = 0
        while i < len(json_str):
            if json_str[i] == "'" and (i == 0 or json_str[i-1] != '\\'):
                result += '"'
            else:
                result += json_str[i]
            i += 1
        
        # 修复常见的尾部逗号问题
        result = re.sub(r',\s*([}\]])', r'\1', result)
        
        return result
    
    def _fix_json_for_demjson(self, json_str: str) -> str:
        """
        为demjson修复JSON
        
        Args:
            json_str: JSON字符串
            
        Returns:
            str: 修复后的JSON字符串
        """
        # 移除注释
        json_str = self._remove_json_comments(json_str)
        
        # 确保键使用双引号
        json_str = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', json_str)
        
        # 修复布尔值和null
        json_str = re.sub(r'\btrue\b', 'true', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'\bfalse\b', 'false', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'\bnull\b', 'null', json_str, flags=re.IGNORECASE)
        
        return json_str
    
    def _extract_json_content(self, text: str) -> str:
        """
        提取JSON内容
        
        Args:
            text: 输入文本
            
        Returns:
            str: 提取的JSON内容
        """
        # 查找第一个{或[
        start_brace = text.find('{')
        start_bracket = text.find('[')
        
        if start_brace == -1 and start_bracket == -1:
            return ""
        
        # 确定JSON开始位置
        if start_brace == -1:
            start = start_bracket
        elif start_bracket == -1:
            start = start_brace
        else:
            start = min(start_brace, start_bracket)
        
        # 查找匹配的结束括号
        stack = []
        for i in range(start, len(text)):
            if text[i] in '{[':
                stack.append(text[i])
            elif text[i] == '}' and stack and stack[-1] == '{':
                stack.pop()
                if not stack:
                    return text[start:i+1]
            elif text[i] == ']' and stack and stack[-1] == '[':
                stack.pop()
                if not stack:
                    return text[start:i+1]
        
        # 如果没有找到匹配的结束括号，返回从开始到结束的所有内容
        return text[start:]
    
    def _remove_json_comments(self, json_str: str) -> str:
        """
        移除JSON注释
        
        Args:
            json_str: JSON字符串
            
        Returns:
            str: 移除注释后的JSON字符串
        """
        # 移除单行注释
        json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
        
        # 移除多行注释
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        return json_str
    
    def _process_parsed_json(self, parsed: Any) -> Dict[str, Any]:
        """
        处理解析后的JSON
        
        Args:
            parsed: 解析后的JSON对象
            
        Returns:
            Dict[str, Any]: 处理后的JSON对象
        """
        if isinstance(parsed, dict):
            return parsed
        elif isinstance(parsed, list):
            return {"items": parsed}
        else:
            return {"value": parsed}
    
    def _llm_assisted_parsing(self, response_text: str) -> Dict[str, Any]:
        """
        使用LLM辅助解析非标准格式的JSON或提取关键决策信息
        
        当所有常规解析方法都失败时，使用LLM来帮助理解和提取响应中的关键信息
        """
        system_prompt = """
你是一个专业的JSON解析助手。你的任务是:
1. 分析提供的文本
2. 提取其中的JSON数据或关键决策信息
3. 返回一个格式正确的JSON对象

如果文本包含不规范的JSON:
- 修复缺失的引号、逗号等语法错误
- 确保所有字段名都有双引号
- 确保数值、布尔值和null值格式正确

如果文本是描述性的而非JSON格式:
- 识别关键决策点和对应的值
- 将这些信息组织成JSON格式
- 确保返回的是一个有效的JSON对象

你的输出必须是一个格式完全正确的JSON，且只包含JSON，不要有任何其他说明。
"""

        prompt = f"""
需要解析以下文本中的JSON或决策信息:

```
{response_text}
```

请提取其中的关键信息，并返回一个格式正确的JSON对象。确保返回的是有效的JSON格式。
"""

        try:
            # 使用自己作为LLM来解析
            parsing_result = self.generate(prompt, system_prompt)
            
            # 从结果中提取JSON
            if isinstance(parsing_result, dict):
                if "choices" in parsing_result and len(parsing_result["choices"]) > 0:
                    content = parsing_result["choices"][0].get("message", {}).get("content", "")
                elif "content" in parsing_result:
                    content = parsing_result["content"]
                elif "response" in parsing_result:
                    content = parsing_result["response"]
                else:
                    content = str(parsing_result)
            else:
                content = str(parsing_result)
                
            # 再次尝试从内容中解析JSON
            try:
                # 检查是否是Markdown格式的代码块
                if '```json' in content or '```' in content:
                    pattern = r'```(?:json)?\s*([\s\S]*?)```'
                    matches = re.findall(pattern, content)
                    if matches:
                        content = matches[0].strip()
                
                # 尝试解析JSON
                result = json.loads(content)
                self.logger.info("LLM辅助解析成功")
                return result
            except json.JSONDecodeError:
                # 如果解析失败，尝试创建一个默认结构
                self.logger.warning("LLM返回的内容不是有效JSON，创建默认结构")
                return {"llm_parsed_response": content}
                
        except Exception as e:
            self.logger.error(f"LLM辅助解析过程中出错: {str(e)}")
            # 提供一个默认的安全返回值
            return {"error": "解析失败", "original_text": response_text[:200] + "..."} 