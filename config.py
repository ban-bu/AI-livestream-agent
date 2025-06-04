"""
配置文件，包含系统参数设置
"""

# Ollama配置
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b-instruct-q4_K_M"
MODEL_PARAMS = {
    "temperature": 0.5,    # 降低temperature以提高确定性
    "top_p": 0.7,          # 降低top_p以减少随机性
    "max_tokens": 500,     # 减少生成的token数量
    "stream": False
}

# 仿真参数
SIMULATION_ROUNDS = 10  # 仿真回合数
SAVE_RESULTS = True     # 是否保存结果
RESULTS_DIR = "results" # 结果保存目录

# 性能优化设置
USE_LLM_RATIO = 1.0      # 使用LLM的比例，确保全部决策基于LLM
SAVE_DETAILED_LOG = False  # 是否保存详细日志
USE_CACHING = True        # 总是使用缓存以提高性能

# 并发和批处理优化
BATCH_SIZE = 10           # LLM请求批处理大小
MAX_WORKERS = 20          # 最大并行工作线程数
MEMORY_CACHE_SIZE = 5000  # 记忆缓存大小

# 测试规模控制参数
ACTIVE_STREAMERS_COUNT = 3  # 参与测试的主播数量
ACTIVE_CONSUMERS_COUNT = 6  # 参与测试的消费者数量
ACTIVE_STREAMERS = ["streamer_1", "streamer_2", "streamer_3"]  # 激活的主播ID列表
ACTIVE_CONSUMERS = ["consumer_high_1", "consumer_mid_1", "consumer_normal_1",
                   "consumer_high_2", "consumer_mid_2", "consumer_normal_2"]  # 激活的消费者ID列表

# 头部主播标识
TOP_STREAMERS = ["streamer_1", "streamer_2"]  # 头部主播ID列表

# 初始化参数
INITIAL_PARAMS = {
    # 平台初始参数
    "platform": {
        "commission_rate": 0.10,  # 默认佣金比例
    },
    
    # 主播初始参数
    "streamers": [
        {
            "id": "streamer_1",
            "name": "top streamer A",
            "streamer_type": "top",  # 头部主播
            "reputation": 90,     # 信誉评分 (0-100)
            "fans_count": 1000000, # 粉丝数量
            "avg_gmv": 5000000,   # 平均GMV
            "conversion_rate": 0.05, # 转化率
            "return_rate": 0.02,  # 退货率
        },
        {
            "id": "streamer_2",
            "name": "top streamer B",
            "streamer_type": "top",  # 头部主播
            "reputation": 88,
            "fans_count": 800000,
            "avg_gmv": 4000000,
            "conversion_rate": 0.05,
            "return_rate": 0.02,
        },
        {
            "id": "streamer_3",
            "name": "middle streamer A",
            "streamer_type": "regular",  # 普通主播
            "reputation": 75,
            "fans_count": 300000,
            "avg_gmv": 1000000,
            "conversion_rate": 0.04,
            "return_rate": 0.03,
        },
        {
            "id": "streamer_4",
            "name": "middle streamer B",
            "streamer_type": "regular",  # 普通主播
            "reputation": 72,
            "fans_count": 250000,
            "avg_gmv": 800000,
            "conversion_rate": 0.04,
            "return_rate": 0.03,
        },
        {
            "id": "streamer_5",
            "name": "new streamer A",
            "streamer_type": "regular",  # 普通主播
            "reputation": 60,
            "fans_count": 50000,
            "avg_gmv": 200000,
            "conversion_rate": 0.03,
            "return_rate": 0.05,
        },
        {
            "id": "streamer_6",
            "name": "new streamer B",
            "streamer_type": "regular",  # 普通主播
            "reputation": 58,
            "fans_count": 40000,
            "avg_gmv": 150000,
            "conversion_rate": 0.03,
            "return_rate": 0.05,
        }
    ],
    
    # 商家初始参数
    "merchants": [
        {
            "id": "merchant_1",
            "name": "famous brand",
            "reputation": 85,     # 信誉评分 (0-100)
            "products": [
                {
                    "id": "product_1",
                    "name": "high-end phone",
                    "base_price": 5000,
                    "stock": 1000,
                    "cost": 3500,
                    "quality": 90,
                },
                {
                    "id": "product_2",
                    "name": "bluetooth headset",
                    "base_price": 800,
                    "stock": 5000,
                    "cost": 300,
                    "quality": 85,
                }
            ]
        },
        {
            "id": "merchant_2",
            "name": "new brand",
            "reputation": 70,
            "products": [
                {
                    "id": "product_3",
                    "name": "smart watch",
                    "base_price": 1200,
                    "stock": 2000,
                    "cost": 600,
                    "quality": 75,
                },
                {
                    "id": "product_4",
                    "name": "smart speaker",
                    "base_price": 500,
                    "stock": 3000,
                    "cost": 250,
                    "quality": 80,
                }
            ]
        }
    ],
    
    # 消费者初始参数
    "consumers": [
        # 高端消费者组 (10个)
        {"id": "consumer_high_1", "name": "high-end consumer 1", "size": 500, "avg_budget": 8500, "price_sensitivity": 0.25, "trust_threshold": 0.75, "impulse_factor": 0.35},
        {"id": "consumer_high_2", "name": "high-end consumer 2", "size": 500, "avg_budget": 8300, "price_sensitivity": 0.28, "trust_threshold": 0.72, "impulse_factor": 0.38},
        {"id": "consumer_high_3", "name": "high-end consumer 3", "size": 500, "avg_budget": 8200, "price_sensitivity": 0.27, "trust_threshold": 0.73, "impulse_factor": 0.37},
        {"id": "consumer_high_4", "name": "high-end consumer 4", "size": 500, "avg_budget": 8100, "price_sensitivity": 0.29, "trust_threshold": 0.71, "impulse_factor": 0.39},
        {"id": "consumer_high_5", "name": "high-end consumer 5", "size": 500, "avg_budget": 8000, "price_sensitivity": 0.30, "trust_threshold": 0.70, "impulse_factor": 0.40},
        {"id": "consumer_high_6", "name": "high-end consumer 6", "size": 500, "avg_budget": 7900, "price_sensitivity": 0.31, "trust_threshold": 0.69, "impulse_factor": 0.41},
        {"id": "consumer_high_7", "name": "high-end consumer 7", "size": 500, "avg_budget": 7800, "price_sensitivity": 0.32, "trust_threshold": 0.68, "impulse_factor": 0.42},
        {"id": "consumer_high_8", "name": "high-end consumer 8", "size": 500, "avg_budget": 7700, "price_sensitivity": 0.33, "trust_threshold": 0.67, "impulse_factor": 0.43},
        {"id": "consumer_high_9", "name": "high-end consumer 9", "size": 500, "avg_budget": 7600, "price_sensitivity": 0.34, "trust_threshold": 0.66, "impulse_factor": 0.44},
        {"id": "consumer_high_10", "name": "high-end consumer 10", "size": 500, "avg_budget": 7500, "price_sensitivity": 0.35, "trust_threshold": 0.65, "impulse_factor": 0.45},
        
        # 中端消费者组 (10个)
        {"id": "consumer_mid_1", "name": "middle consumer 1", "size": 2000, "avg_budget": 3500, "price_sensitivity": 0.55, "trust_threshold": 0.65, "impulse_factor": 0.45},
        {"id": "consumer_mid_2", "name": "middle consumer 2", "size": 2000, "avg_budget": 3400, "price_sensitivity": 0.57, "trust_threshold": 0.63, "impulse_factor": 0.47},
        {"id": "consumer_mid_3", "name": "middle consumer 3", "size": 2000, "avg_budget": 3300, "price_sensitivity": 0.58, "trust_threshold": 0.62, "impulse_factor": 0.48},
        {"id": "consumer_mid_4", "name": "middle consumer 4", "size": 2000, "avg_budget": 3200, "price_sensitivity": 0.59, "trust_threshold": 0.61, "impulse_factor": 0.49},
        {"id": "consumer_mid_5", "name": "middle consumer 5", "size": 2000, "avg_budget": 3100, "price_sensitivity": 0.60, "trust_threshold": 0.60, "impulse_factor": 0.50},
        {"id": "consumer_mid_6", "name": "middle consumer 6", "size": 2000, "avg_budget": 3000, "price_sensitivity": 0.61, "trust_threshold": 0.59, "impulse_factor": 0.51},
        {"id": "consumer_mid_7", "name": "middle consumer 7", "size": 2000, "avg_budget": 2900, "price_sensitivity": 0.62, "trust_threshold": 0.58, "impulse_factor": 0.52},
        {"id": "consumer_mid_8", "name": "middle consumer 8", "size": 2000, "avg_budget": 2800, "price_sensitivity": 0.63, "trust_threshold": 0.57, "impulse_factor": 0.53},
        {"id": "consumer_mid_9", "name": "middle consumer 9", "size": 2000, "avg_budget": 2700, "price_sensitivity": 0.64, "trust_threshold": 0.56, "impulse_factor": 0.54},
        {"id": "consumer_mid_10", "name": "middle consumer 10", "size": 2000, "avg_budget": 2600, "price_sensitivity": 0.65, "trust_threshold": 0.55, "impulse_factor": 0.55},
        
        # 普通消费者组 (10个)
        {"id": "consumer_normal_1", "name": "normal consumer 1", "size": 5000, "avg_budget": 1200, "price_sensitivity": 0.75, "trust_threshold": 0.55, "impulse_factor": 0.55},
        {"id": "consumer_normal_2", "name": "normal consumer 2", "size": 5000, "avg_budget": 1150, "price_sensitivity": 0.76, "trust_threshold": 0.54, "impulse_factor": 0.56},
        {"id": "consumer_normal_3", "name": "normal consumer 3", "size": 5000, "avg_budget": 1100, "price_sensitivity": 0.77, "trust_threshold": 0.53, "impulse_factor": 0.57},
        {"id": "consumer_normal_4", "name": "normal consumer 4", "size": 5000, "avg_budget": 1050, "price_sensitivity": 0.78, "trust_threshold": 0.52, "impulse_factor": 0.58},
        {"id": "consumer_normal_5", "name": "normal consumer 5", "size": 5000, "avg_budget": 1000, "price_sensitivity": 0.80, "trust_threshold": 0.50, "impulse_factor": 0.60},
        {"id": "consumer_normal_6", "name": "normal consumer 6", "size": 5000, "avg_budget": 950, "price_sensitivity": 0.81, "trust_threshold": 0.49, "impulse_factor": 0.61},
        {"id": "consumer_normal_7", "name": "normal consumer 7", "size": 5000, "avg_budget": 900, "price_sensitivity": 0.82, "trust_threshold": 0.48, "impulse_factor": 0.62},
        {"id": "consumer_normal_8", "name": "normal consumer 8", "size": 5000, "avg_budget": 850, "price_sensitivity": 0.83, "trust_threshold": 0.47, "impulse_factor": 0.63},
        {"id": "consumer_normal_9", "name": "normal consumer 9", "size": 5000, "avg_budget": 800, "price_sensitivity": 0.84, "trust_threshold": 0.46, "impulse_factor": 0.64},
        {"id": "consumer_normal_10", "name": "normal consumer 10", "size": 5000, "avg_budget": 750, "price_sensitivity": 0.85, "trust_threshold": 0.45, "impulse_factor": 0.65}
    ]
}