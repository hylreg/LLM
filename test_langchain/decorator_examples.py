"""
Python 装饰器示例
演示装饰器的各种作用和用法
"""

import time
from functools import wraps
from typing import Callable


# ========== 1. 基础装饰器：功能增强 ==========

def timer(func: Callable) -> Callable:
    """计时装饰器：测量函数执行时间"""
    @wraps(func)  # 保留原函数的元数据
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 执行时间: {end - start:.4f} 秒")
        return result
    return wrapper


def logger(func: Callable) -> Callable:
    """日志装饰器：记录函数调用"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"调用函数: {func.__name__}, 参数: args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"函数 {func.__name__} 执行完成，返回值: {result}")
        return result
    return wrapper


# ========== 2. 带参数的装饰器 ==========

def retry(max_attempts: int = 3):
    """重试装饰器：失败时自动重试"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    print(f"第 {attempt} 次尝试失败: {e}, 正在重试...")
            return None
        return wrapper
    return decorator


def cache(func: Callable) -> Callable:
    """简单缓存装饰器：缓存函数结果"""
    cache_dict = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 使用参数作为缓存键
        key = str(args) + str(sorted(kwargs.items()))
        if key in cache_dict:
            print(f"使用缓存结果: {func.__name__}")
            return cache_dict[key]
        result = func(*args, **kwargs)
        cache_dict[key] = result
        return result
    return wrapper


# ========== 3. 权限检查装饰器 ==========

def require_auth(func: Callable) -> Callable:
    """权限检查装饰器：检查用户是否已认证"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 模拟检查用户是否登录
        user_logged_in = True  # 实际应用中从session或token获取
        if not user_logged_in:
            raise PermissionError("需要登录才能访问此功能")
        return func(*args, **kwargs)
    return wrapper


# ========== 4. 多个装饰器组合使用 ==========

@timer
@logger
@cache
def calculate_fibonacci(n: int) -> int:
    """计算斐波那契数列（使用装饰器增强）"""
    if n <= 1:
        return n
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)


# ========== 5. 类装饰器 ==========

class CountCalls:
    """类装饰器：统计函数调用次数"""
    def __init__(self, func: Callable):
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} 被调用了 {self.count} 次")
        return self.func(*args, **kwargs)


# ========== 6. 实际应用示例 ==========

@timer
@retry(max_attempts=3)
def fetch_data(url: str) -> str:
    """模拟获取数据（可能失败）"""
    import random
    if random.random() < 0.3:  # 30% 概率失败
        raise ConnectionError("网络连接失败")
    return f"从 {url} 获取的数据"


@require_auth
def delete_user(user_id: int) -> str:
    """删除用户（需要权限）"""
    return f"用户 {user_id} 已被删除"


@CountCalls
def greet(name: str) -> str:
    """问候函数（统计调用次数）"""
    return f"Hello, {name}!"


# ========== 7. 装饰器在 LangChain 中的应用 ==========
# 类似你代码中的 @tool 装饰器

def tool(func: Callable) -> Callable:
    """
    模拟 LangChain 的 @tool 装饰器
    将普通函数转换为工具，使其可被 Agent 调用
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 在实际的 LangChain 中，这里会：
        # 1. 提取函数的参数信息（类型、描述等）
        # 2. 将函数包装成 Tool 对象
        # 3. 注册到 Agent 的工具列表中
        print(f"[工具调用] {func.__name__}")
        result = func(*args, **kwargs)
        print(f"[工具结果] {result}")
        return result
    return wrapper


@tool
def get_weather(city: str) -> str:
    """获取天气信息（工具函数）"""
    return f"The weather in {city} is sunny."


# ========== 主函数：演示各种装饰器 ==========

if __name__ == "__main__":
    print("=" * 50)
    print("1. 计时和日志装饰器")
    print("=" * 50)
    
    @timer
    @logger
    def add(a: int, b: int) -> int:
        return a + b
    
    result = add(3, 5)
    print()
    
    print("=" * 50)
    print("2. 缓存装饰器（加速递归计算）")
    print("=" * 50)
    fib_result = calculate_fibonacci(10)
    print(f"结果: {fib_result}\n")
    
    print("=" * 50)
    print("3. 重试装饰器")
    print("=" * 50)
    data = fetch_data("https://api.example.com/data")
    print(f"获取的数据: {data}\n")
    
    print("=" * 50)
    print("4. 权限检查装饰器")
    print("=" * 50)
    try:
        delete_user(123)
    except PermissionError as e:
        print(f"错误: {e}\n")
    
    print("=" * 50)
    print("5. 类装饰器（统计调用次数）")
    print("=" * 50)
    greet("Alice")
    greet("Bob")
    greet("Charlie")
    print()
    
    print("=" * 50)
    print("6. 工具装饰器（类似 LangChain @tool）")
    print("=" * 50)
    weather = get_weather("Beijing")
    print(f"天气信息: {weather}")

