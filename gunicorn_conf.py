import multiprocessing
import os

# Gunicorn 설정 
workers_per_core_str = os.getenv("WORKERS_PER_CORE", "1")
max_workers_str = os.getenv("MAX_WORKERS")
use_max_workers = None
if max_workers_str:
    use_max_workers = int(max_workers_str)
web_concurrency_str = os.getenv("WEB_CONCURRENCY", None)

host = os.getenv("HOST", "0.0.0.0")
port = os.getenv("PORT", "8080")
bind_env = os.getenv("BIND", None)
use_loglevel = os.getenv("LOG_LEVEL", "info")
if bind_env:
    use_bind = bind_env
else:
    use_bind = f"{host}:{port}"

cores = multiprocessing.cpu_count()
workers_per_core = float(workers_per_core_str)
default_web_concurrency = 2  # t3.medium에 최적화된 기본값
if web_concurrency_str:
    web_concurrency = int(web_concurrency_str)
    assert web_concurrency > 0
else:
    web_concurrency = max(int(default_web_concurrency), 2)
    if use_max_workers:
        web_concurrency = min(web_concurrency, use_max_workers)

# Gunicorn 설정 변수
loglevel = use_loglevel
workers = web_concurrency
bind = use_bind
keepalive = 120
timeout = 120
worker_class = "uvicorn.workers.UvicornWorker"

# 로그 설정
errorlog = "-"
accesslog = "-"

# 기타 설정
graceful_timeout = 120
forwarded_allow_ips = "*"
