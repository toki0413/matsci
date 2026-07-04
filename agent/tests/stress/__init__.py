"""压力测试套件 — 验证系统在高负载下的稳定性和资源边界.

这批测试用真实 HTTP/WebSocket 请求打本地服务,不走 mock.
需要先启动服务: python -m huginn serve --port 8000
然后: python -m pytest tests/stress/ -v --tb=short -x

跟普通单测的区别:
  1. 打真实网络请求,验证全链路
  2. 持续时间长 (30s ~ 5min),验证内存泄漏和资源耗尽
  3. 并发度高 (20~50 路),验证竞态和锁竞争
  4. 监控 /metrics 端点,量化系统行为
"""
