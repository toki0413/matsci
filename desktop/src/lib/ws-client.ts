/**
 * 带指数退避重连和流式缓冲的 WebSocket 客户端。
 *
 * 设计要点：
 * - 断线后按 1s → 2s → 4s → ... → 30s 指数退避重连
 * - 连接状态对外可观测（connecting / connected / reconnecting / failed）
 * - 断线期间发送的消息暂存到队列，重连成功后批量 flush
 * - 内置心跳，避免被中间代理掐断空闲连接
 */

export type WsStatus = 'idle' | 'connecting' | 'connected' | 'reconnecting' | 'failed';

export interface WsClientOptions {
  /**
   * ws:// 或 wss:// 完整地址. 传函数时每次(重)连都会重新求值,
   * 这样后端换端口后重连能自动跟上新地址, 而不是钉死在旧端口.
   */
  url: string | (() => string | null);
  /** 首次重连延迟，默认 1000ms */
  initialDelay?: number;
  /** 退避上限，默认 30000ms */
  maxDelay?: number;
  /** 最大重连次数，默认 Infinity */
  maxRetries?: number;
  /** 认证拒绝(4001/1008)时的慢速恢复重连次数，默认 3；用尽后进入 failed */
  maxAuthRetries?: number;
  /** 认证拒绝后的恢复窗口，默认 30s */
  recoveryDelay?: number;
  /** 心跳间隔，默认 30s；设为 0 关闭 */
  pingInterval?: number;
  /** 发 ping 后等待 pong 的容忍时长，默认 20s；超时才判定连接死亡。 */
  pingTimeout?: number;
  /** 状态变更回调 */
  onStatus?: (status: WsStatus, info?: { retries: number; delay?: number }) => void;
  /** 收到消息回调（已 JSON.parse） */
  onMessage?: (data: unknown) => void;
  /** 连接成功回调（含首次和重连） */
  onConnected?: () => void;
  /** 认证 token（JWT 或 API key），通过 ?token= 查询参数传递 */
  authToken?: string | (() => string | null);
}

export class ReconnectingWebSocket {
  private ws: WebSocket | null = null;
  private status: WsStatus = 'idle';
  private retries = 0;
  private authRetries = 0;
  private buffer: string[] = [];
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private pongTimer: ReturnType<typeof setTimeout> | null = null;
  private manuallyClosed = false;
  private flushTimer: ReturnType<typeof setTimeout> | null = null;

  private readonly opts: Required<Omit<WsClientOptions, 'onStatus' | 'onMessage' | 'onConnected' | 'url' | 'authToken'>> &
    Pick<WsClientOptions, 'onStatus' | 'onMessage' | 'onConnected' | 'url' | 'authToken'>;

  constructor(options: WsClientOptions) {
    this.opts = {
      url: options.url,
      initialDelay: options.initialDelay ?? 1000,
      maxDelay: options.maxDelay ?? 30000,
      maxRetries: options.maxRetries ?? Infinity,
      maxAuthRetries: options.maxAuthRetries ?? 3,
      recoveryDelay: options.recoveryDelay ?? 30000,
      pingInterval: options.pingInterval ?? 30000,
      pingTimeout: options.pingTimeout ?? 20000,
      onStatus: options.onStatus,
      onMessage: options.onMessage,
      onConnected: options.onConnected,
      authToken: options.authToken,
    };
  }

  /** 当前连接状态 */
  getStatus(): WsStatus {
    return this.status;
  }

  /** 主动连接（首次或手动恢复） */
  connect(): void {
    this.manuallyClosed = false;
    this.openSocket('connecting');
  }

  /** 主动断开，不再自动重连 */
  close(): void {
    this.manuallyClosed = true;
    this.clearTimers();
    if (this.ws) {
      // 避免触发 onclose 里的重连逻辑
      this.ws.onclose = null;
      this.ws.onerror = null;
      this.ws.onopen = null;
      this.ws.onmessage = null;
      try {
        this.ws.close();
      } catch {
        // ignore
      }
      this.ws = null;
    }
    this.setStatus('idle');
  }

  /**
   * 发送消息。断线期间会暂存到队列，重连后自动 flush。
   * @returns true 表示已发出或已入队；false 表示已放弃（超过重连上限）
   */
  send(data: unknown): boolean {
    if (this.status === 'failed') return false;
    const payload = typeof data === 'string' ? data : JSON.stringify(data);
    if (this.ws && this.status === 'connected') {
      try {
        this.ws.send(payload);
        return true;
      } catch {
        // send 失败说明连接其实已经断了，走缓冲逻辑
      }
    }
    this.buffer.push(payload);
    return true;
  }

  // ── 内部实现 ──────────────────────────────────────────────────

  private openSocket(initialStatus: WsStatus): void {
    if (this.ws) {
      try {
        this.ws.close();
      } catch {
        // ignore
      }
      this.ws = null;
    }
    this.setStatus(initialStatus);

    // Build URL with auth token if configured
    let url = typeof this.opts.url === 'function'
      ? (this.opts.url() ?? '')
      : this.opts.url;
    const tokenVal = typeof this.opts.authToken === 'function'
      ? this.opts.authToken()
      : this.opts.authToken;
    if (tokenVal) {
      const sep = url.includes('?') ? '&' : '?';
      url = `${url}${sep}token=${encodeURIComponent(tokenVal)}`;
    }

    try {
      this.ws = new WebSocket(url);
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws.onopen = () => this.handleOpen();
    this.ws.onmessage = (ev) => this.handleMessage(ev);
    this.ws.onerror = () => {
      // error 后通常会跟一个 onclose，重连交给 onclose 触发
    };
    this.ws.onclose = (ev: CloseEvent) => this.handleClose(ev);
  }

  private handleOpen(): void {
    this.retries = 0;
    this.authRetries = 0;
    this.setStatus('connected');
    this.startPing();
    // 重连后先等一个小窗口让服务端就绪，再批量 flush 缓冲
    if (this.buffer.length > 0) {
      if (this.flushTimer) clearTimeout(this.flushTimer);
      this.flushTimer = setTimeout(() => this.flushBuffer(), 200);
    }
    this.opts.onConnected?.();
  }

  private handleMessage(ev: MessageEvent): void {
    let data: unknown = ev.data;
    if (typeof data === 'string') {
      try {
        data = JSON.parse(data);
      } catch {
        // 非 JSON 文本原样透传
      }
    }
    if (this.isPong(data)) {
      // got heartbeat reply — connection is alive
      if (this.pongTimer) { clearTimeout(this.pongTimer); this.pongTimer = null; }
      return;
    }
    this.opts.onMessage?.(data);
  }

  private handleClose(ev?: CloseEvent): void {
    this.stopPing();
    this.ws = null;
    if (this.manuallyClosed) return;
    // 4001 = auth failure from backend, 1008 = policy violation (WS auth
    // timeout in non-dev mode). Both can be transient while the backend is
    // still coming up / issuing tokens on first boot, so instead of locking
    // into failed forever we retry slowly a bounded number of times.
    if (ev?.code === 4001 || ev?.code === 1008) {
      if (this.authRetries >= this.opts.maxAuthRetries) {
        this.setStatus('failed');
        return;
      }
      this.authRetries += 1;
      this.setStatus('reconnecting', {
        retries: this.authRetries,
        delay: this.opts.recoveryDelay,
      });
      if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
      this.reconnectTimer = setTimeout(() => {
        this.openSocket('reconnecting');
      }, this.opts.recoveryDelay);
      return;
    }
    this.scheduleReconnect();
  }

  private scheduleReconnect(): void {
    if (this.manuallyClosed) return;
    if (this.retries >= this.opts.maxRetries) {
      this.setStatus('failed');
      return;
    }
    // 指数退避：delay = min(initial * 2^retries, max)
    const delay = Math.min(
      this.opts.initialDelay * Math.pow(2, this.retries),
      this.opts.maxDelay,
    );
    this.retries += 1;
    this.setStatus('reconnecting', { retries: this.retries, delay });
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = setTimeout(() => {
      this.openSocket('reconnecting');
    }, delay);
  }

  private flushBuffer(): void {
    if (!this.ws || this.status !== 'connected') return;
    const pending = this.buffer.splice(0);
    for (const msg of pending) {
      try {
        this.ws.send(msg);
      } catch {
        // flush 中途断了，把剩余的放回队列等下次重连
        this.buffer.unshift(msg);
        break;
      }
    }
  }

  private startPing(): void {
    if (this.opts.pingInterval <= 0) return;
    this.stopPing();
    this.pingTimer = setInterval(() => {
      this.send({ type: 'ping' });
      // 超时未收到 pong 判定连接死亡, 主动关闭触发重连. 间隔给得比原 10s
      // 宽 (默认 20s): 强负载下 pong 帧偶尔延迟, 误关健康连接反而加剧断连.
      if (this.pongTimer) clearTimeout(this.pongTimer);
      this.pongTimer = setTimeout(() => {
        // ponytail: silent death is worse than a reconnect cycle
        try { this.ws?.close(); } catch { /* will trigger handleClose */ }
      }, this.opts.pingTimeout);
    }, this.opts.pingInterval);
  }

  private stopPing(): void {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
    if (this.pongTimer) {
      clearTimeout(this.pongTimer);
      this.pongTimer = null;
    }
  }

  private clearTimers(): void {
    this.stopPing();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
  }

  private setStatus(status: WsStatus, info?: { retries: number; delay?: number }): void {
    this.status = status;
    this.opts.onStatus?.(status, info);
  }

  private isPong(data: unknown): boolean {
    return typeof data === 'object' && data !== null && (data as { type?: string }).type === 'pong';
  }
}


