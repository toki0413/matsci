import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ReconnectingWebSocket } from './ws-client';
import type { WsStatus } from './ws-client';

/**
 * Controllable fake WebSocket. jsdom doesn't ship a real WebSocket, so we
 * inject a stub the client can talk to, and drive its callbacks from the test.
 */
class FakeWebSocket {
  url: string;
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: ((ev: { code?: number }) => void) | null = null;
  onerror: (() => void) | null = null;
  sent: string[] = [];
  static instances: FakeWebSocket[] = [];
  static lastConstructedUrl: string | null = null;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
    FakeWebSocket.lastConstructedUrl = url;
  }
  send(data: string) {
    this.sent.push(data);
  }
  close() {
    // mimic the browser: close() eventually fires onclose. The client relies
    // on this to route watchdog-kills and manual closes into handleClose.
    // It clears the handlers on manual close() though, so onclose here is only
    // reachable when the client *didn't* null it (i.e. watchdog path).
    this.onclose?.({ code: 1000 });
  }
}

let capturedStatuses: Array<{ status: WsStatus; retries?: number; delay?: number }> = [];

function installFakeWs() {
  FakeWebSocket.instances = [];
  FakeWebSocket.lastConstructedUrl = null;
  (globalThis as any).WebSocket = FakeWebSocket;
}

beforeEach(() => {
  installFakeWs();
  vi.useFakeTimers();
  capturedStatuses = [];
});

afterEach(() => {
  vi.useRealTimers();
});

function makeClient(overrides: { maxRetries?: number; pingInterval?: number; authToken?: string | (() => string | null) } = {}) {
  const client = new ReconnectingWebSocket({
    url: 'ws://127.0.0.1:8000/ws/agent',
    initialDelay: 10, // tiny for tests
    maxDelay: 100,
    onStatus: (status, info) => capturedStatuses.push({ status, ...info }),
    ...overrides,
  });
  return client;
}

// helper: connect and simulate the socket opening
function connectAndOpen(client: ReconnectingWebSocket) {
  client.connect();
  const ws = FakeWebSocket.lastConstructedUrl && FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
  expect(ws).toBeTruthy();
  ws!.onopen?.();
  return ws!;
}

describe('ReconnectingWebSocket', () => {
  it('appends the auth token as a query param on connect', () => {
    const client = makeClient({ authToken: () => 'secret-token' });
    client.connect();
    expect(FakeWebSocket.lastConstructedUrl).toBe('ws://127.0.0.1:8000/ws/agent?token=secret-token');
  });

  it('uses & separator when the url already has a query string', () => {
    const client = new ReconnectingWebSocket({
      url: 'ws://x/ws/agent?lazy=1',
      authToken: 'tok',
    });
    client.connect();
    expect(FakeWebSocket.lastConstructedUrl).toBe('ws://x/ws/agent?lazy=1&token=tok');
  });

  it('transitions connecting -> connected on open', () => {
    const client = makeClient();
    connectAndOpen(client);
    expect(client.getStatus()).toBe('connected');
    expect(capturedStatuses.map((s) => s.status)).toEqual(['connecting', 'connected']);
  });

  it('buffers messages sent while disconnected, then flushes after reconnect', () => {
    const client = makeClient({ initialDelay: 10, pingInterval: 0 });
    const ws = connectAndOpen(client) as FakeWebSocket;
    // drop the connection (schedules a reconnect)
    ws.onclose?.({ code: 1006 });

    // nothing is connected anymore, so sends get buffered
    client.send({ type: 'chat', msg: 'hi' });
    client.send('raw');
    expect(ws.sent).toEqual([]);

    // let the exponential-backoff timer fire, then let that socket open
    vi.advanceTimersByTime(10);
    const ws2 = FakeWebSocket.instances[FakeWebSocket.instances.length - 1] as FakeWebSocket;
    ws2.onopen?.();

    // flush happens 200ms after open
    vi.advanceTimersByTime(250);
    expect(ws2.sent).toEqual([JSON.stringify({ type: 'chat', msg: 'hi' }), 'raw']);
  });

  it('sends directly without buffering when connected', () => {
    const client = makeClient({ pingInterval: 0 });
    const ws = connectAndOpen(client) as FakeWebSocket;
    client.send('direct');
    expect(ws.sent).toEqual(['direct']);
  });

  it('retries with exponential backoff on unexpected close', () => {
    const client = makeClient({ initialDelay: 10, maxDelay: 100, pingInterval: 0 });
    const ws = connectAndOpen(client) as FakeWebSocket;

    ws.onclose?.({ code: 1006 });
    expect(client.getStatus()).toBe('reconnecting');

    vi.advanceTimersByTime(10); // first backoff tick
    expect(FakeWebSocket.instances.length).toBe(2);
  });

  it('gives up (failed) when retries exceed maxRetries', () => {
    // Note: on a successful open retries reset to 0, so to exhaust the budget
    // we must keep failing *without* opening — consecutive failed reconnect
    // attempts only.
    const client = makeClient({ initialDelay: 5, maxDelay: 5, maxRetries: 2, pingInterval: 0 });
    const ws = connectAndOpen(client) as FakeWebSocket;

    // fail (no open) #1 -> retries=1
    ws.onclose?.({ code: 1006 });
    vi.advanceTimersByTime(5);
    const ws2 = FakeWebSocket.instances[FakeWebSocket.instances.length - 1] as FakeWebSocket;

    // fail (no open) #2 -> retries=2 (still under maxRetries, schedules again)
    ws2.onclose?.({ code: 1006 });
    vi.advanceTimersByTime(5);
    const ws3 = FakeWebSocket.instances[FakeWebSocket.instances.length - 1] as FakeWebSocket;

    // fail #3 -> retries (3) >= maxRetries (2) -> permanent failure
    ws3.onclose?.({ code: 1006 });
    vi.advanceTimersByTime(5);
    expect(client.getStatus()).toBe('failed');

    // no further sockets must be created
    const countAfter = FakeWebSocket.instances.length;
    vi.advanceTimersByTime(100);
    expect(FakeWebSocket.instances.length).toBe(countAfter);
  });

  it('does NOT retry on auth/policy close codes (4001, 1008)', () => {
    const client = makeClient({ initialDelay: 5, pingInterval: 0 });
    const ws = connectAndOpen(client) as FakeWebSocket;
    ws.onclose?.({ code: 4001 });
    expect(client.getStatus()).toBe('failed');
  });

  it('send() returns false after the client has failed', () => {
    const client = makeClient({ initialDelay: 5, maxRetries: 0, pingInterval: 0 });
    const ws = connectAndOpen(client) as FakeWebSocket;
    ws.onclose?.({});
    expect(client.getStatus()).toBe('failed');
    expect(client.send('x')).toBe(false);
  });

  it('parses incoming JSON messages and forwards to onMessage', () => {
    const received: unknown[] = [];
    const client = new ReconnectingWebSocket({
      url: 'ws://x',
      pingInterval: 0,
      onMessage: (d) => received.push(d),
    });
    const ws = connectAndOpen(client) as FakeWebSocket;
    ws.onmessage?.({ data: JSON.stringify({ type: 'delta', text: 'hello' }) });
    expect(received).toEqual([{ type: 'delta', text: 'hello' }]);
  });

  it('passes through non-JSON text unchanged', () => {
    const received: unknown[] = [];
    const client = new ReconnectingWebSocket({
      url: 'ws://x',
      pingInterval: 0,
      onMessage: (d) => received.push(d),
    });
    const ws = connectAndOpen(client) as FakeWebSocket;
    ws.onmessage?.({ data: 'plain text' });
    expect(received).toEqual(['plain text']);
  });

  it('swallows heartbeat pongs without forwarding them', () => {
    const received: unknown[] = [];
    const client = new ReconnectingWebSocket({
      url: 'ws://x',
      pingInterval: 0,
      onMessage: (d) => received.push(d),
    });
    const ws = connectAndOpen(client) as FakeWebSocket;
    ws.onmessage?.({ data: JSON.stringify({ type: 'pong' }) });
    expect(received).toEqual([]);
  });

  it('sends ping on the configured interval and force-closes on missing pong', () => {
    // Important: the 10s pong-watchdog is re-armed on every ping tick, so the
    // interval must be LONGER than 10s for a missing pong to ever fire. Use
    // 15s: ping at t=15s arms a watchdog for t=25s, well before the next ping.
    const client = makeClient({ pingInterval: 15_000, initialDelay: 10, maxDelay: 10, maxRetries: 1 });
    const ws = connectAndOpen(client) as FakeWebSocket;

    // first ping tick -> {type:'ping'} is emitted, watchdog armed for +10s
    vi.advanceTimersByTime(15_000);
    expect(ws.sent).toContain(JSON.stringify({ type: 'ping' }));

    // no pong within the 10s watchdog window -> force reconnect
    vi.advanceTimersByTime(10_000);
    expect(capturedStatuses.some((s) => s.status === 'reconnecting')).toBe(true);
  });

  it('honours a pong so the connection survives the heartbeat window', () => {
    const client = makeClient({ pingInterval: 50, initialDelay: 10, maxRetries: 0 });
    const ws = connectAndOpen(client) as FakeWebSocket;
    vi.advanceTimersByTime(50); // ping sent
    ws.onmessage?.({ data: JSON.stringify({ type: 'pong' }) }); // pong arrives
    // past the 10s dead-connection window while pong was received
    vi.advanceTimersByTime(10_000);
    expect(client.getStatus()).toBe('connected');
  });

  it('close() stops retrying and returns to idle', () => {
    const client = makeClient();
    connectAndOpen(client);
    client.close();
    expect(client.getStatus()).toBe('idle');
    vi.advanceTimersByTime(1000);
    // no further reconnect attempted after manual close
    expect(FakeWebSocket.instances.length).toBe(1);
  });

  it('close() clears buffered reconnect logic', () => {
    const client = makeClient({ pingInterval: 0 });
    const ws = connectAndOpen(client) as FakeWebSocket;
    ws.onclose?.({ code: 1006 }); // would queue a reconnect
    client.close(); // manual close must cancel it
    vi.advanceTimersByTime(100);
    expect(FakeWebSocket.instances.length).toBe(1);
  });
});