import { BaseClient, ClientConfig, ClientRequest, ClientResponse } from './base_client';
import WebSocket from 'ws';
import { CandleData } from '../models/market_data';
import { Order, Position } from '../models/position';

export interface OsEngineConfig extends ClientConfig {
  wsHost: string;
  wsPort: number;
  useWebSocket: boolean;
  reconnectInterval: number;
  heartbeatInterval: number;
}

export interface OsEngineInstrument {
  securityName: string;
  securityType: string;
  lot: number;
  priceStep: number;
  priceStepCost: number;
  go: number;
  expiry?: string;
  strike?: number;
}

export interface OsEnginePortfolio {
  number: string;
  valueBegin: number;
  valueCurrent: number;
  valueBlocked: number;
  positions: OsEnginePosition[];
}

export interface OsEnginePosition {
  securityNameCode: string;
  valueBegin: number;
  valueCurrent: number;
  valueBlocked: number;
  openVolume: number;
  side: 'Buy' | 'Sell';
}

export interface OsEngineOrder {
  number: number;
  numberUser: number;
  numberMarket: string;
  securityNameCode: string;
  side: 'Buy' | 'Sell';
  state: 'None' | 'Pending' | 'Done' | 'Partial' | 'Fail' | 'Cancel';
  volume: number;
  volumeExecute: number;
  price: number;
  typeTime: 'GTC' | 'DAY' | 'IOC' | 'FOK';
  serverType: string;
  portfolioNumber: string;
  timeCreate: string;
  timeCallBack: string;
  timeDone: string;
  timeCancel: string;
}

export interface OsEngineTrade {
  number: string;
  numberOrderParent: string;
  securityNameCode: string;
  side: 'Buy' | 'Sell';
  volume: number;
  price: number;
  time: string;
  microseconds: number;
}

export interface OsEngineCandle {
  timeStart: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export class OsEngineClient extends BaseClient {
  private osConfig: OsEngineConfig;
  private ws: WebSocket | null = null;
  private wsConnected: boolean = false;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private subscriptions: Set<string> = new Set();
  private pendingRequests: Map<string, { resolve: Function; reject: Function }> = new Map();

  constructor(config: OsEngineConfig) {
    const baseConfig: ClientConfig = {
      ...config,
      timeout: config.timeout || 30000,
      retryAttempts: config.retryAttempts || 3,
      retryDelay: config.retryDelay || 2000
    };

    super(baseConfig);
    this.osConfig = config;
  }

  async connect(): Promise<void> {
    try {
      if (this.osConfig.useWebSocket) {
        await this.connectWebSocket();
      }
      
      // Test HTTP connection
      const response = await this.request('GET', '/ping');
      if (response.success) {
        this.emit('connected');
      } else {
        throw new Error('Failed to connect to OsEngine');
      }
    } catch (error) {
      this.emit('error', error);
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.wsConnected = false;
    this.emit('disconnected');
  }

  async isHealthy(): Promise<boolean> {
    try {
      const response = await this.request('GET', '/ping');
      return response.success && this.wsConnected;
    } catch {
      return false;
    }
  }

  protected async executeRequest<T>(request: ClientRequest): Promise<ClientResponse<T>> {
    const url = `${this.config.protocol}://${this.config.host}:${this.config.port || 8080}${request.endpoint}`;
    
    try {
      const fetchOptions: RequestInit = {
        method: request.method,
        headers: request.headers as Record<string, string>
      };

      if (request.params && ['POST', 'PUT', 'PATCH'].includes(request.method)) {
        fetchOptions.body = JSON.stringify(request.params);
        fetchOptions.headers = {
          ...fetchOptions.headers,
          'Content-Type': 'application/json'
        };
      }

      const response = await fetch(url, fetchOptions);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const contentType = response.headers.get('content-type');
      let data: any;

      if (contentType && contentType.includes('application/json')) {
        data = await response.json();
      } else {
        data = await response.text();
      }

      return {
        id: request.id,
        success: true,
        data: data as T,
        timestamp: Date.now(),
        duration: 0
      };

    } catch (error) {
      throw error;
    }
  }

  private async connectWebSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      const wsUrl = `ws://${this.osConfig.wsHost}:${this.osConfig.wsPort}`;
      this.ws = new WebSocket(wsUrl);

      this.ws.on('open', () => {
        this.wsConnected = true;
        this.setupHeartbeat();
        this.emit('ws:connected');
        resolve();
      });

      this.ws.on('message', (data) => {
        this.handleWebSocketMessage(data);
      });

      this.ws.on('close', () => {
        this.wsConnected = false;
        this.emit('ws:disconnected');
        this.scheduleReconnect();
      });

      this.ws.on('error', (error) => {
        this.emit('ws:error', error);
        if (!this.wsConnected) {
          reject(error);
        }
      });

      setTimeout(() => {
        if (!this.wsConnected) {
          reject(new Error('WebSocket connection timeout'));
        }
      }, this.config.timeout);
    });
  }

  private handleWebSocketMessage(data: WebSocket.Data): void {
    try {
      const message = JSON.parse(data.toString());
      
      switch (message.type) {
        case 'candle':
          this.emit('candle', this.convertOsCandle(message.data));
          break;
        case 'order':
          this.emit('order', message.data);
          break;
        case 'trade':
          this.emit('trade', message.data);
          break;
        case 'position':
          this.emit('position', message.data);
          break;
        case 'portfolio':
          this.emit('portfolio', message.data);
          break;
        case 'response':
          this.handleResponse(message);
          break;
        case 'ping':
          this.sendPong();
          break;
        default:
          this.emit('ws:message', message);
      }
    } catch (error) {
      this.emit('ws:parse_error', error);
    }
  }

  private handleResponse(message: any): void {
    const { id, success, data, error } = message;
    const pending = this.pendingRequests.get(id);
    
    if (pending) {
      this.pendingRequests.delete(id);
      if (success) {
        pending.resolve(data);
      } else {
        pending.reject(new Error(error || 'Unknown error'));
      }
    }
  }

  private setupHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.ws && this.wsConnected) {
        this.ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
      }
    }, this.osConfig.heartbeatInterval || 30000);
  }

  private sendPong(): void {
    if (this.ws && this.wsConnected) {
      this.ws.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    
    this.reconnectTimer = setTimeout(async () => {
      this.reconnectTimer = null;
      try {
        await this.connectWebSocket();
        // Resubscribe to previous subscriptions
        for (const subscription of this.subscriptions) {
          await this.resubscribe(subscription);
        }
      } catch (error) {
        this.emit('reconnect:failed', error);
        this.scheduleReconnect();
      }
    }, this.osConfig.reconnectInterval || 5000);
  }

  private async resubscribe(subscription: string): Promise<void> {
    const [type, symbol, timeframe] = subscription.split(':');
    
    switch (type) {
      case 'candles':
        await this.subscribeToCandles(symbol, timeframe);
        break;
      case 'orders':
        await this.subscribeToOrders();
        break;
      case 'trades':
        await this.subscribeToTrades();
        break;
    }
  }

  // Public API Methods

  async getInstruments(): Promise<OsEngineInstrument[]> {
    const response = await this.request<OsEngineInstrument[]>('GET', '/instruments');
    return response.success ? response.data || [] : [];
  }

  async getPortfolios(): Promise<OsEnginePortfolio[]> {
    const response = await this.request<OsEnginePortfolio[]>('GET', '/portfolios');
    return response.success ? response.data || [] : [];
  }

  async getCandles(
    symbol: string, 
    timeframe: string, 
    from?: Date, 
    to?: Date
  ): Promise<CandleData[]> {
    const params: any = { symbol, timeframe };
    if (from) params.from = from.toISOString();
    if (to) params.to = to.toISOString();

    const response = await this.request<OsEngineCandle[]>('GET', '/candles', params);
    
    if (response.success && response.data) {
      return response.data.map(this.convertOsCandle);
    }
    
    return [];
  }

  async sendOrder(
    symbol: string,
    side: 'Buy' | 'Sell',
    volume: number,
    price: number,
    portfolio: string,
    orderType: 'Market' | 'Limit' = 'Limit'
  ): Promise<OsEngineOrder> {
    const orderData = {
      securityNameCode: symbol,
      side,
      volume,
      price,
      portfolioNumber: portfolio,
      orderType,
      timeInForce: 'GTC'
    };

    const response = await this.request<OsEngineOrder>('POST', '/orders', orderData);
    
    if (!response.success) {
      throw new Error(`Failed to send order: ${response.error?.message}`);
    }
    
    return response.data!;
  }

  async cancelOrder(orderId: number): Promise<boolean> {
    const response = await this.request('DELETE', `/orders/${orderId}`);
    return response.success;
  }

  async getOrders(portfolio?: string): Promise<OsEngineOrder[]> {
    const params = portfolio ? { portfolio } : undefined;
    const response = await this.request<OsEngineOrder[]>('GET', '/orders', params);
    return response.success ? response.data || [] : [];
  }

  async getPositions(portfolio?: string): Promise<OsEnginePosition[]> {
    const params = portfolio ? { portfolio } : undefined;
    const response = await this.request<OsEnginePosition[]>('GET', '/positions', params);
    return response.success ? response.data || [] : [];
  }

  // WebSocket Subscriptions

  async subscribeToCandles(symbol: string, timeframe: string): Promise<void> {
    if (!this.wsConnected) {
      throw new Error('WebSocket not connected');
    }

    const subscriptionKey = `candles:${symbol}:${timeframe}`;
    
    return new Promise((resolve, reject) => {
      const requestId = `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      this.pendingRequests.set(requestId, { resolve, reject });
      
      this.ws!.send(JSON.stringify({
        type: 'subscribe',
        id: requestId,
        channel: 'candles',
        symbol,
        timeframe
      }));
      
      this.subscriptions.add(subscriptionKey);
      
      setTimeout(() => {
        if (this.pendingRequests.has(requestId)) {
          this.pendingRequests.delete(requestId);
          reject(new Error('Subscription timeout'));
        }
      }, 10000);
    });
  }

  async subscribeToOrders(): Promise<void> {
    if (!this.wsConnected) {
      throw new Error('WebSocket not connected');
    }

    const subscriptionKey = 'orders';
    
    return new Promise((resolve, reject) => {
      const requestId = `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      this.pendingRequests.set(requestId, { resolve, reject });
      
      this.ws!.send(JSON.stringify({
        type: 'subscribe',
        id: requestId,
        channel: 'orders'
      }));
      
      this.subscriptions.add(subscriptionKey);
      
      setTimeout(() => {
        if (this.pendingRequests.has(requestId)) {
          this.pendingRequests.delete(requestId);
          reject(new Error('Subscription timeout'));
        }
      }, 10000);
    });
  }

  async subscribeToTrades(): Promise<void> {
    if (!this.wsConnected) {
      throw new Error('WebSocket not connected');
    }

    const subscriptionKey = 'trades';
    
    return new Promise((resolve, reject) => {
      const requestId = `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      this.pendingRequests.set(requestId, { resolve, reject });
      
      this.ws!.send(JSON.stringify({
        type: 'subscribe',
        id: requestId,
        channel: 'trades'
      }));
      
      this.subscriptions.add(subscriptionKey);
      
      setTimeout(() => {
        if (this.pendingRequests.has(requestId)) {
          this.pendingRequests.delete(requestId);
          reject(new Error('Subscription timeout'));
        }
      }, 10000);
    });
  }

  async unsubscribeFromCandles(symbol: string, timeframe: string): Promise<void> {
    if (!this.wsConnected) return;

    const subscriptionKey = `candles:${symbol}:${timeframe}`;
    
    this.ws!.send(JSON.stringify({
      type: 'unsubscribe',
      channel: 'candles',
      symbol,
      timeframe
    }));
    
    this.subscriptions.delete(subscriptionKey);
  }

  private convertOsCandle(osCandle: OsEngineCandle): CandleData {
    return {
      timestamp: new Date(osCandle.timeStart).getTime(),
      open: osCandle.open,
      high: osCandle.high,
      low: osCandle.low,
      close: osCandle.close,
      volume: osCandle.volume
    };
  }

  // Utility Methods

  getSubscriptions(): string[] {
    return Array.from(this.subscriptions);
  }

  isWebSocketConnected(): boolean {
    return this.wsConnected;
  }

  getPendingRequestCount(): number {
    return this.pendingRequests.size;
  }

  async testConnection(): Promise<{ http: boolean; websocket: boolean }> {
    let httpOk = false;
    let wsOk = false;

    try {
      const response = await this.request('GET', '/ping');
      httpOk = response.success;
    } catch {}

    wsOk = this.wsConnected;

    return { http: httpOk, websocket: wsOk };
  }
}