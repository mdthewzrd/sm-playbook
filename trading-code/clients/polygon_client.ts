import { BaseClient, ClientConfig, ClientRequest, ClientResponse } from './base_client';
import WebSocket from 'ws';
import { CandleData } from '../models/market_data';

export interface PolygonConfig extends ClientConfig {
  apiKey: string;
  useWebSocket: boolean;
  wsHost?: string;
  subscriptionLimit: number;
  tier: 'basic' | 'starter' | 'developer' | 'advanced';
}

export interface PolygonTicker {
  ticker: string;
  name: string;
  market: string;
  locale: string;
  primary_exchange: string;
  type: string;
  active: boolean;
  currency_name?: string;
  cik?: string;
  composite_figi?: string;
  share_class_figi?: string;
  last_updated_utc?: string;
}

export interface PolygonAgg {
  v: number;  // volume
  vw: number; // volume weighted average price
  o: number;  // open
  c: number;  // close
  h: number;  // high
  l: number;  // low
  t: number;  // timestamp
  n: number;  // number of transactions
}

export interface PolygonQuote {
  ticker: string;
  bid: number;
  ask: number;
  bidSize: number;
  askSize: number;
  timestamp: number;
  exchange: number;
}

export interface PolygonTrade {
  ticker: string;
  price: number;
  size: number;
  timestamp: number;
  exchange: number;
  conditions: number[];
}

export interface PolygonMarketStatus {
  market: string;
  servertime: string;
  exchanges: {
    [key: string]: string;
  };
  currencies: {
    fx: string;
    crypto: string;
  };
}

export interface PolygonSubscription {
  channel: 'T' | 'Q' | 'A' | 'AM';  // Trade, Quote, Aggregate, Minute Agg
  symbols: string[];
}

export class PolygonClient extends BaseClient {
  private polygonConfig: PolygonConfig;
  private ws: WebSocket | null = null;
  private wsConnected: boolean = false;
  private subscriptions: Map<string, PolygonSubscription> = new Map();
  private reconnectTimer: NodeJS.Timeout | null = null;
  private rateLimits: Map<string, number> = new Map();

  constructor(config: PolygonConfig) {
    const baseConfig: ClientConfig = {
      ...config,
      host: 'api.polygon.io',
      protocol: 'https',
      timeout: config.timeout || 30000,
      retryAttempts: config.retryAttempts || 3,
      retryDelay: config.retryDelay || 1000,
      headers: {
        'Authorization': `Bearer ${config.apiKey}`,
        ...config.headers
      },
      rateLimiting: {
        enabled: true,
        requestsPerSecond: this.getTierRateLimit(config.tier),
        burstSize: 50
      }
    };

    super(baseConfig);
    this.polygonConfig = config;
  }

  async connect(): Promise<void> {
    try {
      // Test API connection
      const response = await this.request('GET', '/v1/marketstatus/now');
      if (!response.success) {
        throw new Error('Failed to connect to Polygon API');
      }

      if (this.polygonConfig.useWebSocket) {
        await this.connectWebSocket();
      }

      this.emit('connected');
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

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.wsConnected = false;
    this.emit('disconnected');
  }

  async isHealthy(): Promise<boolean> {
    try {
      const response = await this.request('GET', '/v1/marketstatus/now');
      return response.success;
    } catch {
      return false;
    }
  }

  protected async executeRequest<T>(request: ClientRequest): Promise<ClientResponse<T>> {
    const url = `${this.config.protocol}://${this.config.host}${request.endpoint}`;
    const params = new URLSearchParams();
    
    // Add API key to params
    params.append('apikey', this.polygonConfig.apiKey);
    
    // Add request params
    if (request.params) {
      Object.entries(request.params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          params.append(key, String(value));
        }
      });
    }

    const urlWithParams = `${url}?${params.toString()}`;

    try {
      const response = await fetch(urlWithParams, {
        method: request.method,
        headers: request.headers as Record<string, string>
      });

      if (!response.ok) {
        // Handle rate limiting
        if (response.status === 429) {
          const retryAfter = response.headers.get('Retry-After');
          throw {
            response: { status: 429, data: { retryAfter } },
            message: 'Rate limit exceeded'
          };
        }
        
        throw {
          response: { status: response.status },
          message: `HTTP ${response.status}: ${response.statusText}`
        };
      }

      const data = await response.json();

      // Check for API-level errors
      if (data.status === 'ERROR') {
        throw new Error(data.error || 'API error');
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

  private getTierRateLimit(tier: string): number {
    const limits = {
      'basic': 5,
      'starter': 100,
      'developer': 1000,
      'advanced': 100000
    };
    return limits[tier] || 5;
  }

  private async connectWebSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      const wsUrl = this.polygonConfig.wsHost || 'wss://socket.polygon.io/stocks';
      this.ws = new WebSocket(wsUrl);

      this.ws.on('open', () => {
        this.authenticate();
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

      // Listen for auth confirmation
      this.once('ws:authenticated', () => {
        this.wsConnected = true;
        this.emit('ws:connected');
        resolve();
      });

      this.once('ws:auth_failed', (error) => {
        reject(error);
      });

      setTimeout(() => {
        if (!this.wsConnected) {
          reject(new Error('WebSocket authentication timeout'));
        }
      }, this.config.timeout);
    });
  }

  private authenticate(): void {
    if (this.ws) {
      this.ws.send(JSON.stringify({
        action: 'auth',
        params: this.polygonConfig.apiKey
      }));
    }
  }

  private handleWebSocketMessage(data: WebSocket.Data): void {
    try {
      const messages = JSON.parse(data.toString());
      
      if (!Array.isArray(messages)) {
        return;
      }

      for (const message of messages) {
        switch (message.ev) {
          case 'status':
            this.handleStatusMessage(message);
            break;
          case 'T':
            this.emit('trade', this.convertTrade(message));
            break;
          case 'Q':
            this.emit('quote', this.convertQuote(message));
            break;
          case 'A':
          case 'AM':
            this.emit('aggregate', this.convertAggregate(message));
            break;
          default:
            this.emit('ws:message', message);
        }
      }
    } catch (error) {
      this.emit('ws:parse_error', error);
    }
  }

  private handleStatusMessage(message: any): void {
    switch (message.status) {
      case 'connected':
        this.emit('ws:status', 'connected');
        break;
      case 'auth_success':
        this.emit('ws:authenticated');
        break;
      case 'auth_failed':
        this.emit('ws:auth_failed', new Error(message.message));
        break;
      case 'success':
        this.emit('ws:subscription_success', message);
        break;
      case 'error':
        this.emit('ws:error', new Error(message.message));
        break;
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;

    this.reconnectTimer = setTimeout(async () => {
      this.reconnectTimer = null;
      try {
        await this.connectWebSocket();
        // Resubscribe to previous subscriptions
        for (const subscription of this.subscriptions.values()) {
          await this.resubscribe(subscription);
        }
      } catch (error) {
        this.emit('reconnect:failed', error);
        this.scheduleReconnect();
      }
    }, 5000);
  }

  private async resubscribe(subscription: PolygonSubscription): Promise<void> {
    if (this.ws && this.wsConnected) {
      this.ws.send(JSON.stringify({
        action: 'subscribe',
        params: `${subscription.channel}.${subscription.symbols.join(',')}`
      }));
    }
  }

  // Public API Methods

  async getTickers(params?: {
    ticker?: string;
    type?: string;
    market?: string;
    exchange?: string;
    cusip?: string;
    cik?: string;
    date?: string;
    search?: string;
    active?: boolean;
    gte?: string;
    gt?: string;
    lte?: string;
    lt?: string;
    limit?: number;
  }): Promise<PolygonTicker[]> {
    const response = await this.request<{results: PolygonTicker[]}>('GET', '/v3/reference/tickers', params);
    return response.success && response.data?.results ? response.data.results : [];
  }

  async getAggregates(
    ticker: string,
    multiplier: number,
    timespan: 'minute' | 'hour' | 'day' | 'week' | 'month' | 'quarter' | 'year',
    from: string,
    to: string,
    params?: {
      adjusted?: boolean;
      sort?: 'asc' | 'desc';
      limit?: number;
    }
  ): Promise<CandleData[]> {
    const endpoint = `/v2/aggs/ticker/${ticker}/range/${multiplier}/${timespan}/${from}/${to}`;
    const response = await this.request<{results: PolygonAgg[]}>('GET', endpoint, params);
    
    if (response.success && response.data?.results) {
      return response.data.results.map(this.convertAggregateToCandle);
    }
    
    return [];
  }

  async getGroupedDaily(
    date: string,
    params?: {
      adjusted?: boolean;
      include_otc?: boolean;
    }
  ): Promise<PolygonAgg[]> {
    const response = await this.request<{results: PolygonAgg[]}>('GET', `/v2/aggs/grouped/locale/us/market/stocks/${date}`, params);
    return response.success && response.data?.results ? response.data.results : [];
  }

  async getDailyOpenClose(
    ticker: string,
    date: string,
    params?: {
      adjusted?: boolean;
    }
  ): Promise<{
    status: string;
    from: string;
    symbol: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    afterHours: number;
    preMarket: number;
  } | null> {
    const response = await this.request('GET', `/v1/open-close/${ticker}/${date}`, params);
    return response.success ? response.data : null;
  }

  async getPreviousClose(ticker: string, params?: { adjusted?: boolean }): Promise<PolygonAgg[]> {
    const response = await this.request<{results: PolygonAgg[]}>('GET', `/v2/aggs/ticker/${ticker}/prev`, params);
    return response.success && response.data?.results ? response.data.results : [];
  }

  async getLastTrade(ticker: string): Promise<PolygonTrade | null> {
    const response = await this.request<{results: any}>('GET', `/v2/last/trade/${ticker}`);
    
    if (response.success && response.data?.results) {
      return this.convertTrade(response.data.results);
    }
    
    return null;
  }

  async getLastQuote(ticker: string): Promise<PolygonQuote | null> {
    const response = await this.request<{results: any}>('GET', `/v2/last/nbbo/${ticker}`);
    
    if (response.success && response.data?.results) {
      return this.convertQuote(response.data.results);
    }
    
    return null;
  }

  async getMarketStatus(): Promise<PolygonMarketStatus | null> {
    const response = await this.request<PolygonMarketStatus>('GET', '/v1/marketstatus/now');
    return response.success ? response.data : null;
  }

  // WebSocket Subscriptions

  async subscribeToTrades(symbols: string[]): Promise<void> {
    if (!this.wsConnected) {
      throw new Error('WebSocket not connected');
    }

    const subscription: PolygonSubscription = {
      channel: 'T',
      symbols
    };

    this.subscriptions.set('trades', subscription);

    this.ws!.send(JSON.stringify({
      action: 'subscribe',
      params: `T.${symbols.join(',T.')}`
    }));
  }

  async subscribeToQuotes(symbols: string[]): Promise<void> {
    if (!this.wsConnected) {
      throw new Error('WebSocket not connected');
    }

    const subscription: PolygonSubscription = {
      channel: 'Q',
      symbols
    };

    this.subscriptions.set('quotes', subscription);

    this.ws!.send(JSON.stringify({
      action: 'subscribe',
      params: `Q.${symbols.join(',Q.')}`
    }));
  }

  async subscribeToAggregates(symbols: string[]): Promise<void> {
    if (!this.wsConnected) {
      throw new Error('WebSocket not connected');
    }

    const subscription: PolygonSubscription = {
      channel: 'A',
      symbols
    };

    this.subscriptions.set('aggregates', subscription);

    this.ws!.send(JSON.stringify({
      action: 'subscribe',
      params: `A.${symbols.join(',A.')}`
    }));
  }

  async unsubscribeFromTrades(symbols: string[]): Promise<void> {
    if (!this.wsConnected) return;

    this.ws!.send(JSON.stringify({
      action: 'unsubscribe',
      params: `T.${symbols.join(',T.')}`
    }));
  }

  async unsubscribeFromQuotes(symbols: string[]): Promise<void> {
    if (!this.wsConnected) return;

    this.ws!.send(JSON.stringify({
      action: 'unsubscribe',
      params: `Q.${symbols.join(',Q.')}`
    }));
  }

  async unsubscribeFromAggregates(symbols: string[]): Promise<void> {
    if (!this.wsConnected) return;

    this.ws!.send(JSON.stringify({
      action: 'unsubscribe',
      params: `A.${symbols.join(',A.')}`
    }));
  }

  // Conversion Methods

  private convertAggregateToCandle(agg: PolygonAgg): CandleData {
    return {
      timestamp: agg.t,
      open: agg.o,
      high: agg.h,
      low: agg.l,
      close: agg.c,
      volume: agg.v
    };
  }

  private convertTrade(trade: any): PolygonTrade {
    return {
      ticker: trade.sym || trade.T,
      price: trade.p,
      size: trade.s,
      timestamp: trade.t,
      exchange: trade.x,
      conditions: trade.c || []
    };
  }

  private convertQuote(quote: any): PolygonQuote {
    return {
      ticker: quote.sym || quote.T,
      bid: quote.p || quote.bp,
      ask: quote.P || quote.ap,
      bidSize: quote.s || quote.bs,
      askSize: quote.S || quote.as,
      timestamp: quote.t,
      exchange: quote.x
    };
  }

  private convertAggregate(agg: any): CandleData {
    return {
      timestamp: agg.s, // start timestamp
      open: agg.o,
      high: agg.h,
      low: agg.l,
      close: agg.c,
      volume: agg.v
    };
  }

  // Utility Methods

  getSubscriptions(): Map<string, PolygonSubscription> {
    return new Map(this.subscriptions);
  }

  isWebSocketConnected(): boolean {
    return this.wsConnected;
  }

  getTier(): string {
    return this.polygonConfig.tier;
  }

  getRateLimit(): number {
    return this.getTierRateLimit(this.polygonConfig.tier);
  }

  async testConnection(): Promise<{ api: boolean; websocket: boolean }> {
    let apiOk = false;
    let wsOk = false;

    try {
      const response = await this.request('GET', '/v1/marketstatus/now');
      apiOk = response.success;
    } catch {}

    wsOk = this.wsConnected;

    return { api: apiOk, websocket: wsOk };
  }
}