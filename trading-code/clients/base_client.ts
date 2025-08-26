import { EventEmitter } from 'events';

export interface ClientConfig {
  name: string;
  host: string;
  port?: number;
  protocol: 'http' | 'https' | 'ws' | 'wss';
  timeout: number;
  retryAttempts: number;
  retryDelay: number;
  apiKey?: string;
  apiSecret?: string;
  headers?: Record<string, string>;
  rateLimiting?: RateLimitConfig;
}

export interface RateLimitConfig {
  enabled: boolean;
  requestsPerSecond: number;
  burstSize: number;
}

export interface ClientRequest {
  id: string;
  method: string;
  endpoint: string;
  params?: any;
  headers?: Record<string, string>;
  timeout?: number;
  timestamp: number;
}

export interface ClientResponse<T = any> {
  id: string;
  success: boolean;
  data?: T;
  error?: ClientError;
  timestamp: number;
  duration: number;
  metadata?: Record<string, any>;
}

export interface ClientError {
  code: string;
  message: string;
  details?: any;
  retryable: boolean;
}

export interface ConnectionStats {
  connected: boolean;
  connectionTime: number;
  lastActivity: number;
  requestCount: number;
  errorCount: number;
  averageResponseTime: number;
  rateLimitStatus: {
    remaining: number;
    resetTime: number;
    limited: boolean;
  };
}

export abstract class BaseClient extends EventEmitter {
  protected config: ClientConfig;
  protected connected: boolean = false;
  protected stats: ConnectionStats;
  private requestQueue: Map<string, ClientRequest> = new Map();
  private retryQueue: ClientRequest[] = [];
  private rateLimiter: RateLimiter;

  constructor(config: ClientConfig) {
    super();
    this.config = { ...config };
    this.rateLimiter = new RateLimiter(config.rateLimiting);
    this.stats = this.initializeStats();
    this.setupEventHandlers();
  }

  abstract connect(): Promise<void>;
  abstract disconnect(): Promise<void>;
  abstract isHealthy(): Promise<boolean>;
  protected abstract executeRequest<T>(request: ClientRequest): Promise<ClientResponse<T>>;

  async request<T = any>(
    method: string,
    endpoint: string,
    params?: any,
    options?: {
      timeout?: number;
      headers?: Record<string, string>;
      retryOnFailure?: boolean;
    }
  ): Promise<ClientResponse<T>> {
    const requestId = `${this.config.name}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const request: ClientRequest = {
      id: requestId,
      method,
      endpoint,
      params,
      headers: { ...this.config.headers, ...options?.headers },
      timeout: options?.timeout || this.config.timeout,
      timestamp: Date.now()
    };

    try {
      if (!this.connected) {
        await this.connect();
      }

      await this.rateLimiter.waitForRateLimit();
      
      this.requestQueue.set(requestId, request);
      this.emit('request:start', request);
      
      const startTime = Date.now();
      const response = await this.executeRequest<T>(request);
      const duration = Date.now() - startTime;
      
      response.duration = duration;
      this.updateStats(true, duration);
      
      this.requestQueue.delete(requestId);
      this.emit('request:success', { request, response });
      
      return response;
      
    } catch (error) {
      const errorResponse: ClientResponse<T> = {
        id: requestId,
        success: false,
        error: this.normalizeError(error),
        timestamp: Date.now(),
        duration: Date.now() - request.timestamp
      };

      this.updateStats(false, errorResponse.duration);
      this.requestQueue.delete(requestId);
      
      if (options?.retryOnFailure !== false && this.shouldRetry(errorResponse.error!)) {
        return this.retryRequest<T>(request, options);
      }
      
      this.emit('request:error', { request, response: errorResponse });
      return errorResponse;
    }
  }

  private async retryRequest<T>(
    originalRequest: ClientRequest,
    options?: any
  ): Promise<ClientResponse<T>> {
    const retryAttempts = this.config.retryAttempts;
    let lastResponse: ClientResponse<T> | null = null;

    for (let attempt = 1; attempt <= retryAttempts; attempt++) {
      await this.delay(this.config.retryDelay * attempt);
      
      try {
        const retryRequest: ClientRequest = {
          ...originalRequest,
          id: `${originalRequest.id}_retry_${attempt}`,
          timestamp: Date.now()
        };

        await this.rateLimiter.waitForRateLimit();
        
        const startTime = Date.now();
        const response = await this.executeRequest<T>(retryRequest);
        response.duration = Date.now() - startTime;
        
        this.updateStats(true, response.duration);
        this.emit('request:retry:success', { attempt, request: retryRequest, response });
        
        return response;
        
      } catch (error) {
        lastResponse = {
          id: `${originalRequest.id}_retry_${attempt}`,
          success: false,
          error: this.normalizeError(error),
          timestamp: Date.now(),
          duration: Date.now() - originalRequest.timestamp
        };

        this.updateStats(false, lastResponse.duration);
        
        if (attempt === retryAttempts) {
          this.emit('request:retry:failed', { 
            attempts: retryAttempts, 
            request: originalRequest, 
            lastResponse 
          });
        }
      }
    }

    return lastResponse!;
  }

  private shouldRetry(error: ClientError): boolean {
    if (!error.retryable) return false;
    
    const retryableCodes = ['TIMEOUT', 'NETWORK_ERROR', 'RATE_LIMITED', 'INTERNAL_ERROR'];
    return retryableCodes.includes(error.code);
  }

  private normalizeError(error: any): ClientError {
    if (error.code && error.message) {
      return error as ClientError;
    }

    let code = 'UNKNOWN_ERROR';
    let retryable = false;

    if (error.name === 'TimeoutError') {
      code = 'TIMEOUT';
      retryable = true;
    } else if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
      code = 'NETWORK_ERROR';
      retryable = true;
    } else if (error.response?.status === 429) {
      code = 'RATE_LIMITED';
      retryable = true;
    } else if (error.response?.status >= 500) {
      code = 'INTERNAL_ERROR';
      retryable = true;
    }

    return {
      code,
      message: error.message || 'Unknown error occurred',
      details: error.response?.data || error.stack,
      retryable
    };
  }

  private updateStats(success: boolean, duration: number): void {
    this.stats.requestCount++;
    this.stats.lastActivity = Date.now();
    
    if (!success) {
      this.stats.errorCount++;
    }

    const totalRequests = this.stats.requestCount;
    const currentAvg = this.stats.averageResponseTime;
    this.stats.averageResponseTime = (currentAvg * (totalRequests - 1) + duration) / totalRequests;

    this.rateLimiter.updateStats();
  }

  private initializeStats(): ConnectionStats {
    return {
      connected: false,
      connectionTime: 0,
      lastActivity: 0,
      requestCount: 0,
      errorCount: 0,
      averageResponseTime: 0,
      rateLimitStatus: {
        remaining: this.config.rateLimiting?.requestsPerSecond || 100,
        resetTime: Date.now() + 1000,
        limited: false
      }
    };
  }

  private setupEventHandlers(): void {
    this.on('connected', () => {
      this.connected = true;
      this.stats.connected = true;
      this.stats.connectionTime = Date.now();
    });

    this.on('disconnected', () => {
      this.connected = false;
      this.stats.connected = false;
    });

    this.on('error', (error) => {
      console.error(`${this.config.name} client error:`, error);
    });
  }

  private async delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  getStats(): ConnectionStats {
    return { 
      ...this.stats, 
      rateLimitStatus: this.rateLimiter.getStatus() 
    };
  }

  getConfig(): ClientConfig {
    return { ...this.config };
  }

  updateConfig(updates: Partial<ClientConfig>): void {
    this.config = { ...this.config, ...updates };
    if (updates.rateLimiting) {
      this.rateLimiter = new RateLimiter(updates.rateLimiting);
    }
  }

  async ping(): Promise<number> {
    const startTime = Date.now();
    try {
      await this.isHealthy();
      return Date.now() - startTime;
    } catch (error) {
      throw error;
    }
  }

  getActiveRequests(): ClientRequest[] {
    return Array.from(this.requestQueue.values());
  }

  cancelRequest(requestId: string): boolean {
    return this.requestQueue.delete(requestId);
  }

  clearStats(): void {
    this.stats = this.initializeStats();
  }
}

export class RateLimiter {
  private config?: RateLimitConfig;
  private tokens: number;
  private lastRefill: number;
  private queue: Array<{ resolve: () => void; timestamp: number }> = [];

  constructor(config?: RateLimitConfig) {
    this.config = config;
    this.tokens = config?.burstSize || 100;
    this.lastRefill = Date.now();
  }

  async waitForRateLimit(): Promise<void> {
    if (!this.config?.enabled) return;

    return new Promise((resolve) => {
      this.refillTokens();
      
      if (this.tokens > 0) {
        this.tokens--;
        resolve();
      } else {
        this.queue.push({ resolve, timestamp: Date.now() });
        this.processQueue();
      }
    });
  }

  private refillTokens(): void {
    if (!this.config) return;

    const now = Date.now();
    const timePassed = now - this.lastRefill;
    const tokensToAdd = Math.floor(timePassed / 1000 * this.config.requestsPerSecond);
    
    if (tokensToAdd > 0) {
      this.tokens = Math.min(this.config.burstSize, this.tokens + tokensToAdd);
      this.lastRefill = now;
    }
  }

  private processQueue(): void {
    setInterval(() => {
      this.refillTokens();
      
      while (this.tokens > 0 && this.queue.length > 0) {
        const request = this.queue.shift();
        if (request) {
          this.tokens--;
          request.resolve();
        }
      }
    }, 100);
  }

  getStatus(): ConnectionStats['rateLimitStatus'] {
    if (!this.config?.enabled) {
      return {
        remaining: Infinity,
        resetTime: 0,
        limited: false
      };
    }

    this.refillTokens();
    
    return {
      remaining: this.tokens,
      resetTime: this.lastRefill + 1000,
      limited: this.tokens === 0
    };
  }

  updateStats(): void {
    this.refillTokens();
  }
}