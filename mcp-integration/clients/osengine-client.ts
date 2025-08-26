/**
 * OsEngineClient - Interface with OsEngine for trade execution and management
 * 
 * This client provides integration with OsEngine trading platform for:
 * - Order execution and management
 * - Position tracking and portfolio management
 * - Integration with exchange APIs
 * - Risk management and monitoring
 */

import {
  BaseMCPClient,
  MCPServerConfig,
  MCPHealthCheck,
  MCPRequest,
  MCPResponse,
  Order,
  Position,
  Trade,
  RiskManagement,
  MCPError,
  MCPValidationError
} from '../types';

export interface OsEngineClientConfig extends MCPServerConfig {
  tradingMode: 'paper' | 'live';
  defaultExchange: string;
  riskLimits: RiskManagement;
  enableAutoExecution: boolean;
}

export interface OrderRequest {
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop' | 'stop_limit';
  quantity: number;
  price?: number;
  stopPrice?: number;
  timeInForce?: 'GTC' | 'IOC' | 'FOK' | 'DAY';
  clientOrderId?: string;
  reduceOnly?: boolean;
}

export interface PortfolioStatus {
  totalEquity: number;
  availableCash: number;
  usedMargin: number;
  freeMargin: number;
  unrealizedPnL: number;
  realizedPnL: number;
  positions: Position[];
  openOrders: Order[];
  dailyPnL: number;
  totalFees: number;
}

export interface RiskMetrics {
  portfolioRisk: number;
  maxDrawdown: number;
  var: number; // Value at Risk
  sharpeRatio: number;
  positionSizes: Record<string, number>;
  correlations: Record<string, Record<string, number>>;
  leverage: number;
  marginUtilization: number;
}

export interface ExecutionReport {
  orderId: string;
  clientOrderId?: string;
  symbol: string;
  side: 'buy' | 'sell';
  executedQuantity: number;
  executedPrice: number;
  commission: number;
  timestamp: Date;
  executionId: string;
  lastPx: number;
  lastQty: number;
}

export interface TradeSignal {
  symbol: string;
  action: 'buy' | 'sell' | 'close';
  quantity: number;
  price?: number;
  stopLoss?: number;
  takeProfit?: number;
  strategy: string;
  confidence: number;
  timestamp: Date;
}

export interface AutoTradeConfig {
  enabled: boolean;
  strategies: string[];
  maxPositions: number;
  maxRiskPerTrade: number;
  stopLossPercent: number;
  takeProfitPercent: number;
  minConfidence: number;
}

export class OsEngineClient extends BaseMCPClient {
  private clientConfig: OsEngineClientConfig;
  private activeOrders: Map<string, Order> = new Map();
  private positions: Map<string, Position> = new Map();
  private executionCallbacks: Array<(report: ExecutionReport) => void> = [];

  constructor(config: OsEngineClientConfig) {
    super('osengine', config);
    this.clientConfig = {
      tradingMode: 'paper',
      defaultExchange: 'binance',
      riskLimits: {
        stopLoss: 0.02,
        takeProfit: 0.06,
        positionSize: 0.05,
        maxDrawdown: 0.15,
        maxPositions: 10
      },
      enableAutoExecution: false,
      ...config
    };
  }

  async connect(): Promise<void> {
    try {
      const response = await this.request<void, { status: string; mode: string }>({
        method: 'system/status'
      });

      if (response.success && response.data?.status === 'connected') {
        this.connectionStatus = {
          connected: true,
          lastHeartbeat: new Date(),
          retryCount: 0
        };
        
        // Initialize portfolio and positions
        await this.syncPortfolio();
      } else {
        throw new MCPError(`Connection failed: ${response.error}`, this.serverId);
      }
    } catch (error) {
      this.connectionStatus = {
        connected: false,
        error: error instanceof Error ? error.message : String(error),
        retryCount: this.connectionStatus.retryCount + 1
      };
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    // Cancel all pending orders before disconnecting
    await this.cancelAllOrders();
    
    this.activeOrders.clear();
    this.positions.clear();
    this.executionCallbacks = [];
    
    this.connectionStatus = {
      connected: false,
      retryCount: 0
    };
  }

  async healthCheck(): Promise<MCPHealthCheck> {
    const startTime = Date.now();
    
    try {
      const response = await this.request<void, any>({
        method: 'system/ping',
        timeout: 5000
      });

      const responseTime = Date.now() - startTime;

      return {
        serverId: this.serverId,
        status: response.success ? 'healthy' : 'unhealthy',
        responseTime,
        lastCheck: new Date(),
        error: response.error
      };
    } catch (error) {
      return {
        serverId: this.serverId,
        status: 'unhealthy',
        responseTime: Date.now() - startTime,
        lastCheck: new Date(),
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  async request<T, R>(request: MCPRequest<T>): Promise<MCPResponse<R>> {
    const startTime = Date.now();
    
    try {
      const result = await this.makeOsEngineApiCall(request.method, request.params);
      
      return {
        success: true,
        data: result as R,
        timestamp: new Date(),
        responseTime: Date.now() - startTime
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        timestamp: new Date(),
        responseTime: Date.now() - startTime
      };
    }
  }

  /**
   * Place a new order
   */
  async placeOrder(orderRequest: OrderRequest): Promise<Order> {
    this.validateOrderRequest(orderRequest);
    
    // Check risk limits before placing order
    await this.validateRiskLimits(orderRequest);

    const response = await this.request<OrderRequest, Order>({
      method: 'orders/place',
      params: orderRequest
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to place order: ${response.error}`, this.serverId);
    }

    // Store order locally
    this.activeOrders.set(response.data.id, response.data);

    return response.data;
  }

  /**
   * Cancel an existing order
   */
  async cancelOrder(orderId: string): Promise<boolean> {
    const response = await this.request<{ orderId: string }, boolean>({
      method: 'orders/cancel',
      params: { orderId }
    });

    if (response.success && response.data) {
      this.activeOrders.delete(orderId);
      return true;
    }

    return false;
  }

  /**
   * Cancel all open orders
   */
  async cancelAllOrders(symbol?: string): Promise<number> {
    const response = await this.request<{ symbol?: string }, { canceledCount: number }>({
      method: 'orders/cancel-all',
      params: { symbol }
    });

    if (response.success && response.data) {
      if (symbol) {
        // Remove orders for specific symbol
        for (const [orderId, order] of this.activeOrders.entries()) {
          if (order.symbol === symbol) {
            this.activeOrders.delete(orderId);
          }
        }
      } else {
        // Clear all orders
        this.activeOrders.clear();
      }
      
      return response.data.canceledCount;
    }

    throw new MCPError(`Failed to cancel orders: ${response.error}`, this.serverId);
  }

  /**
   * Get order status
   */
  async getOrderStatus(orderId: string): Promise<Order> {
    const response = await this.request<{ orderId: string }, Order>({
      method: 'orders/status',
      params: { orderId }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get order status: ${response.error}`, this.serverId);
    }

    // Update local order cache
    this.activeOrders.set(orderId, response.data);

    return response.data;
  }

  /**
   * Get all open orders
   */
  async getOpenOrders(symbol?: string): Promise<Order[]> {
    const response = await this.request<{ symbol?: string }, Order[]>({
      method: 'orders/open',
      params: { symbol }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get open orders: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Get current positions
   */
  async getPositions(symbol?: string): Promise<Position[]> {
    const response = await this.request<{ symbol?: string }, Position[]>({
      method: 'positions/get',
      params: { symbol }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get positions: ${response.error}`, this.serverId);
    }

    // Update local position cache
    for (const position of response.data) {
      this.positions.set(position.symbol, position);
    }

    return response.data;
  }

  /**
   * Close position
   */
  async closePosition(symbol: string, quantity?: number): Promise<Order> {
    const position = this.positions.get(symbol);
    if (!position) {
      throw new MCPError(`No position found for symbol: ${symbol}`, this.serverId);
    }

    const closeQuantity = quantity || Math.abs(position.quantity);
    const closeSide = position.side === 'long' ? 'sell' : 'buy';

    const orderRequest: OrderRequest = {
      symbol,
      side: closeSide,
      type: 'market',
      quantity: closeQuantity,
      reduceOnly: true
    };

    return this.placeOrder(orderRequest);
  }

  /**
   * Close all positions
   */
  async closeAllPositions(): Promise<Order[]> {
    const positions = await this.getPositions();
    const closeOrders: Order[] = [];

    for (const position of positions) {
      if (position.quantity !== 0) {
        try {
          const closeOrder = await this.closePosition(position.symbol);
          closeOrders.push(closeOrder);
        } catch (error) {
          console.warn(`Failed to close position ${position.symbol}:`, error);
        }
      }
    }

    return closeOrders;
  }

  /**
   * Get portfolio status
   */
  async getPortfolioStatus(): Promise<PortfolioStatus> {
    const response = await this.request<void, PortfolioStatus>({
      method: 'portfolio/status'
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get portfolio status: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Calculate risk metrics
   */
  async calculateRiskMetrics(): Promise<RiskMetrics> {
    const response = await this.request<void, RiskMetrics>({
      method: 'risk/metrics'
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to calculate risk metrics: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Execute trade signal
   */
  async executeTradeSignal(signal: TradeSignal): Promise<Order | null> {
    if (!this.clientConfig.enableAutoExecution) {
      throw new MCPError('Auto execution is disabled', this.serverId);
    }

    this.validateTradeSignal(signal);

    // Check if signal meets minimum confidence threshold
    const autoConfig = await this.getAutoTradeConfig();
    if (signal.confidence < autoConfig.minConfidence) {
      return null;
    }

    let orderRequest: OrderRequest;

    if (signal.action === 'close') {
      const position = this.positions.get(signal.symbol);
      if (position && position.quantity !== 0) {
        return this.closePosition(signal.symbol, signal.quantity);
      }
      return null;
    } else {
      orderRequest = {
        symbol: signal.symbol,
        side: signal.action,
        type: signal.price ? 'limit' : 'market',
        quantity: signal.quantity,
        price: signal.price,
        clientOrderId: `signal_${signal.strategy}_${Date.now()}`
      };
    }

    const order = await this.placeOrder(orderRequest);

    // Set stop loss and take profit if specified
    if (signal.stopLoss || signal.takeProfit) {
      await this.setOrderProtection(order.id, signal.stopLoss, signal.takeProfit);
    }

    return order;
  }

  /**
   * Set stop loss and take profit for an order
   */
  async setOrderProtection(
    orderId: string,
    stopLoss?: number,
    takeProfit?: number
  ): Promise<void> {
    if (stopLoss || takeProfit) {
      await this.request<{
        orderId: string;
        stopLoss?: number;
        takeProfit?: number;
      }, void>({
        method: 'orders/protection',
        params: { orderId, stopLoss, takeProfit }
      });
    }
  }

  /**
   * Get trading history
   */
  async getTradingHistory(
    symbol?: string,
    startDate?: Date,
    endDate?: Date,
    limit?: number
  ): Promise<Trade[]> {
    const response = await this.request<{
      symbol?: string;
      startDate?: Date;
      endDate?: Date;
      limit?: number;
    }, Trade[]>({
      method: 'trades/history',
      params: { symbol, startDate, endDate, limit }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get trading history: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Get auto-trade configuration
   */
  async getAutoTradeConfig(): Promise<AutoTradeConfig> {
    const response = await this.request<void, AutoTradeConfig>({
      method: 'config/auto-trade'
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get auto-trade config: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Update auto-trade configuration
   */
  async updateAutoTradeConfig(config: Partial<AutoTradeConfig>): Promise<void> {
    const response = await this.request<Partial<AutoTradeConfig>, void>({
      method: 'config/auto-trade/update',
      params: config
    });

    if (!response.success) {
      throw new MCPError(`Failed to update auto-trade config: ${response.error}`, this.serverId);
    }
  }

  /**
   * Register execution callback
   */
  onExecution(callback: (report: ExecutionReport) => void): void {
    this.executionCallbacks.push(callback);
  }

  /**
   * Get exchange info
   */
  async getExchangeInfo(exchange?: string): Promise<{
    name: string;
    status: string;
    tradingFees: Record<string, number>;
    symbols: Array<{
      symbol: string;
      baseAsset: string;
      quoteAsset: string;
      minOrderSize: number;
      maxOrderSize: number;
      pricePrecision: number;
      quantityPrecision: number;
    }>;
  }> {
    const response = await this.request<{ exchange?: string }, any>({
      method: 'exchange/info',
      params: { exchange: exchange || this.clientConfig.defaultExchange }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get exchange info: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  private async syncPortfolio(): Promise<void> {
    try {
      const [positions, orders] = await Promise.all([
        this.getPositions(),
        this.getOpenOrders()
      ]);

      this.positions.clear();
      this.activeOrders.clear();

      for (const position of positions) {
        this.positions.set(position.symbol, position);
      }

      for (const order of orders) {
        this.activeOrders.set(order.id, order);
      }
    } catch (error) {
      console.warn('Failed to sync portfolio:', error);
    }
  }

  private validateOrderRequest(request: OrderRequest): void {
    if (!request.symbol) {
      throw new MCPValidationError(this.serverId, 'Symbol is required');
    }

    if (!['buy', 'sell'].includes(request.side)) {
      throw new MCPValidationError(this.serverId, 'Side must be buy or sell');
    }

    if (!['market', 'limit', 'stop', 'stop_limit'].includes(request.type)) {
      throw new MCPValidationError(this.serverId, 'Invalid order type');
    }

    if (request.quantity <= 0) {
      throw new MCPValidationError(this.serverId, 'Quantity must be positive');
    }

    if (request.type === 'limit' && !request.price) {
      throw new MCPValidationError(this.serverId, 'Price is required for limit orders');
    }

    if (request.type.includes('stop') && !request.stopPrice) {
      throw new MCPValidationError(this.serverId, 'Stop price is required for stop orders');
    }
  }

  private async validateRiskLimits(request: OrderRequest): Promise<void> {
    const riskMetrics = await this.calculateRiskMetrics();
    const limits = this.clientConfig.riskLimits;

    // Check portfolio risk
    if (riskMetrics.portfolioRisk > (limits.maxDrawdown || 0.15)) {
      throw new MCPError('Portfolio risk exceeds maximum allowed', this.serverId);
    }

    // Check position count
    if (this.positions.size >= (limits.maxPositions || 10)) {
      throw new MCPError('Maximum number of positions reached', this.serverId);
    }

    // Check margin utilization
    if (riskMetrics.marginUtilization > 0.8) {
      throw new MCPError('Margin utilization too high', this.serverId);
    }
  }

  private validateTradeSignal(signal: TradeSignal): void {
    if (!signal.symbol) {
      throw new MCPValidationError(this.serverId, 'Signal symbol is required');
    }

    if (!['buy', 'sell', 'close'].includes(signal.action)) {
      throw new MCPValidationError(this.serverId, 'Invalid signal action');
    }

    if (signal.quantity <= 0) {
      throw new MCPValidationError(this.serverId, 'Signal quantity must be positive');
    }

    if (signal.confidence < 0 || signal.confidence > 1) {
      throw new MCPValidationError(this.serverId, 'Signal confidence must be between 0 and 1');
    }
  }

  private async makeOsEngineApiCall(method: string, params?: any): Promise<any> {
    // Mock implementation - in reality this would communicate with the MCP server
    switch (method) {
      case 'system/status':
        return { status: 'connected', mode: this.clientConfig.tradingMode };

      case 'system/ping':
        return 'pong';

      case 'orders/place':
        return this.mockPlaceOrder(params);

      case 'orders/cancel':
        return true;

      case 'orders/cancel-all':
        return { canceledCount: this.activeOrders.size };

      case 'orders/status':
        return this.mockOrderStatus(params.orderId);

      case 'orders/open':
        return Array.from(this.activeOrders.values());

      case 'positions/get':
        return Array.from(this.positions.values());

      case 'portfolio/status':
        return this.mockPortfolioStatus();

      case 'risk/metrics':
        return this.mockRiskMetrics();

      case 'trades/history':
        return [];

      case 'config/auto-trade':
        return this.mockAutoTradeConfig();

      case 'exchange/info':
        return this.mockExchangeInfo();

      default:
        throw new MCPError(`Unknown method: ${method}`, this.serverId);
    }
  }

  private mockPlaceOrder(params: OrderRequest): Order {
    return {
      id: `order_${Date.now()}`,
      clientOrderId: params.clientOrderId,
      symbol: params.symbol,
      side: params.side,
      type: params.type,
      quantity: params.quantity,
      price: params.price,
      stopPrice: params.stopPrice,
      timeInForce: params.timeInForce || 'GTC',
      status: 'new',
      executedQuantity: 0,
      timestamp: new Date()
    };
  }

  private mockOrderStatus(orderId: string): Order {
    return {
      id: orderId,
      symbol: 'BTCUSDT',
      side: 'buy',
      type: 'limit',
      quantity: 0.1,
      price: 45000,
      status: 'filled',
      executedQuantity: 0.1,
      averagePrice: 45000,
      timestamp: new Date()
    };
  }

  private mockPortfolioStatus(): PortfolioStatus {
    return {
      totalEquity: 100000,
      availableCash: 50000,
      usedMargin: 25000,
      freeMargin: 25000,
      unrealizedPnL: 1500,
      realizedPnL: 5000,
      positions: [],
      openOrders: [],
      dailyPnL: 500,
      totalFees: 150
    };
  }

  private mockRiskMetrics(): RiskMetrics {
    return {
      portfolioRisk: 0.05,
      maxDrawdown: 0.03,
      var: 2500,
      sharpeRatio: 1.5,
      positionSizes: {},
      correlations: {},
      leverage: 2.0,
      marginUtilization: 0.5
    };
  }

  private mockAutoTradeConfig(): AutoTradeConfig {
    return {
      enabled: false,
      strategies: [],
      maxPositions: 5,
      maxRiskPerTrade: 0.02,
      stopLossPercent: 0.02,
      takeProfitPercent: 0.04,
      minConfidence: 0.7
    };
  }

  private mockExchangeInfo(): any {
    return {
      name: this.clientConfig.defaultExchange,
      status: 'active',
      tradingFees: { maker: 0.001, taker: 0.001 },
      symbols: [
        {
          symbol: 'BTCUSDT',
          baseAsset: 'BTC',
          quoteAsset: 'USDT',
          minOrderSize: 0.001,
          maxOrderSize: 1000,
          pricePrecision: 2,
          quantityPrecision: 6
        }
      ]
    };
  }
}