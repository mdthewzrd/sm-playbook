/**
 * Trade Controller
 * 
 * Manages order execution, position tracking, and trade lifecycle
 * Interfaces with brokers and manages order routing and execution
 */

import { Order, OrderType, OrderFill, Position, PositionSide, PositionStatus } from '../models/position';
import { TradingSignal, SignalType } from '../models/strategy';
import { RiskLimits } from '../models/portfolio';

export interface TradeExecutionConfig {
  brokerId: string;
  accountId: string;
  defaultSlippage: number;
  maxSlippage: number;
  executionTimeout: number;
  retryAttempts: number;
  enablePaperTrading: boolean;
}

export interface ExecutionReport {
  orderId: string;
  status: 'pending' | 'partial' | 'filled' | 'cancelled' | 'rejected';
  executedQuantity: number;
  averagePrice: number;
  commission: number;
  timestamp: Date;
  executionId: string;
  message?: string;
}

export interface OrderRequest {
  symbol: string;
  side: PositionSide;
  type: OrderType;
  quantity: number;
  price?: number;
  stopPrice?: number;
  trailingAmount?: number;
  timeInForce: 'GTC' | 'IOC' | 'FOK' | 'DAY';
  clientOrderId?: string;
  strategyId: string;
  signalId?: string;
  riskLimits?: Partial<RiskLimits>;
}

export interface TradeMetrics {
  totalOrders: number;
  filledOrders: number;
  cancelledOrders: number;
  rejectedOrders: number;
  fillRate: number;
  averageSlippage: number;
  averageFillTime: number;
  totalCommission: number;
  totalVolume: number;
}

export interface PositionUpdate {
  positionId: string;
  currentPrice: number;
  marketValue: number;
  unrealizedPnL: number;
  timestamp: Date;
  source: string;
}

export class TradeController {
  private orders: Map<string, Order> = new Map();
  private positions: Map<string, Position> = new Map();
  private executionHistory: ExecutionReport[] = [];
  private config: TradeExecutionConfig;
  private nextOrderId: number = 1;

  constructor(config: TradeExecutionConfig) {
    this.config = config;
  }

  /**
   * Execute a trading signal by placing appropriate orders
   */
  async executeSignal(signal: TradingSignal): Promise<Order[]> {
    const orders: Order[] = [];

    try {
      switch (signal.type) {
        case SignalType.ENTRY_LONG:
          orders.push(await this.createEntryOrder(signal, PositionSide.LONG));
          break;
          
        case SignalType.ENTRY_SHORT:
          orders.push(await this.createEntryOrder(signal, PositionSide.SHORT));
          break;
          
        case SignalType.EXIT_LONG:
          orders.push(...await this.createExitOrders(signal, PositionSide.LONG));
          break;
          
        case SignalType.EXIT_SHORT:
          orders.push(...await this.createExitOrders(signal, PositionSide.SHORT));
          break;
          
        default:
          throw new Error(`Unsupported signal type: ${signal.type}`);
      }

      // Place orders with broker
      for (const order of orders) {
        await this.placeOrder(order);
      }

      return orders;
    } catch (error) {
      console.error(`Error executing signal ${signal.id}:`, error);
      throw error;
    }
  }

  /**
   * Place a new order
   */
  async placeOrder(order: Order): Promise<boolean> {
    try {
      // Validate order before placement
      const validation = this.validateOrder(order);
      if (!validation.isValid) {
        throw new Error(`Order validation failed: ${validation.errors.join(', ')}`);
      }

      // Store order
      this.orders.set(order.id, order);

      // Execute order based on configuration
      if (this.config.enablePaperTrading) {
        await this.executePaperOrder(order);
      } else {
        await this.executeLiveOrder(order);
      }

      console.log(`Order ${order.id} placed successfully`);
      return true;
    } catch (error) {
      console.error(`Error placing order ${order.id}:`, error);
      
      // Update order status to rejected
      order.status = 'rejected';
      order.reason = error instanceof Error ? error.message : String(error);
      order.lastUpdate = new Date();
      
      return false;
    }
  }

  /**
   * Cancel an existing order
   */
  async cancelOrder(orderId: string, reason?: string): Promise<boolean> {
    const order = this.orders.get(orderId);
    if (!order) {
      throw new Error(`Order ${orderId} not found`);
    }

    if (order.status === 'filled' || order.status === 'cancelled') {
      return false; // Cannot cancel already completed orders
    }

    try {
      // Cancel with broker
      if (!this.config.enablePaperTrading) {
        await this.cancelLiveOrder(orderId);
      }

      // Update order status
      order.status = 'cancelled';
      order.reason = reason || 'Manual cancellation';
      order.lastUpdate = new Date();

      console.log(`Order ${orderId} cancelled`);
      return true;
    } catch (error) {
      console.error(`Error cancelling order ${orderId}:`, error);
      return false;
    }
  }

  /**
   * Modify an existing order
   */
  async modifyOrder(
    orderId: string,
    modifications: Partial<Pick<Order, 'quantity' | 'price' | 'stopPrice'>>
  ): Promise<boolean> {
    const order = this.orders.get(orderId);
    if (!order) {
      throw new Error(`Order ${orderId} not found`);
    }

    if (order.status === 'filled' || order.status === 'cancelled') {
      return false; // Cannot modify completed orders
    }

    try {
      // Apply modifications
      if (modifications.quantity !== undefined) {
        order.quantity = modifications.quantity;
        order.remainingQuantity = modifications.quantity - order.filledQuantity;
      }
      
      if (modifications.price !== undefined) {
        order.price = modifications.price;
      }
      
      if (modifications.stopPrice !== undefined) {
        order.stopPrice = modifications.stopPrice;
      }

      order.lastUpdate = new Date();

      // Submit modification to broker
      if (!this.config.enablePaperTrading) {
        await this.modifyLiveOrder(orderId, modifications);
      }

      console.log(`Order ${orderId} modified successfully`);
      return true;
    } catch (error) {
      console.error(`Error modifying order ${orderId}:`, error);
      return false;
    }
  }

  /**
   * Get order status
   */
  getOrder(orderId: string): Order | undefined {
    return this.orders.get(orderId);
  }

  /**
   * Get all orders for a symbol
   */
  getOrdersBySymbol(symbol: string): Order[] {
    return Array.from(this.orders.values()).filter(order => order.symbol === symbol);
  }

  /**
   * Get all open orders
   */
  getOpenOrders(): Order[] {
    return Array.from(this.orders.values()).filter(
      order => order.status === 'pending' || order.status === 'partial'
    );
  }

  /**
   * Get all positions
   */
  getAllPositions(): Position[] {
    return Array.from(this.positions.values());
  }

  /**
   * Get open positions
   */
  getOpenPositions(): Position[] {
    return Array.from(this.positions.values()).filter(
      position => position.status === PositionStatus.OPEN || position.status === PositionStatus.PARTIAL
    );
  }

  /**
   * Get position for symbol
   */
  getPosition(symbol: string): Position | undefined {
    return Array.from(this.positions.values()).find(position => position.symbol === symbol);
  }

  /**
   * Update position with new market data
   */
  updatePosition(update: PositionUpdate): void {
    const position = this.positions.get(update.positionId);
    if (!position) {
      console.warn(`Position ${update.positionId} not found for update`);
      return;
    }

    // Update position metrics
    position.currentPrice = update.currentPrice;
    position.marketValue = update.marketValue;
    position.lastUpdate = update.timestamp;

    // Recalculate metrics
    this.recalculatePositionMetrics(position);

    console.log(`Position ${update.positionId} updated`);
  }

  /**
   * Close position by creating exit orders
   */
  async closePosition(positionId: string, reason?: string): Promise<Order[]> {
    const position = this.positions.get(positionId);
    if (!position) {
      throw new Error(`Position ${positionId} not found`);
    }

    if (position.status === PositionStatus.CLOSED) {
      return []; // Position already closed
    }

    const exitSide = position.side === PositionSide.LONG ? PositionSide.SHORT : PositionSide.LONG;
    const exitQuantity = position.filledQuantity;

    const exitOrder = this.createOrder({
      symbol: position.symbol,
      side: exitSide,
      type: OrderType.MARKET,
      quantity: exitQuantity,
      timeInForce: 'IOC',
      strategyId: position.strategyId,
      clientOrderId: `exit_${position.id}_${Date.now()}`
    });

    await this.placeOrder(exitOrder);
    return [exitOrder];
  }

  /**
   * Get trading metrics
   */
  getTradeMetrics(): TradeMetrics {
    const allOrders = Array.from(this.orders.values());
    const totalOrders = allOrders.length;
    const filledOrders = allOrders.filter(o => o.status === 'filled').length;
    const cancelledOrders = allOrders.filter(o => o.status === 'cancelled').length;
    const rejectedOrders = allOrders.filter(o => o.status === 'rejected').length;

    const totalVolume = allOrders.reduce((sum, order) => 
      sum + (order.filledQuantity * order.avgFillPrice), 0
    );
    
    const totalCommission = allOrders.reduce((sum, order) => sum + order.totalCommission, 0);
    
    // Calculate average slippage and fill time
    let totalSlippage = 0;
    let totalFillTime = 0;
    let ordersWithMetrics = 0;

    allOrders.filter(o => o.status === 'filled').forEach(order => {
      if (order.price && order.fills.length > 0) {
        const avgSlippage = Math.abs(order.avgFillPrice - order.price) / order.price;
        totalSlippage += avgSlippage;
        
        const fillTime = order.fills[order.fills.length - 1].timestamp.getTime() - order.timestamp.getTime();
        totalFillTime += fillTime;
        
        ordersWithMetrics++;
      }
    });

    return {
      totalOrders,
      filledOrders,
      cancelledOrders,
      rejectedOrders,
      fillRate: totalOrders > 0 ? filledOrders / totalOrders : 0,
      averageSlippage: ordersWithMetrics > 0 ? totalSlippage / ordersWithMetrics : 0,
      averageFillTime: ordersWithMetrics > 0 ? totalFillTime / ordersWithMetrics : 0,
      totalCommission,
      totalVolume
    };
  }

  /**
   * Process order fill notification
   */
  async processOrderFill(fill: OrderFill): Promise<void> {
    const order = this.orders.get(fill.orderId);
    if (!order) {
      console.warn(`Received fill for unknown order: ${fill.orderId}`);
      return;
    }

    // Add fill to order
    order.fills.push(fill);
    order.filledQuantity += fill.quantity;
    order.remainingQuantity = order.quantity - order.filledQuantity;
    order.totalCommission += fill.commission;
    order.lastUpdate = fill.timestamp;

    // Update average fill price
    const totalFillValue = order.fills.reduce((sum, f) => sum + (f.price * f.quantity), 0);
    const totalFillQuantity = order.fills.reduce((sum, f) => sum + f.quantity, 0);
    order.avgFillPrice = totalFillValue / totalFillQuantity;

    // Update order status
    if (order.remainingQuantity <= 0) {
      order.status = 'filled';
    } else if (order.filledQuantity > 0) {
      order.status = 'partial';
    }

    // Update or create position
    await this.updatePositionFromFill(order, fill);

    // Add to execution history
    const executionReport: ExecutionReport = {
      orderId: fill.orderId,
      status: order.status,
      executedQuantity: fill.quantity,
      averagePrice: fill.price,
      commission: fill.commission,
      timestamp: fill.timestamp,
      executionId: fill.executionId,
      message: `Fill processed: ${fill.quantity} @ ${fill.price}`
    };

    this.executionHistory.push(executionReport);

    console.log(`Order fill processed: ${fill.orderId} - ${fill.quantity} @ ${fill.price}`);
  }

  private async createEntryOrder(signal: TradingSignal, side: PositionSide): Promise<Order> {
    const orderType = signal.price ? OrderType.LIMIT : OrderType.MARKET;
    
    return this.createOrder({
      symbol: signal.symbol,
      side,
      type: orderType,
      quantity: signal.quantity || this.calculatePositionSize(signal),
      price: signal.price,
      timeInForce: 'GTC',
      strategyId: signal.reasoning[0] || 'unknown',
      signalId: signal.id,
      clientOrderId: `entry_${signal.id}`
    });
  }

  private async createExitOrders(signal: TradingSignal, side: PositionSide): Promise<Order[]> {
    const position = this.getPosition(signal.symbol);
    if (!position || position.side !== side) {
      return [];
    }

    const exitSide = side === PositionSide.LONG ? PositionSide.SHORT : PositionSide.LONG;
    const exitQuantity = signal.quantity || position.filledQuantity;

    const exitOrder = this.createOrder({
      symbol: signal.symbol,
      side: exitSide,
      type: OrderType.MARKET,
      quantity: exitQuantity,
      timeInForce: 'IOC',
      strategyId: signal.reasoning[0] || 'unknown',
      signalId: signal.id,
      clientOrderId: `exit_${signal.id}`
    });

    return [exitOrder];
  }

  private createOrder(request: OrderRequest): Order {
    const orderId = `ORD_${this.nextOrderId++}_${Date.now()}`;
    
    return {
      id: orderId,
      clientOrderId: request.clientOrderId || orderId,
      timestamp: new Date(),
      symbol: request.symbol,
      side: request.side,
      type: request.type,
      quantity: request.quantity,
      price: request.price,
      stopPrice: request.stopPrice,
      trailingAmount: request.trailingAmount,
      timeInForce: request.timeInForce,
      status: 'pending',
      filledQuantity: 0,
      remainingQuantity: request.quantity,
      avgFillPrice: 0,
      totalCommission: 0,
      fills: [],
      lastUpdate: new Date()
    };
  }

  private validateOrder(order: Order): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    if (!order.symbol) {
      errors.push('Symbol is required');
    }

    if (order.quantity <= 0) {
      errors.push('Quantity must be positive');
    }

    if (order.type === OrderType.LIMIT && !order.price) {
      errors.push('Price is required for limit orders');
    }

    if ((order.type === OrderType.STOP_MARKET || order.type === OrderType.STOP_LIMIT) && !order.stopPrice) {
      errors.push('Stop price is required for stop orders');
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  private async executePaperOrder(order: Order): Promise<void> {
    // Simulate paper trading execution
    await new Promise(resolve => setTimeout(resolve, 100)); // Simulate network delay

    // Create mock fill
    const fill: OrderFill = {
      id: `FILL_${Date.now()}`,
      orderId: order.id,
      timestamp: new Date(),
      price: order.price || this.getMockPrice(order.symbol),
      quantity: order.quantity,
      commission: order.quantity * 0.005, // $0.005 per share
      executionId: `EXEC_${Date.now()}`
    };

    await this.processOrderFill(fill);
  }

  private async executeLiveOrder(order: Order): Promise<void> {
    // This would interface with actual broker API
    // For now, throw error as live trading requires broker integration
    throw new Error('Live order execution not implemented - requires broker API integration');
  }

  private async cancelLiveOrder(orderId: string): Promise<void> {
    // This would interface with actual broker API
    throw new Error('Live order cancellation not implemented - requires broker API integration');
  }

  private async modifyLiveOrder(orderId: string, modifications: any): Promise<void> {
    // This would interface with actual broker API
    throw new Error('Live order modification not implemented - requires broker API integration');
  }

  private getMockPrice(symbol: string): number {
    // Return mock price for paper trading
    return 100 + Math.random() * 10;
  }

  private calculatePositionSize(signal: TradingSignal): number {
    // Calculate position size based on risk parameters
    // This is simplified - real implementation would consider portfolio size, risk limits, etc.
    return Math.floor(1000 / signal.price); // $1000 worth
  }

  private async updatePositionFromFill(order: Order, fill: OrderFill): Promise<void> {
    let position = this.getPosition(order.symbol);

    if (!position) {
      // Create new position
      position = {
        id: `POS_${order.symbol}_${Date.now()}`,
        symbol: order.symbol,
        side: order.side,
        status: PositionStatus.OPEN,
        quantity: 0,
        filledQuantity: 0,
        avgEntryPrice: 0,
        currentPrice: fill.price,
        marketValue: 0,
        openTime: fill.timestamp,
        lastUpdate: fill.timestamp,
        entryOrders: [],
        exitOrders: [],
        activeOrders: [],
        riskLimits: {
          maxPositionSize: 10000,
          maxLoss: 500
        },
        metrics: {
          unrealizedPnL: 0,
          realizedPnL: 0,
          totalPnL: 0,
          unrealizedPercent: 0,
          realizedPercent: 0,
          totalPercent: 0,
          holdingPeriod: 0,
          maxUnrealizedPnL: 0,
          maxDrawdown: 0,
          currentRisk: 0,
          riskRewardRatio: 0,
          commission: 0,
          netPnL: 0
        },
        strategyId: 'unknown',
        tags: [],
        exchange: 'MOCK',
        broker: this.config.brokerId
      };

      this.positions.set(position.id, position);
    }

    // Update position with fill
    const isEntry = (position.side === order.side);
    
    if (isEntry) {
      // Adding to position
      const totalValue = (position.filledQuantity * position.avgEntryPrice) + (fill.quantity * fill.price);
      position.filledQuantity += fill.quantity;
      position.avgEntryPrice = totalValue / position.filledQuantity;
      position.entryOrders.push(order);
    } else {
      // Reducing/closing position
      position.filledQuantity -= fill.quantity;
      position.exitOrders.push(order);
      
      if (position.filledQuantity <= 0) {
        position.status = PositionStatus.CLOSED;
        position.closeTime = fill.timestamp;
      }
    }

    position.currentPrice = fill.price;
    position.marketValue = position.filledQuantity * fill.price;
    position.lastUpdate = fill.timestamp;

    // Recalculate metrics
    this.recalculatePositionMetrics(position);
  }

  private recalculatePositionMetrics(position: Position): void {
    const currentValue = position.filledQuantity * position.currentPrice;
    const entryValue = position.filledQuantity * position.avgEntryPrice;
    
    if (position.side === PositionSide.LONG) {
      position.metrics.unrealizedPnL = currentValue - entryValue;
    } else {
      position.metrics.unrealizedPnL = entryValue - currentValue;
    }

    position.metrics.unrealizedPercent = entryValue !== 0 ? position.metrics.unrealizedPnL / entryValue : 0;
    position.metrics.totalPnL = position.metrics.unrealizedPnL + position.metrics.realizedPnL;
    position.metrics.totalPercent = position.metrics.unrealizedPercent;
    position.metrics.holdingPeriod = (Date.now() - position.openTime.getTime()) / (1000 * 60);

    // Update max metrics
    position.metrics.maxUnrealizedPnL = Math.max(
      position.metrics.maxUnrealizedPnL, 
      position.metrics.unrealizedPnL
    );
    
    position.metrics.maxDrawdown = Math.min(
      position.metrics.maxDrawdown,
      position.metrics.unrealizedPnL
    );

    // Calculate commission
    position.metrics.commission = [...position.entryOrders, ...position.exitOrders]
      .reduce((sum, order) => sum + order.totalCommission, 0);

    position.metrics.netPnL = position.metrics.totalPnL - position.metrics.commission;
  }
}