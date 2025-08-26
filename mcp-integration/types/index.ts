/**
 * Core types and interfaces for MCP integration with SM Playbook BMad Trading System
 * 
 * This module defines the base types, interfaces, and contracts for integrating
 * Model Context Protocol (MCP) servers with the BMad trading framework.
 */

export interface MCPServerConfig {
  name: string;
  command: string;
  args: string[];
  env?: Record<string, string>;
  timeout?: number;
  retryAttempts?: number;
  healthCheckInterval?: number;
}

export interface MCPConnectionStatus {
  connected: boolean;
  lastHeartbeat?: Date;
  error?: string;
  retryCount: number;
}

export interface MCPHealthCheck {
  serverId: string;
  status: 'healthy' | 'unhealthy' | 'unknown';
  responseTime?: number;
  lastCheck: Date;
  error?: string;
}

export interface MCPRequest<T = any> {
  method: string;
  params?: T;
  timeout?: number;
  retries?: number;
}

export interface MCPResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: Date;
  responseTime: number;
}

// Base client interface that all MCP clients must implement
export abstract class BaseMCPClient {
  protected serverId: string;
  protected config: MCPServerConfig;
  protected connectionStatus: MCPConnectionStatus;

  constructor(serverId: string, config: MCPServerConfig) {
    this.serverId = serverId;
    this.config = config;
    this.connectionStatus = {
      connected: false,
      retryCount: 0
    };
  }

  abstract connect(): Promise<void>;
  abstract disconnect(): Promise<void>;
  abstract healthCheck(): Promise<MCPHealthCheck>;
  abstract request<T, R>(request: MCPRequest<T>): Promise<MCPResponse<R>>;

  getStatus(): MCPConnectionStatus {
    return { ...this.connectionStatus };
  }

  getServerId(): string {
    return this.serverId;
  }
}

// Market data types
export interface OHLCV {
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface MarketDataRequest {
  symbol: string;
  timeframe: string;
  from?: Date;
  to?: Date;
  limit?: number;
}

export interface TechnicalIndicatorRequest {
  data: number[];
  period?: number;
  parameters?: Record<string, any>;
}

export interface TechnicalIndicatorResult {
  values: number[];
  metadata?: Record<string, any>;
}

// Strategy and backtesting types
export interface Strategy {
  id: string;
  name: string;
  description?: string;
  parameters: Record<string, any>;
  indicators: string[];
  entryRules: string[];
  exitRules: string[];
  riskManagement: RiskManagement;
}

export interface RiskManagement {
  stopLoss?: number;
  takeProfit?: number;
  positionSize: number;
  maxDrawdown?: number;
  maxPositions?: number;
}

export interface BacktestRequest {
  strategy: Strategy;
  symbol: string;
  startDate: Date;
  endDate: Date;
  initialCapital: number;
  commission?: number;
}

export interface BacktestResult {
  totalReturn: number;
  annualizedReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalTrades: number;
  equity: OHLCV[];
  trades: Trade[];
  metrics: Record<string, number>;
}

export interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  entryPrice: number;
  exitPrice?: number;
  entryTime: Date;
  exitTime?: Date;
  pnl?: number;
  status: 'open' | 'closed';
}

// Order management types
export interface Order {
  id: string;
  clientOrderId?: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop' | 'stop_limit';
  quantity: number;
  price?: number;
  stopPrice?: number;
  timeInForce?: 'GTC' | 'IOC' | 'FOK' | 'DAY';
  status: 'new' | 'partially_filled' | 'filled' | 'canceled' | 'rejected';
  executedQuantity?: number;
  averagePrice?: number;
  timestamp: Date;
}

export interface Position {
  symbol: string;
  quantity: number;
  averagePrice: number;
  marketValue: number;
  unrealizedPnl: number;
  side: 'long' | 'short';
  timestamp: Date;
}

// Notion integration types
export interface NotionPage {
  id: string;
  title: string;
  url: string;
  properties: Record<string, any>;
  content?: string;
  lastModified: Date;
}

export interface NotionStrategy extends NotionPage {
  parameters?: Record<string, any>;
  backtestResults?: BacktestResult[];
  status: 'draft' | 'testing' | 'live' | 'archived';
}

// BMad agent integration types
export interface AgentContext {
  agentId: string;
  sessionId: string;
  parameters: Record<string, any>;
  mcpClients: Record<string, BaseMCPClient>;
}

export interface AgentRequest<T = any> {
  action: string;
  parameters?: T;
  context: AgentContext;
}

export interface AgentResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  recommendations?: string[];
  nextActions?: string[];
}

// Error types
export class MCPError extends Error {
  constructor(
    message: string,
    public serverId: string,
    public code?: string,
    public details?: any
  ) {
    super(message);
    this.name = 'MCPError';
  }
}

export class MCPConnectionError extends MCPError {
  constructor(serverId: string, details?: any) {
    super(`Failed to connect to MCP server: ${serverId}`, serverId, 'CONNECTION_ERROR', details);
    this.name = 'MCPConnectionError';
  }
}

export class MCPTimeoutError extends MCPError {
  constructor(serverId: string, timeout: number) {
    super(`MCP request timed out after ${timeout}ms`, serverId, 'TIMEOUT_ERROR');
    this.name = 'MCPTimeoutError';
  }
}

export class MCPValidationError extends MCPError {
  constructor(serverId: string, message: string, details?: any) {
    super(message, serverId, 'VALIDATION_ERROR', details);
    this.name = 'MCPValidationError';
  }
}