/**
 * MCP Manager - Core system for managing Model Context Protocol server connections
 * 
 * This module provides centralized management for all MCP server connections,
 * including health monitoring, connection pooling, and error recovery.
 */

import { EventEmitter } from 'events';
import {
  BaseMCPClient,
  MCPServerConfig,
  MCPHealthCheck,
  MCPConnectionStatus,
  MCPError,
  MCPConnectionError
} from '../types';

export interface MCPManagerConfig {
  healthCheckInterval: number;
  maxRetryAttempts: number;
  retryBackoff: number;
  connectionTimeout: number;
  enableMetrics: boolean;
}

export interface MCPMetrics {
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  averageResponseTime: number;
  uptime: number;
  lastHealthCheck: Date;
}

export class MCPManager extends EventEmitter {
  private clients: Map<string, BaseMCPClient> = new Map();
  private healthCheckTimer?: NodeJS.Timeout;
  private metrics: Map<string, MCPMetrics> = new Map();
  private config: MCPManagerConfig;

  constructor(config: Partial<MCPManagerConfig> = {}) {
    super();
    
    this.config = {
      healthCheckInterval: config.healthCheckInterval ?? 30000, // 30 seconds
      maxRetryAttempts: config.maxRetryAttempts ?? 3,
      retryBackoff: config.retryBackoff ?? 1000,
      connectionTimeout: config.connectionTimeout ?? 10000,
      enableMetrics: config.enableMetrics ?? true,
      ...config
    };

    this.startHealthMonitoring();
  }

  /**
   * Register a new MCP client with the manager
   */
  async registerClient(client: BaseMCPClient): Promise<void> {
    const serverId = client.getServerId();
    
    if (this.clients.has(serverId)) {
      throw new MCPError(`Client ${serverId} already registered`, serverId);
    }

    this.clients.set(serverId, client);
    this.initializeMetrics(serverId);

    try {
      await this.connectClient(serverId);
      this.emit('clientRegistered', serverId);
    } catch (error) {
      this.emit('clientError', serverId, error);
      throw error;
    }
  }

  /**
   * Unregister and disconnect a client
   */
  async unregisterClient(serverId: string): Promise<void> {
    const client = this.clients.get(serverId);
    if (!client) {
      throw new MCPError(`Client ${serverId} not found`, serverId);
    }

    try {
      await client.disconnect();
    } catch (error) {
      console.warn(`Error disconnecting client ${serverId}:`, error);
    }

    this.clients.delete(serverId);
    this.metrics.delete(serverId);
    this.emit('clientUnregistered', serverId);
  }

  /**
   * Get a registered client by ID
   */
  getClient<T extends BaseMCPClient>(serverId: string): T | undefined {
    return this.clients.get(serverId) as T;
  }

  /**
   * Get all registered client IDs
   */
  getRegisteredClients(): string[] {
    return Array.from(this.clients.keys());
  }

  /**
   * Get connection status for all clients
   */
  getConnectionStatuses(): Record<string, MCPConnectionStatus> {
    const statuses: Record<string, MCPConnectionStatus> = {};
    
    for (const [serverId, client] of this.clients) {
      statuses[serverId] = client.getStatus();
    }
    
    return statuses;
  }

  /**
   * Get metrics for a specific client
   */
  getMetrics(serverId: string): MCPMetrics | undefined {
    return this.metrics.get(serverId);
  }

  /**
   * Get metrics for all clients
   */
  getAllMetrics(): Record<string, MCPMetrics> {
    const allMetrics: Record<string, MCPMetrics> = {};
    
    for (const [serverId, metrics] of this.metrics) {
      allMetrics[serverId] = { ...metrics };
    }
    
    return allMetrics;
  }

  /**
   * Manually trigger health check for all clients
   */
  async performHealthCheck(): Promise<Record<string, MCPHealthCheck>> {
    const healthChecks: Record<string, MCPHealthCheck> = {};
    
    for (const [serverId, client] of this.clients) {
      try {
        const healthCheck = await client.healthCheck();
        healthChecks[serverId] = healthCheck;
        
        if (this.config.enableMetrics) {
          const metrics = this.metrics.get(serverId);
          if (metrics) {
            metrics.lastHealthCheck = new Date();
            if (healthCheck.responseTime) {
              this.updateResponseTime(serverId, healthCheck.responseTime);
            }
          }
        }
        
        if (healthCheck.status === 'unhealthy') {
          this.emit('clientUnhealthy', serverId, healthCheck);
        }
      } catch (error) {
        const errorHealthCheck: MCPHealthCheck = {
          serverId,
          status: 'unhealthy',
          lastCheck: new Date(),
          error: error instanceof Error ? error.message : String(error)
        };
        
        healthChecks[serverId] = errorHealthCheck;
        this.emit('clientError', serverId, error);
      }
    }
    
    return healthChecks;
  }

  /**
   * Attempt to reconnect a specific client
   */
  async reconnectClient(serverId: string): Promise<void> {
    const client = this.clients.get(serverId);
    if (!client) {
      throw new MCPError(`Client ${serverId} not found`, serverId);
    }

    await this.connectClientWithRetry(serverId);
  }

  /**
   * Reconnect all clients
   */
  async reconnectAllClients(): Promise<void> {
    const reconnectPromises = Array.from(this.clients.keys()).map(serverId =>
      this.reconnectClient(serverId).catch(error => {
        console.error(`Failed to reconnect client ${serverId}:`, error);
        return error;
      })
    );

    await Promise.allSettled(reconnectPromises);
  }

  /**
   * Shutdown the MCP manager and all clients
   */
  async shutdown(): Promise<void> {
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
    }

    const disconnectPromises = Array.from(this.clients.entries()).map(
      async ([serverId, client]) => {
        try {
          await client.disconnect();
        } catch (error) {
          console.warn(`Error disconnecting client ${serverId}:`, error);
        }
      }
    );

    await Promise.allSettled(disconnectPromises);
    this.clients.clear();
    this.metrics.clear();
    this.emit('shutdown');
  }

  /**
   * Update request metrics for a client
   */
  updateRequestMetrics(serverId: string, success: boolean, responseTime: number): void {
    if (!this.config.enableMetrics) return;

    const metrics = this.metrics.get(serverId);
    if (!metrics) return;

    metrics.totalRequests++;
    if (success) {
      metrics.successfulRequests++;
    } else {
      metrics.failedRequests++;
    }

    this.updateResponseTime(serverId, responseTime);
  }

  private initializeMetrics(serverId: string): void {
    if (!this.config.enableMetrics) return;

    this.metrics.set(serverId, {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      averageResponseTime: 0,
      uptime: 0,
      lastHealthCheck: new Date()
    });
  }

  private updateResponseTime(serverId: string, responseTime: number): void {
    const metrics = this.metrics.get(serverId);
    if (!metrics) return;

    // Calculate moving average
    const totalResponseTime = metrics.averageResponseTime * (metrics.totalRequests - 1);
    metrics.averageResponseTime = (totalResponseTime + responseTime) / metrics.totalRequests;
  }

  private async connectClient(serverId: string): Promise<void> {
    const client = this.clients.get(serverId);
    if (!client) {
      throw new MCPError(`Client ${serverId} not found`, serverId);
    }

    await this.connectClientWithRetry(serverId);
  }

  private async connectClientWithRetry(serverId: string): Promise<void> {
    const client = this.clients.get(serverId);
    if (!client) {
      throw new MCPError(`Client ${serverId} not found`, serverId);
    }

    let lastError: Error | undefined;
    
    for (let attempt = 1; attempt <= this.config.maxRetryAttempts; attempt++) {
      try {
        await Promise.race([
          client.connect(),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Connection timeout')), this.config.connectionTimeout)
          )
        ]);
        
        this.emit('clientConnected', serverId);
        return;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        
        if (attempt < this.config.maxRetryAttempts) {
          const backoff = this.config.retryBackoff * attempt;
          await new Promise(resolve => setTimeout(resolve, backoff));
        }
      }
    }

    throw new MCPConnectionError(serverId, lastError);
  }

  private startHealthMonitoring(): void {
    this.healthCheckTimer = setInterval(async () => {
      try {
        await this.performHealthCheck();
      } catch (error) {
        console.error('Health check failed:', error);
        this.emit('healthCheckError', error);
      }
    }, this.config.healthCheckInterval);
  }
}