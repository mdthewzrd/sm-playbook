/**
 * NotionClient - Interface with Notion API for strategy documentation and management
 * 
 * This client provides comprehensive integration with Notion for:
 * - Strategy documentation and retrieval
 * - Creating and updating strategy pages
 * - Search functionality for finding strategies
 * - Backtest result storage and analysis
 */

import {
  BaseMCPClient,
  MCPServerConfig,
  MCPHealthCheck,
  MCPRequest,
  MCPResponse,
  NotionPage,
  NotionStrategy,
  BacktestResult,
  MCPError,
  MCPValidationError
} from '../types';

export interface NotionClientConfig extends MCPServerConfig {
  databaseId?: string;
  strategiesTableId?: string;
  backtestResultsTableId?: string;
}

export interface CreateStrategyPageRequest {
  title: string;
  description?: string;
  parameters?: Record<string, any>;
  status?: 'draft' | 'testing' | 'live' | 'archived';
  tags?: string[];
}

export interface UpdateStrategyPageRequest {
  pageId: string;
  title?: string;
  description?: string;
  parameters?: Record<string, any>;
  status?: 'draft' | 'testing' | 'live' | 'archived';
  tags?: string[];
  backtestResults?: BacktestResult[];
}

export interface SearchStrategiesRequest {
  query?: string;
  status?: 'draft' | 'testing' | 'live' | 'archived';
  tags?: string[];
  limit?: number;
  sortBy?: 'created' | 'modified' | 'name';
  sortOrder?: 'asc' | 'desc';
}

export interface NotionDatabaseQuery {
  database_id: string;
  filter?: any;
  sorts?: any[];
  start_cursor?: string;
  page_size?: number;
}

export class NotionClient extends BaseMCPClient {
  private apiToken: string;
  private clientConfig: NotionClientConfig;

  constructor(config: NotionClientConfig) {
    super('notion', config);
    this.clientConfig = config;
    this.apiToken = config.env?.NOTION_API_TOKEN || '';
    
    if (!this.apiToken) {
      throw new MCPValidationError('notion', 'NOTION_API_TOKEN is required');
    }
  }

  async connect(): Promise<void> {
    try {
      // Test connection by getting user info
      const response = await this.request<void, any>({
        method: 'users/me'
      });

      if (response.success) {
        this.connectionStatus = {
          connected: true,
          lastHeartbeat: new Date(),
          retryCount: 0
        };
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
    this.connectionStatus = {
      connected: false,
      retryCount: 0
    };
  }

  async healthCheck(): Promise<MCPHealthCheck> {
    const startTime = Date.now();
    
    try {
      const response = await this.request<void, any>({
        method: 'users/me',
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
      // This would be implemented to communicate with the actual MCP server
      // For now, we'll simulate the request structure
      const result = await this.makeNotionApiCall(request.method, request.params);
      
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
   * Create a new strategy page in Notion
   */
  async createStrategyPage(request: CreateStrategyPageRequest): Promise<NotionStrategy> {
    const response = await this.request<CreateStrategyPageRequest, NotionStrategy>({
      method: 'pages/create',
      params: request
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to create strategy page: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Update an existing strategy page
   */
  async updateStrategyPage(request: UpdateStrategyPageRequest): Promise<NotionStrategy> {
    const response = await this.request<UpdateStrategyPageRequest, NotionStrategy>({
      method: 'pages/update',
      params: request
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to update strategy page: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Get a strategy page by ID
   */
  async getStrategyPage(pageId: string): Promise<NotionStrategy> {
    const response = await this.request<{ pageId: string }, NotionStrategy>({
      method: 'pages/get',
      params: { pageId }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get strategy page: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Search for strategies based on criteria
   */
  async searchStrategies(request: SearchStrategiesRequest): Promise<NotionStrategy[]> {
    const response = await this.request<SearchStrategiesRequest, NotionStrategy[]>({
      method: 'strategies/search',
      params: request
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to search strategies: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Get all strategies with optional filtering
   */
  async getAllStrategies(status?: string, limit?: number): Promise<NotionStrategy[]> {
    return this.searchStrategies({
      status: status as any,
      limit,
      sortBy: 'modified',
      sortOrder: 'desc'
    });
  }

  /**
   * Add backtest results to a strategy page
   */
  async addBacktestResults(pageId: string, results: BacktestResult[]): Promise<NotionStrategy> {
    const response = await this.request<{ pageId: string; results: BacktestResult[] }, NotionStrategy>({
      method: 'strategies/add-backtest-results',
      params: { pageId, results }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to add backtest results: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Get backtest results for a strategy
   */
  async getBacktestResults(pageId: string): Promise<BacktestResult[]> {
    const response = await this.request<{ pageId: string }, BacktestResult[]>({
      method: 'strategies/get-backtest-results',
      params: { pageId }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get backtest results: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Archive a strategy (set status to archived)
   */
  async archiveStrategy(pageId: string): Promise<NotionStrategy> {
    return this.updateStrategyPage({
      pageId,
      status: 'archived'
    });
  }

  /**
   * Duplicate a strategy page
   */
  async duplicateStrategy(pageId: string, newTitle: string): Promise<NotionStrategy> {
    const originalStrategy = await this.getStrategyPage(pageId);
    
    return this.createStrategyPage({
      title: newTitle,
      description: originalStrategy.properties.description,
      parameters: originalStrategy.parameters,
      status: 'draft',
      tags: originalStrategy.properties.tags
    });
  }

  /**
   * Get strategy performance summary across all strategies
   */
  async getStrategyPerformanceSummary(): Promise<{
    totalStrategies: number;
    activeStrategies: number;
    averageReturn: number;
    bestPerformingStrategy: NotionStrategy | null;
    worstPerformingStrategy: NotionStrategy | null;
  }> {
    const strategies = await this.getAllStrategies();
    const activeStrategies = strategies.filter(s => s.status === 'live');
    
    let totalReturn = 0;
    let bestStrategy: NotionStrategy | null = null;
    let worstStrategy: NotionStrategy | null = null;
    let strategiesWithResults = 0;

    for (const strategy of strategies) {
      if (strategy.backtestResults && strategy.backtestResults.length > 0) {
        const latestResult = strategy.backtestResults[strategy.backtestResults.length - 1];
        totalReturn += latestResult.totalReturn;
        strategiesWithResults++;

        if (!bestStrategy || latestResult.totalReturn > (bestStrategy.backtestResults?.[0]?.totalReturn || 0)) {
          bestStrategy = strategy;
        }

        if (!worstStrategy || latestResult.totalReturn < (worstStrategy.backtestResults?.[0]?.totalReturn || 0)) {
          worstStrategy = strategy;
        }
      }
    }

    return {
      totalStrategies: strategies.length,
      activeStrategies: activeStrategies.length,
      averageReturn: strategiesWithResults > 0 ? totalReturn / strategiesWithResults : 0,
      bestPerformingStrategy: bestStrategy,
      worstPerformingStrategy: worstStrategy
    };
  }

  /**
   * Export strategies to different formats
   */
  async exportStrategies(format: 'json' | 'csv' | 'markdown'): Promise<string> {
    const strategies = await this.getAllStrategies();
    
    switch (format) {
      case 'json':
        return JSON.stringify(strategies, null, 2);
      
      case 'csv':
        return this.strategiesToCSV(strategies);
      
      case 'markdown':
        return this.strategiesToMarkdown(strategies);
      
      default:
        throw new MCPValidationError(this.serverId, `Unsupported export format: ${format}`);
    }
  }

  private async makeNotionApiCall(method: string, params?: any): Promise<any> {
    // This would be the actual implementation that communicates with the MCP server
    // For now, we'll return mock data based on the method
    
    switch (method) {
      case 'users/me':
        return { id: 'user-123', name: 'Trading System User' };
      
      case 'pages/create':
        return this.mockCreateStrategyPage(params);
      
      case 'pages/update':
        return this.mockUpdateStrategyPage(params);
      
      case 'pages/get':
        return this.mockGetStrategyPage(params.pageId);
      
      case 'strategies/search':
        return this.mockSearchStrategies(params);
      
      default:
        throw new MCPError(`Unknown method: ${method}`, this.serverId);
    }
  }

  private mockCreateStrategyPage(params: CreateStrategyPageRequest): NotionStrategy {
    return {
      id: `strategy-${Date.now()}`,
      title: params.title,
      url: `https://notion.so/strategy-${Date.now()}`,
      properties: {
        description: params.description || '',
        status: params.status || 'draft',
        tags: params.tags || []
      },
      parameters: params.parameters,
      backtestResults: [],
      status: params.status || 'draft',
      lastModified: new Date()
    };
  }

  private mockUpdateStrategyPage(params: UpdateStrategyPageRequest): NotionStrategy {
    return {
      id: params.pageId,
      title: params.title || 'Updated Strategy',
      url: `https://notion.so/${params.pageId}`,
      properties: {
        description: params.description || '',
        status: params.status || 'draft',
        tags: params.tags || []
      },
      parameters: params.parameters,
      backtestResults: params.backtestResults || [],
      status: params.status || 'draft',
      lastModified: new Date()
    };
  }

  private mockGetStrategyPage(pageId: string): NotionStrategy {
    return {
      id: pageId,
      title: 'Mock Strategy',
      url: `https://notion.so/${pageId}`,
      properties: {
        description: 'A mock strategy for testing',
        status: 'testing',
        tags: ['mock', 'test']
      },
      parameters: {
        rsi_period: 14,
        ma_period: 20
      },
      backtestResults: [],
      status: 'testing',
      lastModified: new Date()
    };
  }

  private mockSearchStrategies(params: SearchStrategiesRequest): NotionStrategy[] {
    return [
      {
        id: 'strategy-1',
        title: 'RSI Mean Reversion',
        url: 'https://notion.so/strategy-1',
        properties: {
          description: 'RSI-based mean reversion strategy',
          status: 'live',
          tags: ['rsi', 'mean-reversion']
        },
        parameters: { rsi_period: 14, threshold: 30 },
        backtestResults: [],
        status: 'live',
        lastModified: new Date()
      },
      {
        id: 'strategy-2',
        title: 'EMA Crossover',
        url: 'https://notion.so/strategy-2',
        properties: {
          description: 'EMA crossover momentum strategy',
          status: 'testing',
          tags: ['ema', 'momentum']
        },
        parameters: { fast_ema: 9, slow_ema: 21 },
        backtestResults: [],
        status: 'testing',
        lastModified: new Date()
      }
    ];
  }

  private strategiesToCSV(strategies: NotionStrategy[]): string {
    const headers = ['ID', 'Title', 'Status', 'Description', 'Last Modified'];
    const rows = strategies.map(s => [
      s.id,
      s.title,
      s.status,
      s.properties.description || '',
      s.lastModified.toISOString()
    ]);

    return [headers, ...rows]
      .map(row => row.map(cell => `"${cell}"`).join(','))
      .join('\n');
  }

  private strategiesToMarkdown(strategies: NotionStrategy[]): string {
    let markdown = '# Trading Strategies\n\n';
    
    for (const strategy of strategies) {
      markdown += `## ${strategy.title}\n\n`;
      markdown += `- **Status**: ${strategy.status}\n`;
      markdown += `- **Description**: ${strategy.properties.description || 'No description'}\n`;
      markdown += `- **Last Modified**: ${strategy.lastModified.toISOString()}\n`;
      
      if (strategy.parameters) {
        markdown += `- **Parameters**: ${JSON.stringify(strategy.parameters, null, 2)}\n`;
      }
      
      markdown += '\n---\n\n';
    }

    return markdown;
  }
}