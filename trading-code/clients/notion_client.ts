import { BaseClient, ClientConfig, ClientRequest, ClientResponse } from './base_client';
import fetch from 'node-fetch';

export interface NotionConfig extends ClientConfig {
  apiKey: string;
  version: string;
}

export interface NotionPage {
  id: string;
  title: string;
  url: string;
  createdTime: string;
  lastEditedTime: string;
  properties: Record<string, any>;
  content?: NotionBlock[];
}

export interface NotionBlock {
  id: string;
  type: string;
  content: any;
  children?: NotionBlock[];
}

export interface NotionDatabase {
  id: string;
  title: string;
  description?: string;
  properties: Record<string, NotionProperty>;
  pages: NotionPage[];
}

export interface NotionProperty {
  id: string;
  name: string;
  type: string;
  config: any;
}

export interface TradingJournal {
  date: string;
  symbol: string;
  strategy: string;
  entryPrice: number;
  exitPrice?: number;
  quantity: number;
  pnl?: number;
  notes: string;
  tags: string[];
  status: 'open' | 'closed';
}

export interface StrategyAnalysis {
  strategyName: string;
  totalTrades: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  profitFactor: number;
  maxDrawdown: number;
  analysis: string;
  recommendations: string[];
}

export class NotionClient extends BaseClient {
  private notionConfig: NotionConfig;
  private journalDatabaseId?: string;
  private strategyDatabaseId?: string;

  constructor(config: NotionConfig) {
    const baseConfig: ClientConfig = {
      ...config,
      host: 'api.notion.com',
      protocol: 'https',
      timeout: config.timeout || 30000,
      retryAttempts: config.retryAttempts || 3,
      retryDelay: config.retryDelay || 1000,
      headers: {
        'Authorization': `Bearer ${config.apiKey}`,
        'Notion-Version': config.version || '2022-06-28',
        'Content-Type': 'application/json',
        ...config.headers
      }
    };

    super(baseConfig);
    this.notionConfig = config;
  }

  async connect(): Promise<void> {
    try {
      const response = await this.request('GET', '/v1/users/me');
      if (response.success) {
        this.emit('connected');
        await this.initializeDatabases();
      } else {
        throw new Error('Failed to authenticate with Notion');
      }
    } catch (error) {
      this.emit('error', error);
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    this.emit('disconnected');
  }

  async isHealthy(): Promise<boolean> {
    try {
      const response = await this.request('GET', '/v1/users/me');
      return response.success;
    } catch (error) {
      return false;
    }
  }

  protected async executeRequest<T>(request: ClientRequest): Promise<ClientResponse<T>> {
    const url = `${this.config.protocol}://${this.config.host}${request.endpoint}`;
    
    try {
      const fetchOptions: any = {
        method: request.method,
        headers: request.headers,
        timeout: request.timeout
      };

      if (request.params && ['POST', 'PATCH', 'PUT'].includes(request.method)) {
        fetchOptions.body = JSON.stringify(request.params);
      } else if (request.params && request.method === 'GET') {
        const params = new URLSearchParams(request.params);
        const urlWithParams = `${url}?${params.toString()}`;
      }

      const response = await fetch(url, fetchOptions);
      const data = await response.json();

      if (!response.ok) {
        throw {
          response: { status: response.status, data },
          message: data.message || `HTTP ${response.status}`
        };
      }

      return {
        id: request.id,
        success: true,
        data: data as T,
        timestamp: Date.now(),
        duration: 0 // Will be set by base class
      };

    } catch (error) {
      throw error;
    }
  }

  private async initializeDatabases(): Promise<void> {
    try {
      const databases = await this.listDatabases();
      
      const journalDb = databases.find(db => 
        db.title.toLowerCase().includes('trading') && 
        db.title.toLowerCase().includes('journal')
      );
      
      const strategyDb = databases.find(db => 
        db.title.toLowerCase().includes('strategy') && 
        db.title.toLowerCase().includes('analysis')
      );

      if (journalDb) {
        this.journalDatabaseId = journalDb.id;
      } else {
        this.journalDatabaseId = await this.createTradingJournalDatabase();
      }

      if (strategyDb) {
        this.strategyDatabaseId = strategyDb.id;
      } else {
        this.strategyDatabaseId = await this.createStrategyAnalysisDatabase();
      }

    } catch (error) {
      console.warn('Failed to initialize Notion databases:', error);
    }
  }

  async listDatabases(): Promise<NotionDatabase[]> {
    const response = await this.request<any>('POST', '/v1/search', {
      filter: { property: 'object', value: 'database' }
    });

    if (!response.success || !response.data?.results) {
      return [];
    }

    return response.data.results.map(db => ({
      id: db.id,
      title: db.title?.[0]?.plain_text || 'Untitled',
      description: db.description?.[0]?.plain_text,
      properties: db.properties || {},
      pages: []
    }));
  }

  async createTradingJournalDatabase(): Promise<string> {
    const databaseConfig = {
      parent: { type: 'page_id', page_id: await this.getWorkspaceId() },
      title: [{ type: 'text', text: { content: 'Trading Journal' } }],
      properties: {
        'Date': { type: 'date' },
        'Symbol': { type: 'title' },
        'Strategy': { type: 'select', select: { options: [] } },
        'Entry Price': { type: 'number', number: { format: 'dollar' } },
        'Exit Price': { type: 'number', number: { format: 'dollar' } },
        'Quantity': { type: 'number', number: {} },
        'P&L': { type: 'formula', formula: { expression: 'prop("Exit Price") * prop("Quantity") - prop("Entry Price") * prop("Quantity")' } },
        'Notes': { type: 'rich_text' },
        'Tags': { type: 'multi_select', multi_select: { options: [] } },
        'Status': { 
          type: 'select', 
          select: { 
            options: [
              { name: 'Open', color: 'yellow' },
              { name: 'Closed', color: 'green' }
            ] 
          } 
        }
      }
    };

    const response = await this.request<any>('POST', '/v1/databases', databaseConfig);
    
    if (!response.success) {
      throw new Error('Failed to create trading journal database');
    }

    return response.data.id;
  }

  async createStrategyAnalysisDatabase(): Promise<string> {
    const databaseConfig = {
      parent: { type: 'page_id', page_id: await this.getWorkspaceId() },
      title: [{ type: 'text', text: { content: 'Strategy Analysis' } }],
      properties: {
        'Strategy Name': { type: 'title' },
        'Total Trades': { type: 'number' },
        'Win Rate': { type: 'number', number: { format: 'percent' } },
        'Average Win': { type: 'number', number: { format: 'dollar' } },
        'Average Loss': { type: 'number', number: { format: 'dollar' } },
        'Profit Factor': { type: 'number' },
        'Max Drawdown': { type: 'number', number: { format: 'percent' } },
        'Last Updated': { type: 'last_edited_time' },
        'Status': { 
          type: 'select', 
          select: { 
            options: [
              { name: 'Active', color: 'green' },
              { name: 'Inactive', color: 'red' },
              { name: 'Testing', color: 'yellow' }
            ] 
          } 
        }
      }
    };

    const response = await this.request<any>('POST', '/v1/databases', databaseConfig);
    
    if (!response.success) {
      throw new Error('Failed to create strategy analysis database');
    }

    return response.data.id;
  }

  async addTradingJournalEntry(entry: TradingJournal): Promise<string> {
    if (!this.journalDatabaseId) {
      throw new Error('Trading journal database not initialized');
    }

    const pageConfig = {
      parent: { database_id: this.journalDatabaseId },
      properties: {
        'Date': { date: { start: entry.date } },
        'Symbol': { title: [{ text: { content: entry.symbol } }] },
        'Strategy': { select: { name: entry.strategy } },
        'Entry Price': { number: entry.entryPrice },
        'Exit Price': entry.exitPrice ? { number: entry.exitPrice } : null,
        'Quantity': { number: entry.quantity },
        'Notes': { rich_text: [{ text: { content: entry.notes } }] },
        'Tags': { 
          multi_select: entry.tags.map(tag => ({ name: tag }))
        },
        'Status': { select: { name: entry.status === 'open' ? 'Open' : 'Closed' } }
      }
    };

    const response = await this.request<any>('POST', '/v1/pages', pageConfig);
    
    if (!response.success) {
      throw new Error('Failed to add trading journal entry');
    }

    return response.data.id;
  }

  async updateTradingJournalEntry(pageId: string, updates: Partial<TradingJournal>): Promise<void> {
    const properties: any = {};

    if (updates.exitPrice !== undefined) {
      properties['Exit Price'] = { number: updates.exitPrice };
    }
    if (updates.status) {
      properties['Status'] = { select: { name: updates.status === 'open' ? 'Open' : 'Closed' } };
    }
    if (updates.notes) {
      properties['Notes'] = { rich_text: [{ text: { content: updates.notes } }] };
    }
    if (updates.tags) {
      properties['Tags'] = { multi_select: updates.tags.map(tag => ({ name: tag })) };
    }

    const response = await this.request('PATCH', `/v1/pages/${pageId}`, { properties });
    
    if (!response.success) {
      throw new Error('Failed to update trading journal entry');
    }
  }

  async addStrategyAnalysis(analysis: StrategyAnalysis): Promise<string> {
    if (!this.strategyDatabaseId) {
      throw new Error('Strategy analysis database not initialized');
    }

    const pageConfig = {
      parent: { database_id: this.strategyDatabaseId },
      properties: {
        'Strategy Name': { title: [{ text: { content: analysis.strategyName } }] },
        'Total Trades': { number: analysis.totalTrades },
        'Win Rate': { number: analysis.winRate },
        'Average Win': { number: analysis.avgWin },
        'Average Loss': { number: analysis.avgLoss },
        'Profit Factor': { number: analysis.profitFactor },
        'Max Drawdown': { number: analysis.maxDrawdown },
        'Status': { select: { name: 'Active' } }
      },
      children: [
        {
          type: 'paragraph',
          paragraph: {
            rich_text: [{ text: { content: analysis.analysis } }]
          }
        },
        {
          type: 'bulleted_list_item',
          bulleted_list_item: {
            rich_text: [{ text: { content: 'Recommendations:' } }]
          }
        },
        ...analysis.recommendations.map(rec => ({
          type: 'bulleted_list_item',
          bulleted_list_item: {
            rich_text: [{ text: { content: rec } }]
          }
        }))
      ]
    };

    const response = await this.request<any>('POST', '/v1/pages', pageConfig);
    
    if (!response.success) {
      throw new Error('Failed to add strategy analysis');
    }

    return response.data.id;
  }

  async getTradingJournalEntries(filters?: {
    symbol?: string;
    strategy?: string;
    status?: 'open' | 'closed';
    dateRange?: { start: string; end: string };
  }): Promise<TradingJournal[]> {
    if (!this.journalDatabaseId) {
      throw new Error('Trading journal database not initialized');
    }

    let filter: any = undefined;
    
    if (filters) {
      const conditions: any[] = [];
      
      if (filters.symbol) {
        conditions.push({
          property: 'Symbol',
          title: { equals: filters.symbol }
        });
      }
      
      if (filters.strategy) {
        conditions.push({
          property: 'Strategy',
          select: { equals: filters.strategy }
        });
      }
      
      if (filters.status) {
        conditions.push({
          property: 'Status',
          select: { equals: filters.status === 'open' ? 'Open' : 'Closed' }
        });
      }
      
      if (conditions.length > 0) {
        filter = conditions.length === 1 ? conditions[0] : {
          and: conditions
        };
      }
    }

    const response = await this.request<any>('POST', `/v1/databases/${this.journalDatabaseId}/query`, {
      filter
    });

    if (!response.success) {
      throw new Error('Failed to query trading journal entries');
    }

    return response.data.results.map(this.mapNotionPageToJournalEntry);
  }

  private mapNotionPageToJournalEntry(page: any): TradingJournal {
    const props = page.properties;
    
    return {
      date: props.Date?.date?.start || '',
      symbol: props.Symbol?.title?.[0]?.plain_text || '',
      strategy: props.Strategy?.select?.name || '',
      entryPrice: props['Entry Price']?.number || 0,
      exitPrice: props['Exit Price']?.number,
      quantity: props.Quantity?.number || 0,
      pnl: props['P&L']?.formula?.number,
      notes: props.Notes?.rich_text?.[0]?.plain_text || '',
      tags: props.Tags?.multi_select?.map(tag => tag.name) || [],
      status: props.Status?.select?.name === 'Open' ? 'open' : 'closed'
    };
  }

  private async getWorkspaceId(): Promise<string> {
    // This is a simplified approach - in reality, you'd need to get a specific page ID
    // or create a workspace page to serve as the parent for databases
    const response = await this.request<any>('POST', '/v1/search', {
      filter: { property: 'object', value: 'page' },
      page_size: 1
    });

    if (response.success && response.data?.results?.length > 0) {
      return response.data.results[0].id;
    }

    throw new Error('No accessible workspace found');
  }
}