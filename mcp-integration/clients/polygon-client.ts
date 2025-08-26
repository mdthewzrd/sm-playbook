/**
 * PolygonClient - Interface with Polygon.io for comprehensive market data
 * 
 * This client provides integration with Polygon.io API for:
 * - Real-time and historical OHLCV data for multiple timeframes
 * - Market scanner functionality and symbol search
 * - WebSocket support for live data streaming
 * - News and fundamental data access
 */

import {
  BaseMCPClient,
  MCPServerConfig,
  MCPHealthCheck,
  MCPRequest,
  MCPResponse,
  OHLCV,
  MarketDataRequest,
  MCPError,
  MCPValidationError
} from '../types';

export interface PolygonClientConfig extends MCPServerConfig {
  apiKey: string;
  enableWebSocket: boolean;
  enableNewsData: boolean;
  defaultTimeframe: string;
  maxConcurrentRequests: number;
}

export interface TickerDetails {
  ticker: string;
  name: string;
  market: string;
  locale: string;
  primaryExchange: string;
  type: string;
  active: boolean;
  currencyName: string;
  cik?: string;
  compositeSymbol?: string;
  shareClassSymbol?: string;
  lastUpdatedUtc: string;
}

export interface MarketStatus {
  market: string;
  serverTime: string;
  exchanges: Record<string, 'open' | 'closed' | 'extended-hours'>;
  currencies: Record<string, 'open' | 'closed'>;
}

export interface TickerSnapshot {
  ticker: string;
  todaysChangePerc: number;
  todaysChange: number;
  updated: number;
  timeframe: string;
  value: number;
  day: {
    o: number;
    h: number;
    l: number;
    c: number;
    v: number;
    vw: number;
  };
  min: {
    av: number;
    t: number;
    n: number;
    o: number;
    h: number;
    l: number;
    c: number;
    v: number;
    vw: number;
  };
  prevDay: {
    o: number;
    h: number;
    l: number;
    c: number;
    v: number;
    vw: number;
  };
}

export interface NewsArticle {
  id: string;
  publisher: {
    name: string;
    homepage_url: string;
    logo_url: string;
    favicon_url: string;
  };
  title: string;
  author: string;
  published_utc: string;
  article_url: string;
  tickers: string[];
  amp_url?: string;
  image_url?: string;
  description: string;
  keywords: string[];
}

export interface ScannerCriteria {
  marketCap?: { min?: number; max?: number };
  price?: { min?: number; max?: number };
  volume?: { min?: number; max?: number };
  changePercent?: { min?: number; max?: number };
  sector?: string[];
  exchange?: string[];
  limit?: number;
}

export interface ScannerResult {
  ticker: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  sector: string;
  exchange: string;
  pe?: number;
  beta?: number;
}

export interface FinancialData {
  ticker: string;
  period: string;
  calendarDate: string;
  reportPeriod: string;
  updated: string;
  accumulatedOtherComprehensiveIncome?: number;
  assets?: number;
  assetsAverage?: number;
  assetsCurrent?: number;
  assetsNonCurrent?: number;
  bookValuePerShare?: number;
  capitalExpenditure?: number;
  cashAndEquivalents?: number;
  cashAndEquivalentsUSD?: number;
  costOfRevenue?: number;
  consolidatedIncome?: number;
  currentRatio?: number;
  debtToEquityRatio?: number;
  debt?: number;
  debtCurrent?: number;
  debtNonCurrent?: number;
  debtUSD?: number;
  deferredRevenue?: number;
  depreciationAmortizationAndAccretion?: number;
  deposits?: number;
  dividendYield?: number;
  dividendsPerBasicCommonShare?: number;
  earningBeforeInterestTaxes?: number;
  earningsBeforeInterestTaxesDepreciationAmortization?: number;
  EBITDAMargin?: number;
  earningsBeforeInterestTaxesDepreciationAmortizationUSD?: number;
  earningBeforeInterestTaxesUSD?: number;
  earningsBeforeTax?: number;
  earningsPerBasicShare?: number;
  earningsPerDilutedShare?: number;
  earningsPerBasicShareUSD?: number;
  shareholderEquity?: number;
  averageEquity?: number;
  shareholderEquityUSD?: number;
  enterpriseValue?: number;
  enterpriseValueOverEBIT?: number;
  enterpriseValueOverEBITDA?: number;
  freeCashFlow?: number;
  freeCashFlowPerShare?: number;
  foreignCurrencyUSDExchangeRate?: number;
  grossProfit?: number;
  grossMargin?: number;
  goodwillAndIntangibleAssets?: number;
  interestExpense?: number;
  investedCapital?: number;
  investedCapitalAverage?: number;
  inventory?: number;
  investments?: number;
  investmentsCurrent?: number;
  investmentsNonCurrent?: number;
  totalLiabilities?: number;
  currentLiabilities?: number;
  liabilitiesNonCurrent?: number;
  marketCapitalization?: number;
  netCashFlow?: number;
  netCashFlowBusinessAcquisitionsDisposals?: number;
  issuanceEquityShares?: number;
  issuanceDebtSecurities?: number;
  paymentDividendsOtherCashDistributions?: number;
  netCashFlowFromFinancing?: number;
  netCashFlowFromInvesting?: number;
  netCashFlowInvestmentAcquisitionsDisposals?: number;
  netCashFlowFromOperations?: number;
  effectOfExchangeRateChangesOnCash?: number;
  netIncome?: number;
  netIncomeCommonStock?: number;
  netIncomeCommonStockUSD?: number;
  netLossIncomeFromDiscontinuedOperations?: number;
  netIncomeToNonControllingInterests?: number;
  profitMargin?: number;
  operatingExpenses?: number;
  operatingIncome?: number;
  tradeAndNonTradePayables?: number;
  payoutRatio?: number;
  priceToBookValue?: number;
  priceEarnings?: number;
  priceToEarningsRatio?: number;
  propertyPlantEquipmentNet?: number;
  preferredDividendsIncomeStatementImpact?: number;
  sharePriceAdjustedClose?: number;
  priceSales?: number;
  priceToSalesRatio?: number;
  tradeAndNonTradeReceivables?: number;
  accumulatedRetainedEarningsDeficit?: number;
  revenues?: number;
  revenuesUSD?: number;
  researchAndDevelopmentExpense?: number;
  shareBasedCompensation?: number;
  sellingGeneralAndAdministrativeExpense?: number;
  shareFactor?: number;
  shares?: number;
  weightedAverageShares?: number;
  weightedAverageSharesDiluted?: number;
  salesPerShare?: number;
  tangibleAssetValue?: number;
  taxAssets?: number;
  incomeTaxExpense?: number;
  taxLiabilities?: number;
  tangibleAssetsBookValuePerShare?: number;
  workingCapital?: number;
}

export interface AggregateBar {
  T: string; // ticker
  v: number; // volume
  vw: number; // volume weighted average price
  o: number; // open
  c: number; // close
  h: number; // high
  l: number; // low
  t: number; // timestamp
  n: number; // number of transactions
}

export interface WebSocketSubscription {
  channel: string;
  symbol: string;
  callback: (data: any) => void;
}

export class PolygonClient extends BaseMCPClient {
  private clientConfig: PolygonClientConfig;
  private subscriptions: Map<string, WebSocketSubscription> = new Map();
  private rateLimitRemaining: number = 1000;
  private rateLimitReset: Date = new Date();

  constructor(config: PolygonClientConfig) {
    super('polygon', config);
    this.clientConfig = {
      apiKey: '',
      enableWebSocket: true,
      enableNewsData: true,
      defaultTimeframe: '1Day',
      maxConcurrentRequests: 10,
      ...config
    };

    if (!this.clientConfig.apiKey && !config.env?.POLYGON_API_KEY) {
      throw new MCPValidationError('polygon', 'Polygon API key is required');
    }

    this.clientConfig.apiKey = this.clientConfig.apiKey || config.env?.POLYGON_API_KEY || '';
  }

  async connect(): Promise<void> {
    try {
      const response = await this.request<void, MarketStatus>({
        method: 'market/status'
      });

      if (response.success && response.data) {
        this.connectionStatus = {
          connected: true,
          lastHeartbeat: new Date(),
          retryCount: 0
        };

        if (this.clientConfig.enableWebSocket) {
          await this.initializeWebSocket();
        }
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
    // Close WebSocket connections
    if (this.clientConfig.enableWebSocket) {
      await this.closeWebSocket();
    }

    this.subscriptions.clear();
    this.connectionStatus = {
      connected: false,
      retryCount: 0
    };
  }

  async healthCheck(): Promise<MCPHealthCheck> {
    const startTime = Date.now();
    
    try {
      const response = await this.request<void, MarketStatus>({
        method: 'market/status',
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
    
    // Check rate limits
    await this.checkRateLimit();
    
    try {
      const result = await this.makePolygonApiCall(request.method, request.params);
      
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
   * Get historical OHLCV data
   */
  async getHistoricalData(request: MarketDataRequest): Promise<OHLCV[]> {
    this.validateMarketDataRequest(request);

    const response = await this.request<MarketDataRequest, OHLCV[]>({
      method: 'market-data/historical',
      params: request
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get historical data: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Get real-time quote data
   */
  async getQuote(symbol: string): Promise<{
    ticker: string;
    last: {
      price: number;
      size: number;
      exchange: number;
      timestamp: number;
    };
    bid: {
      price: number;
      size: number;
      exchange: number;
    };
    ask: {
      price: number;
      size: number;
      exchange: number;
    };
  }> {
    const response = await this.request<{ symbol: string }, any>({
      method: 'market-data/quote',
      params: { symbol }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get quote for ${symbol}: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Get ticker snapshot
   */
  async getSnapshot(symbol: string): Promise<TickerSnapshot> {
    const response = await this.request<{ symbol: string }, TickerSnapshot>({
      method: 'market-data/snapshot',
      params: { symbol }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get snapshot for ${symbol}: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Get multiple ticker snapshots
   */
  async getSnapshots(symbols: string[]): Promise<TickerSnapshot[]> {
    const response = await this.request<{ symbols: string[] }, TickerSnapshot[]>({
      method: 'market-data/snapshots',
      params: { symbols }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get snapshots: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Get market status
   */
  async getMarketStatus(): Promise<MarketStatus> {
    const response = await this.request<void, MarketStatus>({
      method: 'market/status'
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get market status: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Get ticker details
   */
  async getTickerDetails(symbol: string): Promise<TickerDetails> {
    const response = await this.request<{ symbol: string }, TickerDetails>({
      method: 'reference/tickers/details',
      params: { symbol }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get ticker details for ${symbol}: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Search for tickers
   */
  async searchTickers(
    query: string,
    type?: string,
    market?: string,
    limit?: number
  ): Promise<TickerDetails[]> {
    const response = await this.request<{
      query: string;
      type?: string;
      market?: string;
      limit?: number;
    }, TickerDetails[]>({
      method: 'reference/tickers/search',
      params: { query, type, market, limit }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to search tickers: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Get news articles
   */
  async getNews(
    ticker?: string,
    publishedUtc?: string,
    order?: 'asc' | 'desc',
    limit?: number
  ): Promise<NewsArticle[]> {
    if (!this.clientConfig.enableNewsData) {
      throw new MCPError('News data is disabled', this.serverId);
    }

    const response = await this.request<{
      ticker?: string;
      publishedUtc?: string;
      order?: string;
      limit?: number;
    }, NewsArticle[]>({
      method: 'reference/news',
      params: { ticker, publishedUtc, order, limit }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get news: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Run market scanner
   */
  async scanMarket(criteria: ScannerCriteria): Promise<ScannerResult[]> {
    const response = await this.request<ScannerCriteria, ScannerResult[]>({
      method: 'market/scanner',
      params: criteria
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to run market scanner: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Get financial data
   */
  async getFinancials(
    ticker: string,
    period?: 'annual' | 'quarterly',
    limit?: number
  ): Promise<FinancialData[]> {
    const response = await this.request<{
      ticker: string;
      period?: string;
      limit?: number;
    }, FinancialData[]>({
      method: 'reference/financials',
      params: { ticker, period, limit }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get financials for ${ticker}: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Get gainers and losers
   */
  async getGainersLosers(direction: 'gainers' | 'losers' = 'gainers'): Promise<TickerSnapshot[]> {
    const response = await this.request<{ direction: string }, TickerSnapshot[]>({
      method: 'market/gainers-losers',
      params: { direction }
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get ${direction}: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Get most active stocks
   */
  async getMostActive(): Promise<TickerSnapshot[]> {
    const response = await this.request<void, TickerSnapshot[]>({
      method: 'market/most-active'
    });

    if (!response.success || !response.data) {
      throw new MCPError(`Failed to get most active stocks: ${response.error}`, this.serverId);
    }

    return response.data;
  }

  /**
   * Subscribe to real-time data
   */
  async subscribe(
    channel: 'trades' | 'quotes' | 'aggregates',
    symbol: string,
    callback: (data: any) => void
  ): Promise<string> {
    if (!this.clientConfig.enableWebSocket) {
      throw new MCPError('WebSocket is disabled', this.serverId);
    }

    const subscriptionId = `${channel}_${symbol}_${Date.now()}`;
    const subscription: WebSocketSubscription = {
      channel,
      symbol,
      callback
    };

    this.subscriptions.set(subscriptionId, subscription);

    // Send subscription request via WebSocket
    await this.request<{
      action: string;
      params: { channels: string[]; symbols: string[] };
    }, void>({
      method: 'websocket/subscribe',
      params: {
        action: 'subscribe',
        params: {
          channels: [channel],
          symbols: [symbol]
        }
      }
    });

    return subscriptionId;
  }

  /**
   * Unsubscribe from real-time data
   */
  async unsubscribe(subscriptionId: string): Promise<void> {
    const subscription = this.subscriptions.get(subscriptionId);
    if (!subscription) {
      throw new MCPError(`Subscription ${subscriptionId} not found`, this.serverId);
    }

    await this.request<{
      action: string;
      params: { channels: string[]; symbols: string[] };
    }, void>({
      method: 'websocket/unsubscribe',
      params: {
        action: 'unsubscribe',
        params: {
          channels: [subscription.channel],
          symbols: [subscription.symbol]
        }
      }
    });

    this.subscriptions.delete(subscriptionId);
  }

  /**
   * Get all active subscriptions
   */
  getSubscriptions(): Record<string, WebSocketSubscription> {
    const subscriptions: Record<string, WebSocketSubscription> = {};
    for (const [id, subscription] of this.subscriptions.entries()) {
      subscriptions[id] = { ...subscription };
    }
    return subscriptions;
  }

  /**
   * Get rate limit status
   */
  getRateLimitStatus(): {
    remaining: number;
    reset: Date;
    limit: number;
  } {
    return {
      remaining: this.rateLimitRemaining,
      reset: this.rateLimitReset,
      limit: 1000 // Mock limit
    };
  }

  private async initializeWebSocket(): Promise<void> {
    // Mock WebSocket initialization
    console.log('WebSocket initialized for Polygon client');
  }

  private async closeWebSocket(): Promise<void> {
    // Mock WebSocket cleanup
    console.log('WebSocket closed for Polygon client');
  }

  private async checkRateLimit(): Promise<void> {
    if (this.rateLimitRemaining <= 0 && new Date() < this.rateLimitReset) {
      const waitTime = this.rateLimitReset.getTime() - Date.now();
      throw new MCPError(`Rate limit exceeded. Reset in ${waitTime}ms`, this.serverId);
    }

    if (new Date() >= this.rateLimitReset) {
      this.rateLimitRemaining = 1000;
      this.rateLimitReset = new Date(Date.now() + 60000); // 1 minute from now
    }

    this.rateLimitRemaining--;
  }

  private validateMarketDataRequest(request: MarketDataRequest): void {
    if (!request.symbol) {
      throw new MCPValidationError(this.serverId, 'Symbol is required');
    }

    if (!request.timeframe) {
      throw new MCPValidationError(this.serverId, 'Timeframe is required');
    }

    if (request.from && request.to && request.from >= request.to) {
      throw new MCPValidationError(this.serverId, 'From date must be before to date');
    }

    if (request.limit && (request.limit <= 0 || request.limit > 50000)) {
      throw new MCPValidationError(this.serverId, 'Limit must be between 1 and 50000');
    }
  }

  private async makePolygonApiCall(method: string, params?: any): Promise<any> {
    // Mock implementation - in reality this would communicate with the MCP server
    switch (method) {
      case 'market/status':
        return this.mockMarketStatus();

      case 'market-data/historical':
        return this.mockHistoricalData(params);

      case 'market-data/quote':
        return this.mockQuote(params.symbol);

      case 'market-data/snapshot':
        return this.mockSnapshot(params.symbol);

      case 'market-data/snapshots':
        return params.symbols.map((symbol: string) => this.mockSnapshot(symbol));

      case 'reference/tickers/details':
        return this.mockTickerDetails(params.symbol);

      case 'reference/tickers/search':
        return this.mockTickerSearch();

      case 'reference/news':
        return this.mockNews();

      case 'reference/financials':
        return this.mockFinancials();

      case 'market/scanner':
        return this.mockScannerResults();

      case 'market/gainers-losers':
        return this.mockGainersLosers();

      case 'market/most-active':
        return this.mockMostActive();

      case 'websocket/subscribe':
      case 'websocket/unsubscribe':
        return; // WebSocket operations don't return data

      default:
        throw new MCPError(`Unknown method: ${method}`, this.serverId);
    }
  }

  private mockMarketStatus(): MarketStatus {
    return {
      market: 'open',
      serverTime: new Date().toISOString(),
      exchanges: {
        nasdaq: 'open',
        nyse: 'open',
        otc: 'open'
      },
      currencies: {
        fx: 'open',
        crypto: 'open'
      }
    };
  }

  private mockHistoricalData(params: MarketDataRequest): OHLCV[] {
    const data: OHLCV[] = [];
    const basePrice = 100;
    const startDate = params.from || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
    const endDate = params.to || new Date();
    const limit = params.limit || 100;

    for (let i = 0; i < Math.min(limit, 30); i++) {
      const timestamp = new Date(startDate.getTime() + (i * 24 * 60 * 60 * 1000));
      const variation = (Math.random() - 0.5) * 10;
      const open = basePrice + variation;
      const close = open + (Math.random() - 0.5) * 5;
      const high = Math.max(open, close) + Math.random() * 2;
      const low = Math.min(open, close) - Math.random() * 2;

      data.push({
        timestamp,
        open,
        high,
        low,
        close,
        volume: Math.floor(Math.random() * 1000000) + 100000
      });
    }

    return data;
  }

  private mockQuote(symbol: string): any {
    return {
      ticker: symbol,
      last: {
        price: 100 + Math.random() * 50,
        size: 100,
        exchange: 1,
        timestamp: Date.now()
      },
      bid: {
        price: 99.5,
        size: 200,
        exchange: 1
      },
      ask: {
        price: 100.5,
        size: 150,
        exchange: 1
      }
    };
  }

  private mockSnapshot(symbol: string): TickerSnapshot {
    const price = 100 + Math.random() * 50;
    const change = (Math.random() - 0.5) * 10;
    
    return {
      ticker: symbol,
      todaysChangePerc: change,
      todaysChange: change,
      updated: Date.now(),
      timeframe: 'DELAYED',
      value: price,
      day: {
        o: price - 1,
        h: price + 2,
        l: price - 2,
        c: price,
        v: 1000000,
        vw: price
      },
      min: {
        av: 50000,
        t: Date.now(),
        n: 100,
        o: price,
        h: price + 0.5,
        l: price - 0.5,
        c: price,
        v: 10000,
        vw: price
      },
      prevDay: {
        o: price - 2,
        h: price,
        l: price - 3,
        c: price - 1,
        v: 900000,
        vw: price - 1
      }
    };
  }

  private mockTickerDetails(symbol: string): TickerDetails {
    return {
      ticker: symbol,
      name: `${symbol} Inc.`,
      market: 'stocks',
      locale: 'us',
      primaryExchange: 'NASDAQ',
      type: 'CS',
      active: true,
      currencyName: 'usd',
      cik: '0001234567',
      compositeSymbol: symbol,
      shareClassSymbol: symbol,
      lastUpdatedUtc: new Date().toISOString()
    };
  }

  private mockTickerSearch(): TickerDetails[] {
    return [
      this.mockTickerDetails('AAPL'),
      this.mockTickerDetails('GOOGL'),
      this.mockTickerDetails('MSFT')
    ];
  }

  private mockNews(): NewsArticle[] {
    return [
      {
        id: '1',
        publisher: {
          name: 'MarketWatch',
          homepage_url: 'https://www.marketwatch.com',
          logo_url: 'https://www.marketwatch.com/logo.png',
          favicon_url: 'https://www.marketwatch.com/favicon.ico'
        },
        title: 'Market Update: Stocks Rise on Economic Data',
        author: 'John Smith',
        published_utc: new Date().toISOString(),
        article_url: 'https://www.marketwatch.com/article/1',
        tickers: ['AAPL', 'GOOGL'],
        description: 'Stocks moved higher following positive economic indicators.',
        keywords: ['stocks', 'economy', 'market']
      }
    ];
  }

  private mockFinancials(): FinancialData[] {
    return [
      {
        ticker: 'AAPL',
        period: 'quarterly',
        calendarDate: '2023-12-31',
        reportPeriod: 'Q4',
        updated: new Date().toISOString(),
        revenues: 119575000000,
        netIncome: 33916000000,
        assets: 352755000000,
        shareholderEquity: 62146000000,
        marketCapitalization: 3000000000000
      }
    ];
  }

  private mockScannerResults(): ScannerResult[] {
    return [
      {
        ticker: 'AAPL',
        name: 'Apple Inc.',
        price: 150.25,
        change: 2.50,
        changePercent: 1.69,
        volume: 50000000,
        marketCap: 2500000000000,
        sector: 'Technology',
        exchange: 'NASDAQ',
        pe: 25.6,
        beta: 1.2
      }
    ];
  }

  private mockGainersLosers(): TickerSnapshot[] {
    return [
      this.mockSnapshot('AAPL'),
      this.mockSnapshot('GOOGL')
    ];
  }

  private mockMostActive(): TickerSnapshot[] {
    return [
      this.mockSnapshot('SPY'),
      this.mockSnapshot('QQQ'),
      this.mockSnapshot('TSLA')
    ];
  }
}