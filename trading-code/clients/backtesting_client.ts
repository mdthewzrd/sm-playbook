import { BaseClient, ClientConfig, ClientRequest, ClientResponse } from './base_client';
import { spawn, ChildProcess } from 'child_process';
import { promises as fs } from 'fs';
import * as path from 'path';
import { CandleData } from '../models/market_data';
import { BacktestResult } from '../processors/backtest_processor';

export interface BacktestingConfig extends ClientConfig {
  pythonPath: string;
  scriptPath: string;
  workingDirectory: string;
  maxConcurrentBacktests: number;
  keepTempFiles: boolean;
}

export interface PythonBacktestConfig {
  cash: number;
  commission: number;
  margin: number;
  tradeOnOpen: boolean;
  tradeOnClose: boolean;
  hedging: boolean;
  exclusive_orders: boolean;
}

export interface BacktestStrategy {
  name: string;
  code: string;
  parameters: Record<string, any>;
  indicators: string[];
}

export interface PythonBacktestResult {
  trades: PythonTrade[];
  equity_curve: number[];
  stats: PythonStats;
  plots: PythonPlot[];
  logs: string[];
}

export interface PythonTrade {
  EntryTime: string;
  ExitTime: string;
  Duration: string;
  EntryPrice: number;
  ExitPrice: number;
  Size: number;
  PnL: number;
  ReturnPct: number;
  EntrySignal: string;
  ExitSignal: string;
}

export interface PythonStats {
  'Start': string;
  'End': string;
  'Duration': string;
  'Exposure Time [%]': number;
  'Equity Final [$]': number;
  'Equity Peak [$]': number;
  'Return [%]': number;
  'Buy & Hold Return [%]': number;
  'Return (Ann.) [%]': number;
  'Volatility (Ann.) [%]': number;
  'Sharpe Ratio': number;
  'Sortino Ratio': number;
  'Calmar Ratio': number;
  'Max. Drawdown [%]': number;
  'Avg. Drawdown [%]': number;
  'Max. Drawdown Duration': string;
  'Avg. Drawdown Duration': string;
  '# Trades': number;
  'Win Rate [%]': number;
  'Best Trade [%]': number;
  'Worst Trade [%]': number;
  'Avg. Trade [%]': number;
  'Max. Trade Duration': string;
  'Avg. Trade Duration': string;
  'Profit Factor': number;
  'Expectancy [$]': number;
  'SQN': number;
  '_strategy': string;
  '_equity_curve': number[];
  '_trades': PythonTrade[];
}

export interface PythonPlot {
  type: 'equity' | 'trades' | 'drawdown' | 'returns';
  data: any;
  title: string;
  filename?: string;
}

export interface BacktestJob {
  id: string;
  strategy: BacktestStrategy;
  data: CandleData[];
  config: PythonBacktestConfig;
  status: 'pending' | 'running' | 'completed' | 'failed';
  startTime: number;
  endTime?: number;
  result?: PythonBacktestResult;
  error?: string;
  tempFiles: string[];
}

export class BacktestingClient extends BaseClient {
  private pythoConfig: BacktestingConfig;
  private activeJobs: Map<string, BacktestJob> = new Map();
  private jobQueue: BacktestJob[] = [];
  private runningProcesses: Map<string, ChildProcess> = new Map();

  constructor(config: BacktestingConfig) {
    const baseConfig: ClientConfig = {
      ...config,
      host: 'localhost',
      protocol: 'http' as const,
      timeout: config.timeout || 300000, // 5 minutes default
      retryAttempts: config.retryAttempts || 2,
      retryDelay: config.retryDelay || 5000
    };

    super(baseConfig);
    this.pythoConfig = config;
  }

  async connect(): Promise<void> {
    try {
      await this.validatePythonEnvironment();
      await this.setupWorkingDirectory();
      this.emit('connected');
    } catch (error) {
      this.emit('error', error);
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    for (const [jobId, process] of this.runningProcesses) {
      process.kill('SIGTERM');
      this.runningProcesses.delete(jobId);
    }

    if (!this.pythoConfig.keepTempFiles) {
      await this.cleanupTempFiles();
    }

    this.emit('disconnected');
  }

  async isHealthy(): Promise<boolean> {
    try {
      await this.validatePythonEnvironment();
      return true;
    } catch {
      return false;
    }
  }

  protected async executeRequest<T>(request: ClientRequest): Promise<ClientResponse<T>> {
    const { method, endpoint, params } = request;

    try {
      let result: any;

      switch (endpoint) {
        case '/backtest':
          result = await this.runBacktest(params);
          break;
        case '/optimize':
          result = await this.optimizeStrategy(params);
          break;
        case '/validate':
          result = await this.validateStrategy(params);
          break;
        default:
          throw new Error(`Unknown endpoint: ${endpoint}`);
      }

      return {
        id: request.id,
        success: true,
        data: result as T,
        timestamp: Date.now(),
        duration: 0
      };

    } catch (error) {
      throw error;
    }
  }

  async runBacktest(params: {
    strategy: BacktestStrategy;
    data: CandleData[];
    config: PythonBacktestConfig;
  }): Promise<BacktestJob> {
    const jobId = `backtest_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const job: BacktestJob = {
      id: jobId,
      strategy: params.strategy,
      data: params.data,
      config: params.config,
      status: 'pending',
      startTime: Date.now(),
      tempFiles: []
    };

    this.activeJobs.set(jobId, job);

    if (this.runningProcesses.size < this.pythoConfig.maxConcurrentBacktests) {
      await this.executeBacktestJob(job);
    } else {
      this.jobQueue.push(job);
    }

    return job;
  }

  private async executeBacktestJob(job: BacktestJob): Promise<void> {
    try {
      job.status = 'running';
      this.emit('job:started', job);

      const dataFile = await this.writeDataToFile(job.data, job.id);
      const strategyFile = await this.writeStrategyToFile(job.strategy, job.id);
      const configFile = await this.writeConfigToFile(job.config, job.id);
      
      job.tempFiles.push(dataFile, strategyFile, configFile);

      const result = await this.executePythonBacktest(
        job.id,
        dataFile,
        strategyFile,
        configFile
      );

      job.result = result;
      job.status = 'completed';
      job.endTime = Date.now();
      
      this.emit('job:completed', job);

    } catch (error) {
      job.error = error.message;
      job.status = 'failed';
      job.endTime = Date.now();
      
      this.emit('job:failed', { job, error });
    } finally {
      this.runningProcesses.delete(job.id);
      this.processJobQueue();
    }
  }

  private async executePythonBacktest(
    jobId: string,
    dataFile: string,
    strategyFile: string,
    configFile: string
  ): Promise<PythonBacktestResult> {
    return new Promise((resolve, reject) => {
      const pythonScript = this.generateBacktestScript(dataFile, strategyFile, configFile);
      const scriptFile = path.join(this.pythoConfig.workingDirectory, `backtest_${jobId}.py`);
      
      fs.writeFile(scriptFile, pythonScript).then(() => {
        const process = spawn(this.pythoConfig.pythonPath, [scriptFile], {
          cwd: this.pythoConfig.workingDirectory,
          stdio: ['pipe', 'pipe', 'pipe']
        });

        this.runningProcesses.set(jobId, process);

        let stdout = '';
        let stderr = '';

        process.stdout?.on('data', (data) => {
          stdout += data.toString();
        });

        process.stderr?.on('data', (data) => {
          stderr += data.toString();
        });

        process.on('close', async (code) => {
          this.runningProcesses.delete(jobId);
          
          try {
            await fs.unlink(scriptFile);
          } catch {}

          if (code === 0) {
            try {
              const result = JSON.parse(stdout);
              resolve(result);
            } catch (parseError) {
              reject(new Error(`Failed to parse backtest result: ${parseError.message}`));
            }
          } else {
            reject(new Error(`Python process exited with code ${code}: ${stderr}`));
          }
        });

        process.on('error', (error) => {
          this.runningProcesses.delete(jobId);
          reject(new Error(`Failed to start Python process: ${error.message}`));
        });

        setTimeout(() => {
          if (this.runningProcesses.has(jobId)) {
            process.kill('SIGTERM');
            reject(new Error('Backtest timed out'));
          }
        }, this.config.timeout);
      }).catch(reject);
    });
  }

  private generateBacktestScript(dataFile: string, strategyFile: string, configFile: string): string {
    return `
import json
import pandas as pd
from backtesting import Backtest
import sys
import os

# Add strategy file directory to path
sys.path.insert(0, '${path.dirname(strategyFile)}')

try:
    # Load data
    data = pd.read_csv('${dataFile}', parse_dates=['timestamp'], index_col='timestamp')
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Load config
    with open('${configFile}', 'r') as f:
        config = json.load(f)
    
    # Import strategy
    strategy_module = __import__('${path.basename(strategyFile, '.py')}')
    Strategy = getattr(strategy_module, 'Strategy')
    
    # Run backtest
    bt = Backtest(data, Strategy, **config)
    result = bt.run()
    
    # Convert result to serializable format
    output = {
        'trades': result._trades.to_dict('records') if hasattr(result, '_trades') else [],
        'equity_curve': result._equity_curve.tolist() if hasattr(result, '_equity_curve') else [],
        'stats': {k: (v.total_seconds() if hasattr(v, 'total_seconds') else 
                     str(v) if not isinstance(v, (int, float)) else v) 
                 for k, v in result.items()},
        'plots': [],
        'logs': []
    }
    
    print(json.dumps(output, default=str))
    
except Exception as e:
    error = {
        'error': str(e),
        'type': type(e).__name__,
        'traceback': __import__('traceback').format_exc()
    }
    print(json.dumps(error))
    sys.exit(1)
`;
  }

  private async writeDataToFile(data: CandleData[], jobId: string): Promise<string> {
    const filename = path.join(this.pythoConfig.workingDirectory, `data_${jobId}.csv`);
    
    const csvContent = [
      'timestamp,open,high,low,close,volume',
      ...data.map(candle => 
        `${new Date(candle.timestamp).toISOString()},${candle.open},${candle.high},${candle.low},${candle.close},${candle.volume}`
      )
    ].join('\n');

    await fs.writeFile(filename, csvContent);
    return filename;
  }

  private async writeStrategyToFile(strategy: BacktestStrategy, jobId: string): Promise<string> {
    const filename = path.join(this.pythoConfig.workingDirectory, `strategy_${jobId}.py`);
    await fs.writeFile(filename, strategy.code);
    return filename;
  }

  private async writeConfigToFile(config: PythonBacktestConfig, jobId: string): Promise<string> {
    const filename = path.join(this.pythoConfig.workingDirectory, `config_${jobId}.json`);
    await fs.writeFile(filename, JSON.stringify(config, null, 2));
    return filename;
  }

  private async validatePythonEnvironment(): Promise<void> {
    return new Promise((resolve, reject) => {
      const process = spawn(this.pythoConfig.pythonPath, ['-c', 
        'import backtesting; import pandas; import numpy; print("OK")'
      ], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let output = '';
      let error = '';

      process.stdout?.on('data', (data) => output += data.toString());
      process.stderr?.on('data', (data) => error += data.toString());

      process.on('close', (code) => {
        if (code === 0 && output.trim() === 'OK') {
          resolve();
        } else {
          reject(new Error(`Python environment validation failed: ${error || 'Unknown error'}`));
        }
      });

      process.on('error', (err) => {
        reject(new Error(`Failed to execute Python: ${err.message}`));
      });
    });
  }

  private async setupWorkingDirectory(): Promise<void> {
    try {
      await fs.access(this.pythoConfig.workingDirectory);
    } catch {
      await fs.mkdir(this.pythoConfig.workingDirectory, { recursive: true });
    }
  }

  private async cleanupTempFiles(): Promise<void> {
    for (const job of this.activeJobs.values()) {
      for (const file of job.tempFiles) {
        try {
          await fs.unlink(file);
        } catch {}
      }
    }
  }

  private processJobQueue(): void {
    if (this.jobQueue.length > 0 && this.runningProcesses.size < this.pythoConfig.maxConcurrentBacktests) {
      const nextJob = this.jobQueue.shift();
      if (nextJob) {
        this.executeBacktestJob(nextJob);
      }
    }
  }

  async optimizeStrategy(params: {
    strategy: BacktestStrategy;
    data: CandleData[];
    config: PythonBacktestConfig;
    parameterRanges: Record<string, { min: number; max: number; step: number }>;
  }): Promise<any> {
    // Implementation for strategy optimization
    throw new Error('Strategy optimization not yet implemented');
  }

  async validateStrategy(params: {
    strategy: BacktestStrategy;
  }): Promise<{ valid: boolean; errors: string[] }> {
    try {
      const tempFile = path.join(this.pythoConfig.workingDirectory, `validate_${Date.now()}.py`);
      await fs.writeFile(tempFile, params.strategy.code);

      const result = await new Promise<{ valid: boolean; errors: string[] }>((resolve) => {
        const process = spawn(this.pythoConfig.pythonPath, ['-m', 'py_compile', tempFile], {
          stdio: ['pipe', 'pipe', 'pipe']
        });

        let stderr = '';
        process.stderr?.on('data', (data) => stderr += data.toString());

        process.on('close', async (code) => {
          try {
            await fs.unlink(tempFile);
          } catch {}

          resolve({
            valid: code === 0,
            errors: code === 0 ? [] : [stderr]
          });
        });
      });

      return result;
    } catch (error) {
      return {
        valid: false,
        errors: [`Validation failed: ${error.message}`]
      };
    }
  }

  getJob(jobId: string): BacktestJob | undefined {
    return this.activeJobs.get(jobId);
  }

  getAllJobs(): BacktestJob[] {
    return Array.from(this.activeJobs.values());
  }

  cancelJob(jobId: string): boolean {
    const process = this.runningProcesses.get(jobId);
    if (process) {
      process.kill('SIGTERM');
      this.runningProcesses.delete(jobId);
      
      const job = this.activeJobs.get(jobId);
      if (job) {
        job.status = 'failed';
        job.error = 'Cancelled by user';
        job.endTime = Date.now();
      }
      
      return true;
    }
    
    return false;
  }

  getQueueStatus(): { running: number; queued: number; total: number } {
    return {
      running: this.runningProcesses.size,
      queued: this.jobQueue.length,
      total: this.activeJobs.size
    };
  }
}