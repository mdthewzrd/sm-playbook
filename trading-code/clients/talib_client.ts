import { BaseClient, ClientConfig, ClientRequest, ClientResponse } from './base_client';
import { spawn, ChildProcess } from 'child_process';
import { promises as fs } from 'fs';
import * as path from 'path';
import { CandleData } from '../models/market_data';

export interface TALibConfig extends ClientConfig {
  pythonPath: string;
  workingDirectory: string;
  keepTempFiles: boolean;
}

export interface TALibIndicator {
  name: string;
  function: string;
  parameters: Record<string, any>;
  inputs: ('open' | 'high' | 'low' | 'close' | 'volume')[];
  outputs: string[];
}

export interface TALibRequest {
  indicator: TALibIndicator;
  data: CandleData[];
  options?: {
    startIdx?: number;
    endIdx?: number;
    outputType?: 'array' | 'object';
  };
}

export interface TALibResponse {
  indicator: string;
  values: number[] | Record<string, number[]>;
  metadata: {
    periods: number;
    validOutputs: number;
    inputLength: number;
    startIdx: number;
    endIdx: number;
  };
}

export interface TALibFunction {
  name: string;
  group: string;
  inputs: TALibInput[];
  optInputs: TALibOptInput[];
  outputs: TALibOutput[];
  description: string;
}

export interface TALibInput {
  name: string;
  type: 'price' | 'real';
  description?: string;
}

export interface TALibOptInput {
  name: string;
  type: 'integer' | 'real' | 'matype';
  defaultValue: any;
  min?: number;
  max?: number;
  description?: string;
}

export interface TALibOutput {
  name: string;
  type: 'real' | 'integer';
  description?: string;
}

export class TALibClient extends BaseClient {
  private talibConfig: TALibConfig;
  private availableFunctions: Map<string, TALibFunction> = new Map();
  private initialized: boolean = false;

  constructor(config: TALibConfig) {
    const baseConfig: ClientConfig = {
      ...config,
      host: 'localhost',
      protocol: 'http' as const,
      timeout: config.timeout || 60000,
      retryAttempts: config.retryAttempts || 2,
      retryDelay: config.retryDelay || 1000
    };

    super(baseConfig);
    this.talibConfig = config;
  }

  async connect(): Promise<void> {
    try {
      await this.validateTALibEnvironment();
      await this.setupWorkingDirectory();
      await this.loadAvailableFunctions();
      this.initialized = true;
      this.emit('connected');
    } catch (error) {
      this.emit('error', error);
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    if (!this.talibConfig.keepTempFiles) {
      await this.cleanupTempFiles();
    }
    this.emit('disconnected');
  }

  async isHealthy(): Promise<boolean> {
    try {
      await this.validateTALibEnvironment();
      return true;
    } catch {
      return false;
    }
  }

  protected async executeRequest<T>(request: ClientRequest): Promise<ClientResponse<T>> {
    if (!this.initialized) {
      throw new Error('TALib client not initialized');
    }

    const { method, endpoint, params } = request;

    try {
      let result: any;

      switch (endpoint) {
        case '/calculate':
          result = await this.calculateIndicator(params);
          break;
        case '/functions':
          result = await this.getFunctions();
          break;
        case '/function-info':
          result = await this.getFunctionInfo(params.name);
          break;
        case '/batch-calculate':
          result = await this.batchCalculate(params);
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

  async calculateIndicator(request: TALibRequest): Promise<TALibResponse> {
    const { indicator, data, options = {} } = request;
    
    if (!this.availableFunctions.has(indicator.function.toUpperCase())) {
      throw new Error(`Unknown TA-Lib function: ${indicator.function}`);
    }

    const tempId = `calc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const dataFile = await this.writeDataToFile(data, tempId);
    
    try {
      const result = await this.executeTALibCalculation(indicator, dataFile, options);
      
      if (!this.talibConfig.keepTempFiles) {
        await fs.unlink(dataFile);
      }
      
      return result;
    } catch (error) {
      if (!this.talibConfig.keepTempFiles) {
        try {
          await fs.unlink(dataFile);
        } catch {}
      }
      throw error;
    }
  }

  private async executeTALibCalculation(
    indicator: TALibIndicator,
    dataFile: string,
    options: any
  ): Promise<TALibResponse> {
    return new Promise((resolve, reject) => {
      const pythonScript = this.generateCalculationScript(indicator, dataFile, options);
      const scriptFile = path.join(this.talibConfig.workingDirectory, `script_${Date.now()}.py`);
      
      fs.writeFile(scriptFile, pythonScript).then(() => {
        const process = spawn(this.talibConfig.pythonPath, [scriptFile], {
          cwd: this.talibConfig.workingDirectory,
          stdio: ['pipe', 'pipe', 'pipe']
        });

        let stdout = '';
        let stderr = '';

        process.stdout?.on('data', (data) => {
          stdout += data.toString();
        });

        process.stderr?.on('data', (data) => {
          stderr += data.toString();
        });

        process.on('close', async (code) => {
          try {
            await fs.unlink(scriptFile);
          } catch {}

          if (code === 0) {
            try {
              const result = JSON.parse(stdout);
              resolve(result);
            } catch (parseError) {
              reject(new Error(`Failed to parse TA-Lib result: ${parseError.message}`));
            }
          } else {
            reject(new Error(`TA-Lib calculation failed: ${stderr}`));
          }
        });

        process.on('error', (error) => {
          reject(new Error(`Failed to execute TA-Lib script: ${error.message}`));
        });
      }).catch(reject);
    });
  }

  private generateCalculationScript(
    indicator: TALibIndicator,
    dataFile: string,
    options: any
  ): string {
    const functionName = indicator.function.toUpperCase();
    const params = Object.entries(indicator.parameters)
      .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
      .join(', ');

    return `
import json
import pandas as pd
import numpy as np
import talib
from talib import abstract

try:
    # Load data
    data = pd.read_csv('${dataFile}')
    
    # Convert to numpy arrays
    inputs = {}
    ${indicator.inputs.map(input => `
    if '${input}' in data.columns:
        inputs['${input}'] = data['${input}'].values`).join('')}
    
    # Calculate indicator
    func = abstract.Function('${functionName}')
    
    # Set parameters
    ${Object.entries(indicator.parameters).map(([key, value]) => 
      `func.set_parameters(${key}=${JSON.stringify(value)})`).join('\n    ')}
    
    # Execute calculation
    result = func(inputs)
    
    # Handle different output types
    if isinstance(result, np.ndarray):
        values = result.tolist()
        # Remove NaN values and convert to list
        values = [None if pd.isna(x) else float(x) for x in values]
    elif isinstance(result, tuple):
        # Multiple outputs
        values = {}
        outputs = ${JSON.stringify(indicator.outputs)}
        for i, output_name in enumerate(outputs):
            if i < len(result):
                arr = result[i]
                values[output_name] = [None if pd.isna(x) else float(x) for x in arr.tolist()]
    else:
        values = result
    
    # Calculate metadata
    valid_outputs = 0
    if isinstance(values, list):
        valid_outputs = sum(1 for x in values if x is not None)
    elif isinstance(values, dict):
        valid_outputs = sum(sum(1 for x in arr if x is not None) for arr in values.values())
    
    output = {
        'indicator': '${indicator.name}',
        'values': values,
        'metadata': {
            'periods': ${indicator.parameters.timeperiod || indicator.parameters.period || 'None'},
            'validOutputs': valid_outputs,
            'inputLength': len(data),
            'startIdx': ${options.startIdx || 0},
            'endIdx': ${options.endIdx || 'len(data) - 1'}
        }
    }
    
    print(json.dumps(output, default=str))
    
except Exception as e:
    import traceback
    error = {
        'error': str(e),
        'type': type(e).__name__,
        'traceback': traceback.format_exc()
    }
    print(json.dumps(error))
    import sys
    sys.exit(1)
`;
  }

  async batchCalculate(requests: TALibRequest[]): Promise<TALibResponse[]> {
    const results: TALibResponse[] = [];
    
    for (const request of requests) {
      try {
        const result = await this.calculateIndicator(request);
        results.push(result);
      } catch (error) {
        // Continue with other calculations even if one fails
        results.push({
          indicator: request.indicator.name,
          values: [],
          metadata: {
            periods: 0,
            validOutputs: 0,
            inputLength: request.data.length,
            startIdx: 0,
            endIdx: request.data.length - 1
          }
        });
      }
    }
    
    return results;
  }

  private async writeDataToFile(data: CandleData[], tempId: string): Promise<string> {
    const filename = path.join(this.talibConfig.workingDirectory, `data_${tempId}.csv`);
    
    const csvContent = [
      'timestamp,open,high,low,close,volume',
      ...data.map(candle => 
        `${candle.timestamp},${candle.open},${candle.high},${candle.low},${candle.close},${candle.volume}`
      )
    ].join('\n');

    await fs.writeFile(filename, csvContent);
    return filename;
  }

  private async validateTALibEnvironment(): Promise<void> {
    return new Promise((resolve, reject) => {
      const process = spawn(this.talibConfig.pythonPath, ['-c', 
        'import talib; import pandas; import numpy; print("OK")'
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
          reject(new Error(`TA-Lib environment validation failed: ${error || 'Unknown error'}`));
        }
      });

      process.on('error', (err) => {
        reject(new Error(`Failed to execute Python: ${err.message}`));
      });
    });
  }

  private async setupWorkingDirectory(): Promise<void> {
    try {
      await fs.access(this.talibConfig.workingDirectory);
    } catch {
      await fs.mkdir(this.talibConfig.workingDirectory, { recursive: true });
    }
  }

  private async loadAvailableFunctions(): Promise<void> {
    return new Promise((resolve, reject) => {
      const script = `
import json
import talib
from talib import abstract

try:
    functions = {}
    
    for func_name in talib.get_functions():
        func = abstract.Function(func_name)
        
        functions[func_name] = {
            'name': func_name,
            'group': talib.get_function_groups().get(func_name, 'Unknown'),
            'inputs': [{'name': inp, 'type': 'price'} for inp in func.input_names],
            'optInputs': [
                {
                    'name': opt,
                    'type': 'real' if 'price' in opt.lower() else 'integer',
                    'defaultValue': func.parameters.get(opt)
                } for opt in func.parameters.keys()
            ],
            'outputs': [{'name': out, 'type': 'real'} for out in func.output_names],
            'description': f'{func_name} - {talib.get_function_groups().get(func_name, "Technical Analysis Function")}'
        }
    
    print(json.dumps(functions))
    
except Exception as e:
    import traceback
    error = {
        'error': str(e),
        'traceback': traceback.format_exc()
    }
    print(json.dumps(error))
    import sys
    sys.exit(1)
`;

      const process = spawn(this.talibConfig.pythonPath, ['-c', script], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let stdout = '';
      let stderr = '';

      process.stdout?.on('data', (data) => stdout += data.toString());
      process.stderr?.on('data', (data) => stderr += data.toString());

      process.on('close', (code) => {
        if (code === 0) {
          try {
            const functions = JSON.parse(stdout);
            for (const [name, func] of Object.entries(functions)) {
              this.availableFunctions.set(name, func as TALibFunction);
            }
            resolve();
          } catch (parseError) {
            reject(new Error(`Failed to parse TA-Lib functions: ${parseError.message}`));
          }
        } else {
          reject(new Error(`Failed to load TA-Lib functions: ${stderr}`));
        }
      });
    });
  }

  private async cleanupTempFiles(): Promise<void> {
    try {
      const files = await fs.readdir(this.talibConfig.workingDirectory);
      const tempFiles = files.filter(file => 
        file.startsWith('data_') || file.startsWith('script_')
      );

      for (const file of tempFiles) {
        try {
          await fs.unlink(path.join(this.talibConfig.workingDirectory, file));
        } catch {}
      }
    } catch {}
  }

  async getFunctions(): Promise<TALibFunction[]> {
    if (!this.initialized) {
      await this.loadAvailableFunctions();
    }
    return Array.from(this.availableFunctions.values());
  }

  async getFunctionInfo(functionName: string): Promise<TALibFunction | null> {
    if (!this.initialized) {
      await this.loadAvailableFunctions();
    }
    return this.availableFunctions.get(functionName.toUpperCase()) || null;
  }

  // Convenience methods for common indicators
  async calculateSMA(data: CandleData[], period: number = 14): Promise<number[]> {
    const result = await this.calculateIndicator({
      indicator: {
        name: 'SMA',
        function: 'SMA',
        parameters: { timeperiod: period },
        inputs: ['close'],
        outputs: ['sma']
      },
      data
    });

    return Array.isArray(result.values) ? result.values : [];
  }

  async calculateEMA(data: CandleData[], period: number = 14): Promise<number[]> {
    const result = await this.calculateIndicator({
      indicator: {
        name: 'EMA',
        function: 'EMA',
        parameters: { timeperiod: period },
        inputs: ['close'],
        outputs: ['ema']
      },
      data
    });

    return Array.isArray(result.values) ? result.values : [];
  }

  async calculateRSI(data: CandleData[], period: number = 14): Promise<number[]> {
    const result = await this.calculateIndicator({
      indicator: {
        name: 'RSI',
        function: 'RSI',
        parameters: { timeperiod: period },
        inputs: ['close'],
        outputs: ['rsi']
      },
      data
    });

    return Array.isArray(result.values) ? result.values : [];
  }

  async calculateMACD(
    data: CandleData[], 
    fastPeriod: number = 12, 
    slowPeriod: number = 26, 
    signalPeriod: number = 9
  ): Promise<{ macd: number[], signal: number[], histogram: number[] }> {
    const result = await this.calculateIndicator({
      indicator: {
        name: 'MACD',
        function: 'MACD',
        parameters: { 
          fastperiod: fastPeriod, 
          slowperiod: slowPeriod, 
          signalperiod: signalPeriod 
        },
        inputs: ['close'],
        outputs: ['macd', 'macdsignal', 'macdhist']
      },
      data
    });

    if (typeof result.values === 'object' && !Array.isArray(result.values)) {
      return {
        macd: result.values.macd || [],
        signal: result.values.macdsignal || [],
        histogram: result.values.macdhist || []
      };
    }

    return { macd: [], signal: [], histogram: [] };
  }

  async calculateBollingerBands(
    data: CandleData[], 
    period: number = 20, 
    stdDev: number = 2
  ): Promise<{ upper: number[], middle: number[], lower: number[] }> {
    const result = await this.calculateIndicator({
      indicator: {
        name: 'BBANDS',
        function: 'BBANDS',
        parameters: { 
          timeperiod: period, 
          nbdevup: stdDev, 
          nbdevdn: stdDev 
        },
        inputs: ['close'],
        outputs: ['upperband', 'middleband', 'lowerband']
      },
      data
    });

    if (typeof result.values === 'object' && !Array.isArray(result.values)) {
      return {
        upper: result.values.upperband || [],
        middle: result.values.middleband || [],
        lower: result.values.lowerband || []
      };
    }

    return { upper: [], middle: [], lower: [] };
  }
}