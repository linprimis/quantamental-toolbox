# quantamental-toolbox

基本面量化分析和实证资产定价分析中常用的一些基础数据处理函数和类

- [quantamental-toolbox](#quantamental-toolbox)
  - [1. 基本工具类](#1-基本工具类)
    - [1. ​parallelmap并行计算处理](#1-parallelmap并行计算处理)
    - [2. ​extpandas Pandas拓展](#2-extpandas-pandas拓展)
  - [2. 未来收益率计算器](#2-未来收益率计算器)
    - [1. 未来M到N（N \>= M \>= 1）日，共 N-M+1 个交易日的累计收益率ret\_pM\_pN：](#1-未来m到nn--m--1日共-n-m1-个交易日的累计收益率ret_pm_pn)
    - [2. 固定日期月收益率ret\_m\_K：](#2-固定日期月收益率ret_m_k)
  - [3. 财务报告数据处理](#3-财务报告数据处理)



## 1. 基本工具类

### 1. ​parallelmap并行计算处理

拓展了普通多进程功能，提供更强大的map和starmap的接口。

   - 提供使用多线程/进程计算和进度条的map、starmap接口，方便项目中快速调用

   - 使用[joblib/loky](https://github.com/joblib/loky)提供的进程池接口，实现jupyter notebook中可调用，支持更多类型参数传入

   - 实现多进程timeout，可设置单个任务超时时间，超时任务自动终止

   - 示例：

     ```python
     import parallelmap
     import time
     ## 多进程并行执行function：
     def example(x):
         time.sleep(3)
         return x**2
     
     ## 基于concurrent.futures的实现，不支持notebook
     parallelmap.map_process_raw(example,[1,5,6,2,3,6,2,-3,2],max_workers=8)
     
     ## 基于joblib/loky的实现，支持notebook，不支持timeout
     parallelmap.map_loky_raw(example,[1,5,6,2,3,6,2,-3,2],max_workers=8)
     
     ## 支持timeout，功能最强大，但有超时处理过程的开销
     parallelmap.map_loky(example,[1,5,6,2,3,6,2,-3,2],max_workers=8,timeout=1)
     ```

     

### 2. ​extpandas Pandas拓展

拓展研究分析中常用的groupby / rolling - apply功能。

   - **parallel_groupby_apply(grouped, func, ******kwargs) ** 提供group层面多进程并行的groupby-apply功能

   - **rolling_apply(rolling, func, ******kwargs)** 中apply函数可操作整个数据框的多列，便于计算而pandas原版rolling-apply只能操作一列，且加进度条

   - **parallel_rolling_apply(rolling, func, ******kwargs)**** 提供多进程并行滚动计算功能

   - 示例：

     ```python
     ## 多进程分组因子标准化，传入grouped和函数
     parallel_groupby_apply(df.groupby('date')['factor'],lambda x:(x-x.mean())/x.std())
     
     ## 多进程滚动回归,传入rolling和函数
     import statsmodels.api as sm
     parallel_rolling_apply(df['y','x1','x2'].rolling(60,min_period=30),
                            lambda data:sm.OLS(data['y'],data['x1','x2']).fit().params)
     ```
     
     

## 2. 未来收益率计算器

收益率计算器类**RetCalc**位于**retutils**中，实现两种未来收益率的计算。

通过数据初始化加载，传入带股票代码、日期、日收益率的数据框：

```python
from retutils import RetCalc
calculator=RetCalc(ret_data,
                   calendar_path='SSE.csv',	## 不计算月收益就不用传入
                   symbol_col='symbol',
                   date_col='date',
                   ret_col='ret')
```

未来收益率计算的常用两种方法实现：

### 1. 未来M到N（N >= M >= 1）日，共 N-M+1 个交易日的累计收益率ret_pM_pN：

   - **计算方法**：对股票i, 时间t，计算股票i的t+M(含)到t+N(含)交易日期间累计收益率。例如ret_p2_p5表示股票在t+2,t+3,t+4,t+5四天的累计收益率。

   - **停牌处理**：

     (1) 若t日当天停牌，则t日数据中股票i被剔除；

     (2) t+M到t+N之间的停牌被忽略，计算天数固定为N-M+1天整。

   - **结果shape**：计算数据不足的以NaN填充，计算结果shape与原始日收益数据相同。

   - **输入输出要求**：输入需至少包含股票代码、日期、收益率；每个股票需按照日期升序。

   - 示例：

     ```python
     ## 在内部计算未来第3到第5天累计收益率
     calculator.ret_pM_pN(3,5,inplace=True)
     ## 修改多进程核数
     calculator.ret_pM_pN(3,5,inplace=True,max_workers=8)
     ## 不修改原数据
     calculator.ret_pM_pN(3,5,inplace=False,max_workers=8)
     ```

     

### 2. 固定日期月收益率ret_m_K：

   - **计算时段**：对股票i, 时间t, 指定月中日期key_day(如15日)：计算从每月**不早于key_day日**的最近交易日(含key_day日)开始，到下个月**key_day日之前**的最后交易日(不含key_day日)期间的累计收益率。

   - **停牌处理**：

     (1) 若股票i在开始计算的交易日当天停牌，则股票i被剔除；

     (2) 若股票i在结束计算的交易日停牌，则再向前累计计算到复盘首日；

     (3) 若停牌后复牌首日恰好为开始计算的交易日，则该日既是上月结束，又是本月开始

   - **结果shape**：计算结果变为月频数据，数据量减少；日频数据中开始第一个月日期在key_day之后、结束的最后一个月日期在key_day之前的两头数据忽略，只计算完整的月份数据。

   - **输入输出要求**：输入需至少包含股票代码、日期、收益率；每个股票需按照日期升序。同时需要有对应交易日日历文件。为什么需要日历文件而不是简单按照key_day分割计算：分割点前夕停牌股票需要到复牌方可卖出，所以需使用交易日历辅助停牌判断，使结果可交易。

   - 示例：

     ```python
     ## 以每月1日为起始点，计算各个股票月收益率
     calculator.ret_m_K(1)
     ## 以每月5日为起始点，计算各个股票月收益率
     calculator.ret_m_K(5)
     ```
     
     

## 3. 财务报告数据处理

待完善