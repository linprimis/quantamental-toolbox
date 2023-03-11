import pandas as pd
import numpy as np
import datetime
import parallelmap
import parallelpandas


class RetCalc:
    def __init__(self,
                 data: pd.DataFrame,
                 calendar_path: str = 'SSE.csv',
                 symbol_col: str = 'symbol',
                 date_col: str = 'date',
                 ret_col: str = 'ret'):
        """
        :param data: returns data，including columns "symbol", "date" and "ret",
        """
        self.data = data
        self.calendar = None
        self.grouped = None

        self.calendar_path = calendar_path
        self.start_end = {}

        self.symbol_col = symbol_col
        self.date_col = date_col
        self.ret_col = ret_col

    def _load_calendar(self):
        ## load trading day calendar
        self.calendar = pd.read_csv(self.calendar_path, header=None, parse_dates=[0])[0]

    def _gen_grouped(self):
        ## group the data by symbol before the calculation of stock returns
        self.grouped = self.data.groupby(self.symbol_col)
        self.symbols = self.data[self.symbol_col].unique()

    ###############################################
    ##  Calculating the cumulative return rate   ##
    #   for multiple trading days in the future  ##
    ###############################################

    ###############################
    ##  计算未来一段时间的累计收益率  ##
    ###############################

    @classmethod
    def ret_pM_pN_core(cls, grouped, M=1, N=5, **kwargs):
        return parallelpandas.parallel_groupby_apply(grouped, lambda x: x.iloc[::-1].rolling(N - M + 1).apply(
            lambda x: (1 + x).prod() - 1).shift(M).iloc[::-1], **kwargs)

    def ret_pM_pN(self, M, N, inplace=False, **kwargs):
        ## N must be greater than M
        assert N >= M >= 1
        ## If the group is not loaded, it will be loaded first
        ## 如果未加载分组，则先加载分组
        if self.grouped is None:
            self._gen_grouped()

        ## When inplace is True, add the response column in the source data; otherwise return the result column
        ## 当inplace为True时，在源数据中添加响应列；否则返回结果列
        if inplace:
            self.data[f'ret_p{M}_p{N}'] = self.ret_pM_pN_core(self.grouped['ret'], M, N, **kwargs)
            return
        else:
            return self.ret_pM_pN_core(self.grouped['ret'], M, N, **kwargs)

    ##############################################
    ##  Calculate monthly returns, you can      ##
    ##  customize the start date of each month  ##
    ##############################################

    ######################################################
    ##  计算月度收益率，可自定义每个月的开始日期，增强分析稳健性  ##
    ######################################################
    def _key_day_mask(self, x: pd.DataFrame, key_day: int):
        ## check if the date is the key day(the seperation day of month)
        ## 检查日期是否为关键日期（月份分割日）
        d: pd.Series = x['calendar']
        ## the previous trading day of current date
        ## 当前交易日的前一个交易日
        d_pre: pd.Series = x['calendar_pre']
        key_day_datetime = datetime.datetime(d.year, d.month, key_day)
        return d >= key_day_datetime and d_pre < key_day_datetime

    def _gen_start_end(self, T):
        ## generate the start and end date DataFrame of each month according to the key day setting
        ## 根据自定义月起始日设置生成每个月的开始日期和结束日期DataFrame

        ## if the calendar is not loaded, load it first
        ## 如果没有加载交易日日历则先加载，同时加载交易日和上一个交易日
        if self.calendar is None:
            self._load_calendar()

        calendar_df = pd.DataFrame({'calendar': self.calendar, 'calendar_pre': self.calendar.shift(1)})

        ## select the start and end date for each month
        ## 筛选出每个月开始日和结束日
        calendar_df = calendar_df[calendar_df.apply(lambda x: self._key_day_mask(x, T), axis=1)]
        calendar_df['calendar_pre'] = calendar_df['calendar_pre'].shift(-1)
        calendar_df.columns = ['start', 'end']

        ## generate and store in dictionary
        ## 生成后存储在字典中
        self.start_end[T] = calendar_df
        return calendar_df

    def get_start_end(self, key_day):
        ## 获取每个月的开始日和结束日配对
        ## key_day: the start day of each month
        ## 检查是否已经计算过key_day字典，如果没有存储则先生成
        if key_day in self.start_end:
            return self.start_end[key_day]
        else:
            return self._gen_start_end(key_day)

    @classmethod
    def ret_m_K_core(cls,
                     df: pd.DataFrame,
                     calendar_df: pd.DataFrame,
                     key_day: int,
                     symbol_col: str = 'symbol',
                     date_col: str = 'date',
                     ret_col: str = 'ret'):
        ## 对单股票计算
        ## 输入：单股票Dataframe，包含单个股票代码、日期和收益率，按日期升序排列
        ## 输出：单股票Dataframe，包含单个股票代码、日期、收益率、月度收益率、月度开始日期、月度结束日期

        ## calculate for single stock
        ## input: single stock Dataframe, including single stock code, date and return, sorted by date in ascending order
        ## output: single stock Dataframe, including single stock code, date, return, monthly return, monthly start date, monthly end date

        ## 配对关键日期，使得每日对应最近的关键日期
        ## pairing key date to make each day correspond to the nearest key date
        df = pd.merge_asof(left=df, right=calendar_df, left_on=date_col, right_on='start')

        ## store the ordinal index of the key column
        idx_symbol = df.columns.get_loc(symbol_col)
        idx_date = df.columns.get_loc(date_col)
        idx_ret = df.columns.get_loc(ret_col)
        idx_start = df.columns.get_loc('start')
        idx_end = df.columns.get_loc('end')

        results = []
        ## 使用双指标遍历循环, i指向当前日期，j指向当前日期后的第一个关键日期
        ## i points to the current date, j points to the first key date after the current date
        i = j = 0
        while i < df.shape[0] and j < df.shape[0]:
            ## 精确匹配起始日期，当日停牌则无数据

            ## 如果i指向的当前日期与关键日期匹配，则将j指向当前日期，然后j开始往后遍历，直到不早于本“月”结束时间，计算累计收益率
            ## if the current date pointed to by i matches the key date, then j points to the current date,
            # and then j starts to traverse forward until it is not earlier than the end date of the current month,
            # and the cumulative return rate is calculated
            if df.iat[i, idx_date] == df.iat[i, idx_start]:
                j = i  ## 起始日匹配时（未停牌），将j指向当前日期, j points to the current date when the start date matches (not suspended)
                while j < df.shape[0]:  ## j独立往下循环，直到不早于本“月”结束时间，计算累计收益率，并插入相关信息
                    if df.iat[j, idx_date] >= df.iat[i, idx_end]:
                        results.append((df.iat[i, idx_symbol],  ## 股票代码, symbol
                                        df.iat[i, idx_date],  ## ”月“收益统计开始日期, start date
                                        df.iat[j, idx_date],  ## ”月“收益统计实际结束日期, end date
                                        df.iat[i, idx_end],  ## ”月“收益统计预期结束日期, end date expected
                                        (df.iloc[i:j + 1, idx_ret] + 1).prod(
                                            skipna=True) - 1)),  ## ”月“收益率, monthly return
                        i = j  ## 更新指标i，i跳过本“月”已经计算过的日期, update i, i skips the dates that have been calculated in the current month
                        break
                    j += 1
            else:
                i += 1
        return pd.DataFrame(results, columns=[symbol_col, 'start', 'end', 'end_expected', f'ret_m_{key_day}'])

    def ret_m_K(self, key_day: int, **kwargs):
        ## 计算月度收益率，通过key_day指定月度起始日
        ## calculate monthly return rate, specify monthly start date through key_day
        ## input: key_day: the start day of each month

        assert 1 <= key_day <= 28

        ## 若没有生成过分组，先生成分组
        ## if the grouping has not been generated, generate the grouping first
        if self.grouped is None:
            self._gen_grouped()

        ## 获取”月“起止日期，并以多线程计算
        calendar_df = self.get_start_end(key_day)
        datas = parallelpandas.parallel_groupby_apply(self.grouped,
                                                      lambda group: self.ret_m_K_core(df=group,
                                                                                      calendar_df=calendar_df,
                                                                                      key_day=key_day,
                                                                                      symbol_col=self.symbol_col,
                                                                                      date_col=self.date_col,
                                                                                      ret_col=self.ret_col),
                                                      **kwargs)
        return datas.reset_index(drop=True)
