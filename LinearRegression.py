import pandas as pd
import numpy as np
import ast
from datetime import timedelta
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # 引入线性回归模型
from sklearn.metrics import mean_squared_error, r2_score
# 假设这些 API 导入是正确的
from API.GetEventTracker import event_tracker
from API.GetEventInfo import get_event_info

# --- 数据库 ORM 定义 (保持不变) ---
Base = declarative_base()


class Event(Base):
    __tablename__ = 'event'
    ID = Column(String(100), primary_key=True, nullable=False)
    EventID = Column(Integer)
    EventBand = Column(String(100))
    EventName = Column(String(100))
    EventType = Column(String(100))
    StartAt = Column(Integer)  # Unix 时间戳 (ms)
    EndAt = Column(Integer)  # Unix 时间戳 (ms)
    Rank = Column(Integer)
    PointRank = Column(String(100000))  # 存储 [{time, ep}, ...] 列表的字符串
    Country = Column(String(100))


# --- 数据库配置 ---
DATABASE_URL = "sqlite:///data/db.sqlite3"
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)


def parse_and_extract_features_for_ep_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    解析数据，提取特征：总持续时间、平均EP增长率、时间特征、分类特征。
    目标变量：活动结束时的总分数 (Total EP)。
    """

    # 初始化目标 EP 列和特征列
    df['Target_Total_EP'] = 0.0
    df['Duration_S'] = 0.0
    # 新增核心特征：历史平均 EP 增长率 (EP/秒)
    df['Avg_EP_Rate'] = 0.0

    for index, row in df.iterrows():
        try:
            point_data = ast.literal_eval(row['PointRank'])
            if not point_data:
                continue

            # 1. 目标变量：最终分数
            final_ep = point_data[-1]['ep']
            df.loc[index, 'Target_Total_EP'] = float(final_ep)

            # 2. 特征：总持续时间和平均增长率
            duration_ms = row['EndAt'] - row['StartAt']
            duration_s = duration_ms / 1000

            df.loc[index, 'Duration_S'] = duration_s

            if duration_s > 0:
                # 计算整个活动的平均 EP 增长率
                df.loc[index, 'Avg_EP_Rate'] = final_ep / duration_s

        except Exception as e:
            # print(f"处理活动 ID={row['ID']} 时的错误: {e}")
            continue

    # 3. 处理分类特征 (One-Hot Encoding)
    categorical_features = ['EventBand', 'EventType', 'Country']
    df_encoded = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features).fillna(0)

    # 4. 提取时间特征
    df_encoded['Start_Time'] = pd.to_datetime(df_encoded['StartAt'], unit='ms')
    df_encoded['DayOfWeek'] = df_encoded['Start_Time'].dt.dayofweek.astype(np.int64)
    df_encoded['HourOfDay'] = df_encoded['Start_Time'].dt.hour.astype(np.int64)

    return df_encoded


# --- 完整程序主流程 (改为线性回归模型) ---

def run_ep_prediction_pipeline(session, Rank, Use_ID):
    """
    执行数据加载、特征工程、模型训练和评估的完整流程（目标：预测 EP）。
    """
    # ----------------------------------------------------
    # A. 数据加载
    # ----------------------------------------------------
    try:
        query = session.query(Event)
        query = query.filter(Event.Rank == Rank).filter(Event.EventID >= Use_ID)
        data = [{c.name: getattr(e, c.name) for c in e.__table__.columns} for e in query]
        df_raw = pd.DataFrame(data)

        if df_raw.empty:
            print("🚨 数据库中未找到任何活动数据。")
            return None, []

        print(f"✅ 成功加载 {len(df_raw)} 条活动数据。")

    except Exception as e:
        print(f"❌ 数据库查询失败: {e}")
        return None, []

    # ----------------------------------------------------
    # B. 特征工程
    # ----------------------------------------------------
    print("🔧 开始特征工程...")
    df_features = parse_and_extract_features_for_ep_prediction(df_raw)

    target_col = 'Target_Total_EP'

    # 核心特征改为 Avg_EP_Rate 和 Duration_S
    base_features = ['Duration_S', 'Avg_EP_Rate', 'DayOfWeek', 'HourOfDay']

    one_hot_cols = [
        col for col in df_features.columns
        if col.startswith(('EventBand_', 'EventType_', 'Country_'))
    ]

    feature_cols = base_features + one_hot_cols
    feature_cols = [col for col in feature_cols if col in df_features.columns]

    X = df_features[feature_cols]
    y = df_features[target_col]

    print(f"✅ 特征工程完成。选定特征数量: {len(feature_cols)}")
    print(f"使用的特征列表: {feature_cols}")

    # ----------------------------------------------------
    # C. 模型训练 (使用线性回归)
    # ----------------------------------------------------
    print("🤖 开始训练线性回归模型...")
    if X.empty:
        print("❌ 训练集为空，无法训练模型。")
        return None, []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 线性回归模型
    linear_model = LinearRegression()

    linear_model.fit(X_train, y_train)
    print("✅ 模型训练完成。")

    # ----------------------------------------------------
    # D. 模型评估
    # ----------------------------------------------------
    y_pred = linear_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n--- 📊 模型评估结果 ---")
    print(f"均方根误差 (RMSE): {rmse:.2f} EP")
    print(f"决定系数 (R-squared): {r2:.4f}")

    # 打印特征系数（线性回归中替代特征重要性）
    coefficients = pd.Series(linear_model.coef_, index=X.columns).abs().sort_values(ascending=False)
    print("\n--- 🔍 最重要的 5 个特征系数 ---")
    print(coefficients.nlargest(5))

    return linear_model, X.columns

def get_input(prompt, default_value):
    user_input = input(prompt)
    if not user_input:
        user_input = default_value
    return user_input
# ----------------------------------------------------
# E. 运行整个程序
# ----------------------------------------------------

if __name__ == '__main__':
    Activity = int(input("请输入活动ID："))
    Rank = int(input("请输入预测分数线："))
    Use_ID = int(get_input("请输入使用多少次活动以后的数据训练：",226))
    session = Session()
    Country_list = ["日本", "国际", "中国台湾", "中国大陆", "韩国"]

    # 训练模型并获取特征列表
    model, feature_names = run_ep_prediction_pipeline(session,Rank,Use_ID)

    # --- 预测新活动总分数 ---

    if model is not None and len(feature_names) > 0:
        print("\n--- 🔮 新活动总分数预测 ---")

        # 实际 API 调用
        info, startAt, endAt = get_event_info(Activity=Activity, Country=3)
        Point = event_tracker(Country=3, Activity=Activity, Rank=Rank)

        # 准备新活动的特征
        new_activity_start_at = int(startAt)
        new_activity_end_at = int(endAt)
        new_activity_type = info['eventType']
        new_activity_country = '中国大陆'  # 假设国家

        # 1. 计算活动持续时间特征
        duration_s = (new_activity_end_at - new_activity_start_at) / 1000

        # 2. 核心：计算新活动的“理论平均 EP 增长率”
        # 假设我们使用 PointRank 的前几个点来计算当前的增长率，
        # 并用这个 CURRENT_RATE 来代替历史的 Avg_EP_Rate 进行预测。

        # 提取当前测量的 EP 点 (仅使用 event_tracker 得到的分数点)
        if Point and len(Point) > 1:
            # 最后一个测量点的时间和分数
            latest_time = Point[-1]['time']
            latest_ep = Point[-1]['ep']

            # 从活动开始到最新测量点的时间差（秒）
            current_measured_duration_s = (latest_time - new_activity_start_at) / 1000

            if current_measured_duration_s > 0:
                # 使用当前观测到的增长率作为预测特征（即，假设当前增速保持不变）
                current_ep_rate = latest_ep / current_measured_duration_s
            else:
                current_ep_rate = 0.0
        else:
            print("⚠️ 警告：当前无足够的 PointRank 数据来计算增长率，使用历史平均率0。")
            current_ep_rate = 0.0

        # 3. 创建特征 DataFrame
        new_X = pd.DataFrame(0.0, index=[0], columns=feature_names)

        # 4. 填充特征
        if 'Duration_S' in feature_names:
            new_X.loc[0, 'Duration_S'] = duration_s
        if 'Avg_EP_Rate' in feature_names:
            # 使用当前观测到的增长率作为预测输入
            new_X.loc[0, 'Avg_EP_Rate'] = current_ep_rate

            # 填充时间特征
        start_time_dt = pd.to_datetime(int(new_activity_start_at), unit='ms')
        if 'DayOfWeek' in feature_names:
            new_X.loc[0, 'DayOfWeek'] = start_time_dt.dayofweek
        if 'HourOfDay' in feature_names:
            new_X.loc[0, 'HourOfDay'] = start_time_dt.hour

        # 填充 One-Hot 编码特征 (需要手动提供 EventBand)
        new_activity_band = 'B'  # 假设 EventBand
        band_col = f'EventBand_{new_activity_band}'
        type_col = f'EventType_{new_activity_type}'
        country_col = f'Country_{new_activity_country}'

        for col in [band_col, type_col, country_col]:
            if col in feature_names:
                new_X.loc[0, col] = 1.0

        # 5. 预测
        predicted_ep = model.predict(new_X)[0]

        print(f"活动开始时间: {start_time_dt}")
        print(f"活动持续时间: {duration_s / 3600:.2f} 小时 ({duration_s:.0f} 秒)")
        print(f"当前观测到的 EP 增长率: {current_ep_rate:.2f} EP/s")
        print(f"---")
        print(f"预测活动总分数 (Total EP): {predicted_ep.item():.0f}")

    session.close()