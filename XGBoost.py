import pandas as pd
import numpy as np
import json
import ast
from datetime import timedelta
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from API.GetEventTracker import event_tracker
from API.GetEventInfo import get_event_info
from sklearn.metrics import mean_squared_error, r2_score

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
    解析数据，提取特征（持续时间、分类、时间），并将目标变量
    设置为活动结束时的总分数 (Total EP)。
    """

    # 1. 目标变量：活动结束时的最终分数
    # 初始化目标 EP 列
    df['Target_Total_EP'] = 0.0

    for index, row in df.iterrows():
        try:
            point_data = ast.literal_eval(row['PointRank'])
            if not point_data:
                continue

            # 最终 EP 是 PointRank 列表中的最后一个 'ep' 值
            final_ep = point_data[-1]['ep']
            df.loc[index, 'Target_Total_EP'] = float(final_ep)

        except Exception as e:
            # print(f"处理活动 ID={row['ID']} 时的错误: {e}")
            continue

    # 2. 特征：活动持续时间 (Duration)
    # 持续时间 (秒) 是最重要的特征
    df['Duration_S'] = (df['EndAt'] - df['StartAt']) / 1000

    # 3. 处理分类特征 (One-Hot Encoding)
    categorical_features = ['EventBand', 'EventType', 'Country']
    df_encoded = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features).fillna(0)

    # 4. 提取时间特征
    df_encoded['Start_Time'] = pd.to_datetime(df_encoded['StartAt'], unit='ms')
    df_encoded['DayOfWeek'] = df_encoded['Start_Time'].dt.dayofweek.astype(np.int64)
    df_encoded['HourOfDay'] = df_encoded['Start_Time'].dt.hour.astype(np.int64)

    return df_encoded


# --- 完整程序主流程 (修改为预测 EP) ---

def run_ep_prediction_pipeline(session):
    """
    执行数据加载、特征工程、模型训练和评估的完整流程（目标：预测 EP）。
    """
    # ----------------------------------------------------
    # A. 数据加载
    # ----------------------------------------------------
    try:
        query = session.query(Event)
        query = query.filter(Event.Rank == 2000).filter(Event.EventID >= 250)
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
    # 使用新的特征提取函数
    df_features = parse_and_extract_features_for_ep_prediction(df_raw)

    # 定义目标列和特征列
    target_col = 'Target_Total_EP'

    # 明确定义数值/时间特征
    base_features = ['Duration_S', 'DayOfWeek', 'HourOfDay']

    # 自动获取 One-Hot 编码特征
    one_hot_cols = [
        col for col in df_features.columns
        if col.startswith(('EventBand_', 'EventType_', 'Country_'))
    ]

    # 合并所有特征列
    feature_cols = base_features + one_hot_cols
    feature_cols = [col for col in feature_cols if col in df_features.columns]

    X = df_features[feature_cols]
    y = df_features[target_col]

    print(f"✅ 特征工程完成。选定特征数量: {len(feature_cols)}")
    print(f"使用的特征列表: {feature_cols}")

    # ----------------------------------------------------
    # C. 模型训练
    # ----------------------------------------------------
    print("🤖 开始训练 XGBoost 模型...")
    if X.empty:
        print("❌ 训练集为空，无法训练模型。")
        return None, []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(X_train, y_train)
    print("✅ 模型训练完成。")

    # ----------------------------------------------------
    # D. 模型评估
    # ----------------------------------------------------
    y_pred = xgb_model.predict(X_test)

    # 评估指标改为 MSE 和 R2，但衡量的是 EP 值的差异
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n--- 📊 模型评估结果 ---")
    print(f"均方根误差 (RMSE): {rmse:.2f} EP")
    print(f"决定系数 (R-squared): {r2:.4f}")

    # 打印特征重要性
    importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
    print("\n--- 🔍 最重要的 5 个特征 ---")
    print(importances.nlargest(5))

    return xgb_model, X.columns


# ----------------------------------------------------
# E. 运行整个程序
# ----------------------------------------------------

if __name__ == '__main__':
    session = Session()
    Country_list = ["日本", "国际", "中国台湾", "中国大陆", "韩国"]

    # 训练模型并获取特征列表
    model, feature_names = run_ep_prediction_pipeline(session)

    # --- 预测新活动总分数 ---

    if model is not None and len(feature_names) > 0:
        print("\n--- 🔮 新活动总分数预测 ---")


        # 实际 API 调用（使用 Mock 函数代替）
        # 假设活动 ID 293，国家 ID 3
        info, startAt, endAt = get_event_info(Activity=293, Country=3)

        # 准备新活动的特征
        new_activity_start_at = int(startAt)
        new_activity_end_at = int(endAt)
        new_activity_band = 'B'  # 假设是 B 团
        new_activity_type = info['eventType']
        new_activity_country = Country_list[3]  # 中国大陆

        # 1. 创建新活动的特征 DataFrame
        new_X = pd.DataFrame(0.0, index=[0], columns=feature_names)

        # 2. 填充数值特征：持续时间
        duration_s = (new_activity_end_at - new_activity_start_at) / 1000
        if 'Duration_S' in feature_names:
            new_X.loc[0, 'Duration_S'] = duration_s

        # 3. 填充时间特征
        start_time_dt = pd.to_datetime(int(new_activity_start_at), unit='ms')

        if 'DayOfWeek' in feature_names:
            new_X.loc[0, 'DayOfWeek'] = start_time_dt.dayofweek
        if 'HourOfDay' in feature_names:
            new_X.loc[0, 'HourOfDay'] = start_time_dt.hour

        # 4. 填充 One-Hot 编码特征
        band_col = f'EventBand_{new_activity_band}'
        type_col = f'EventType_{new_activity_type}'
        country_col = f'Country_{new_activity_country}'

        if band_col in feature_names:
            new_X.loc[0, band_col] = 1.0
        if type_col in feature_names:
            new_X.loc[0, type_col] = 1.0
        if country_col in feature_names:
            new_X.loc[0, country_col] = 1.0

        # 5. 预测
        predicted_ep = model.predict(new_X)[0]  # 这是一个 numpy.float32/64 类型

        print(f"活动开始时间: {start_time_dt}")
        print(f"活动结束时间: {pd.to_datetime(int(new_activity_end_at), unit='ms')}")
        print(f"活动持续时间: {duration_s / 3600:.2f} 小时 ({duration_s:.0f} 秒)")
        print(f"---")
        print(f"预测活动总分数 (Total EP): {predicted_ep.item():.0f}")

    session.close()