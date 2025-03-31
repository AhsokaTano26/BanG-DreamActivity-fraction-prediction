import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class BestdoriTracker:
    """Bestdori API 数据获取封装"""

    def __init__(self, server: str, event_id: int, me: Optional[Me] = None):
        self.tracker = EventTracker(server=server, event=event_id, me=me)

    def get_current_scores(self, target_rank: int) -> Tuple[List[int], List[int]]:
        """获取当前排名分数数据"""
        try:
            # 获取指定排名的追踪数据
            tracker_data = self.tracker.get_data(tier=target_rank)
            return (
                [x["rank"] for x in tracker_data["history"]],
                [x["score"] for x in tracker_data["history"]]
            )
        except NotExistException:
            # 降级获取T10数据
            top_data = self.tracker.get_top(interval=3600)
            return (
                [x["rank"] for x in top_data["rankings"]],
                [x["score"] for x in top_data["rankings"]]
            )

    def get_event_end_time(self) -> datetime:
        """获取活动结束时间"""
        event_info = events.get(self.tracker.event)
        return datetime.fromisoformat(event_info["endAt"][:-1])  # 移除末尾的Z


class BangDreamPredictor:
    """改进版活动分数线预测器"""

    def __init__(self, server: str, event_id: int, me: Optional[Me] = None):
        self.tracker = BestdoriTracker(server, event_id, me)
        self.event_end = self.tracker.get_event_end_time()

    def power_law(self, x: float, a: float, b: float) -> float:
        """幂函数模型"""
        return a * np.power(x, b)

    def time_decay_factor(self, remain_seconds: float) -> float:
        """动态时间衰减系数"""
        total_duration = (self.event_end - datetime.now()).total_seconds()
        if remain_seconds > 6 * 3600:
            return 1.0
        return 1.0 + 0.5 * (6 - remain_seconds / 3600) / 6

    def predict(self, target_rank: int) -> int:
        """综合预测主函数"""
        # 获取实时数据
        ranks, scores = self.tracker.get_current_scores(target_rank)
        ranks = np.array(ranks)
        scores = np.array(scores)

        # 数据预处理
        valid_idx = scores > 0
        ranks = ranks[valid_idx]
        scores = scores[valid_idx]

        # 分区间建模
        if target_rank <= 100:
            return self._predict_top_ranks(ranks, scores, target_rank)
        return self._predict_normal_ranks(ranks, scores, target_rank)

    def _predict_top_ranks(self, ranks: np.ndarray, scores: np.ndarray, target: int) -> int:
        """头部排名预测（二次多项式拟合）"""
        coeffs = np.polyfit(ranks, scores, 2)
        return int(np.polyval(coeffs, target))

    def _predict_normal_ranks(self, ranks: np.ndarray, scores: np.ndarray, target: int) -> int:
        """常规排名预测（幂函数+时间修正）"""
        try:
            popt, _ = curve_fit(self.power_law, ranks, scores, maxfev=2000)
            base_pred = self.power_law(target, *popt)
        except RuntimeError:
            # 拟合失败时使用线性回归
            coeffs = np.polyfit(ranks, np.log(scores), 1)
            base_pred = np.exp(coeffs[1] + coeffs[0] * target)

        remain_seconds = (self.event_end - datetime.now()).total_seconds()
        return int(base_pred * self.time_decay_factor(remain_seconds))

    def plot_prediction(self, target_rank: int):
        """可视化预测曲线"""
        ranks, scores = self.tracker.get_current_scores(target_rank)

        plt.figure(figsize=(12, 6))
        x_range = np.linspace(min(ranks), max(ranks) * 2, 100)

        # 绘制预测曲线
        y_pred = [self.predict(int(x)) for x in x_range]
        plt.plot(x_range, y_pred, label='预测曲线', color='navy')

        # 绘制实际数据点
        plt.scatter(ranks, scores, color='red', label='实时数据')

        # 标注目标排名
        target_score = self.predict(target_rank)
        plt.scatter([target_rank], [target_score],
                    color='gold', s=200, label=f'预测 {target_rank}名')

        plt.title(f'活动 {self.tracker.tracker.event} 分数线预测', fontsize=14)
        plt.xlabel('排名', fontsize=12)
        plt.ylabel('分数', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.ticklabel_format(axis='y', style='plain')
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 初始化认证信息（根据需要）
    from bestdori import Me

    me = Me(cookies={"session": "your_session_cookie"})

    # 创建预测器（参数：服务器、活动ID、认证信息）
    predictor = BangDreamPredictor('jp', 100, me=me)

    # 进行预测并可视化
    target_rank = 200
    print(f"预测分数线：{predictor.predict(target_rank):,}")
    predictor.plot_prediction(target_rank)