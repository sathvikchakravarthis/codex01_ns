from datetime import datetime, timedelta
import unittest

from btc_bot import BotConfig, DecisionEngine, MarketSnapshot, RiskState


def sample_config() -> BotConfig:
    return BotConfig.from_dict(
        {
            "bot": {"trading_pair": "BTC/USDT", "allow_short": False},
            "risk_management": {
                "risk_per_trade_percent": 1.0,
                "max_daily_drawdown_percent": 3.0,
                "max_total_drawdown_percent": 10.0,
                "cooldown_after_loss_minutes": 30,
                "min_rr_ratio": 2.0,
            },
            "position_sizing": {
                "method": "risk_based",
                "atr_multiplier": 1.5,
                "fee_buffer_percent": 0.1,
                "slippage_buffer_percent": 0.05,
            },
            "indicators": {
                "ema": {"fast": 20, "slow": 50, "trend_filter": 200},
                "rsi": {"period": 14, "overbought": 70, "oversold": 30},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "atr": {"period": 14},
                "volume": {"ma_period": 20, "spike_multiplier": 1.5},
            },
            "entry_rules": {
                "long": {
                    "trend_confirmation": True,
                    "momentum_confirmation": True,
                    "volume_confirmation": True,
                    "rsi_min": 40,
                    "rsi_max": 65,
                }
            },
            "exit_rules": {
                "stop_loss": {"type": "atr", "atr_multiplier": 1.5},
                "take_profit": {"type": "rr_ratio", "rr_ratio": 2.5},
                "trailing_stop": {
                    "enabled": True,
                    "activation_rr": 1.5,
                    "trail_atr_multiplier": 1.0,
                },
            },
        }
    )


class DecisionEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = DecisionEngine(sample_config())
        self.now = datetime.utcnow()

    def snapshot(self, **kwargs):
        base = dict(
            timestamp=self.now,
            close=65000,
            ema_fast=64800,
            ema_slow=64500,
            ema_trend=64000,
            rsi=55,
            macd=40,
            macd_signal=20,
            atr=400,
            volume=1800,
            volume_ma=1000,
            spread_percent=0.08,
            data_ok=True,
            major_news=False,
        )
        base.update(kwargs)
        return MarketSnapshot(**base)

    def state(self, **kwargs):
        base = dict(equity=10000, daily_drawdown_percent=0.5, total_drawdown_percent=1.0, last_loss_timestamp=None)
        base.update(kwargs)
        return RiskState(**base)

    def test_buy_when_all_confluence_passes(self):
        plan = self.engine.evaluate(self.snapshot(), self.state())
        self.assertEqual(plan.action, "BUY")
        self.assertIsNotNone(plan.quantity_btc)
        self.assertGreater(plan.take_profit, plan.entry_price)

    def test_halt_when_daily_drawdown_hit(self):
        plan = self.engine.evaluate(self.snapshot(), self.state(daily_drawdown_percent=3.0))
        self.assertEqual(plan.action, "HALT")

    def test_hold_during_cooldown(self):
        plan = self.engine.evaluate(
            self.snapshot(),
            self.state(last_loss_timestamp=self.now - timedelta(minutes=10)),
        )
        self.assertEqual(plan.action, "HOLD")


if __name__ == "__main__":
    unittest.main()
